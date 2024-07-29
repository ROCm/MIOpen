/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "mha_common.hpp"

#include <miopen/mha/solvers.hpp>

#include <miopen/mha/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#include <algorithm>
#include <vector>
#include <tuple>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_NAIVE_FWD)

namespace miopen {

namespace solver {

namespace mha {

namespace { // TODO: Issue #2748
MultiBufferWorkspaceTraits
SplitBufferToWorkspace(size_t S, size_t D, size_t NHS, miopenDataType_t out_type)
{
    // the first MatMul (N*H*S*D) * (N*H*S*D)T = (N*H*S*S)
    // the second MatMul (N*H*S*S) * (N*H*S*D) = (N*H*S*D)
    return MultiBufferWorkspaceTraits{
        NHS * std::max(S, D) * get_data_size(miopenFloat), // first and second matmuls tensor
        NHS * S * get_data_size(out_type)};                // first matmul tensor
}

MultiBufferWorkspaceTraits SplitBufferToWorkspace(const std::vector<size_t>& lengths,
                                                  miopenDataType_t out_type)
{
    const auto [N, H, S, D] = miopen::tien<4>(lengths);
    return SplitBufferToWorkspace(S, D, N * H * S, out_type);
}
} // namespace

bool MhaForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                              const miopen::mha::ProblemDescription& problem) const
{
#if MIOPEN_USE_ROCBLAS
    // It's important to have this check before problem.GetDescsForward() call
    if(!problem.IsForward())
    {
        return false;
    }

    const miopen::mha::MhaInputDescsForward& descsFwd = problem.GetDescsForward();

    auto [N, H, S, D] = miopen::tien<4>(descsFwd.kDesc.GetLengths());

    return !env::disabled(MIOPEN_DEBUG_ATTN_NAIVE_FWD)              //
           && S <= std::numeric_limits<uint32_t>::max()             //
           && descsFwd.kDesc.IsPacked()                             //
           && descsFwd.qDesc.IsPacked()                             //
           && descsFwd.vDesc.IsPacked()                             //
           && descsFwd.oDesc.IsPacked()                             //
           && descsFwd.mDesc.IsPacked()                             //
           && descsFwd.zInvDesc.IsPacked()                          //
           && descsFwd.mDesc.GetType() == miopenFloat               //
           && descsFwd.zInvDesc.GetType() == miopenFloat            //
           && descsFwd.kDesc.GetType() == descsFwd.qDesc.GetType()  //
           && descsFwd.kDesc.GetType() == descsFwd.vDesc.GetType()  //
           && descsFwd.kDesc.GetType() == descsFwd.oDesc.GetType()  //
           && ((descsFwd.kDesc.GetType() == miopenFloat)            //
               || (USE_ROCBLAS_EX3                                  //
                   && (MIOPEN_FP8_IEEE_EXPONENT_BIAS == 0)          //
                   && (descsFwd.kDesc.GetType() == miopenFloat8))); //
#else
    return false;
#endif
}

std::size_t MhaForward::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                         const miopen::mha::ProblemDescription& problem) const
{
    const auto& kDesc = problem.GetDescsForward().kDesc;
    return SplitBufferToWorkspace(kDesc.GetLengths(), kDesc.GetType()).GetSize();
}

ConvSolution MhaForward::GetSolution(const ExecutionContext& context,
                                     const miopen::mha::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    uint64_t N, H, S, D;
    std::tie(N, H, S, D) = miopen::tien<4>(problem.GetDescsForward().kDesc.GetLengths());
    uint32_t seq_len     = static_cast<uint32_t>(S);
    uint64_t nhs         = N * H * S;
    uint64_t nhsd        = N * H * S * D;
    float scale          = problem.GetDescsForward().scale; // just to capture it into lambda

    auto ABType = problem.GetDescsForward().kDesc.GetType();

    auto warpSize = context.GetStream().GetWavefrontWidth();

    size_t local_threads  = std::clamp(nextPow2(S), warpSize, static_cast<size_t>(256));
    size_t global_threads = nhs * local_threads;

    auto softmax_kernel         = KernelInfo{};
    softmax_kernel.comp_options = KernelBuildParameters{
        {"THREADS", local_threads},
        {"OUT_TYPE",
         GetDataType(ABType)}}.GenerateFor(kbp::HIP{});
    softmax_kernel.kernel_file = "MIOpenSoftmaxAttn.cpp";
    softmax_kernel.kernel_name = S > local_threads ? "SoftMaxCommon"
                                 : S > warpSize    ? "SoftMaxBlock"
                                                   : "SoftMaxWarp";
    if(S <= warpSize)
    {
        global_threads = Ceil(global_threads, local_threads / warpSize);
    }
    softmax_kernel.l_wk = {local_threads, 1, 1};
    softmax_kernel.g_wk = {global_threads, 1, 1};
    result.construction_params.push_back(softmax_kernel);

    auto getBuffPart = [ws = SplitBufferToWorkspace(S, D, nhs, ABType)](void* buffer,
                                                                        size_t part_idx) {
        return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
    };

    local_threads  = std::clamp(nextPow2(nhsd), warpSize, static_cast<size_t>(256));
    global_threads = RoundUpToMultiple(nhsd, local_threads);

    auto scale_reduce_kernel         = KernelInfo{};
    scale_reduce_kernel.comp_options = KernelBuildParameters{
        {"THREADS", local_threads},
        {"OUT_TYPE", GetDataType(ABType)}}.GenerateFor(kbp::HIP{});
    scale_reduce_kernel.kernel_file = "MIOpenSoftmaxAttn.cpp";
    scale_reduce_kernel.kernel_name = "ScaleReduce";
    scale_reduce_kernel.l_wk        = {local_threads, 1, 1};
    scale_reduce_kernel.g_wk        = {global_threads, 1, 1};
    result.construction_params.push_back(scale_reduce_kernel);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::mha::InvokeParams>();

            HipEventPtr start    = nullptr;
            HipEventPtr stop     = nullptr;
            const bool profiling = handle_.IsProfilingEnabled();
            const auto& dataFwd  = params.GetDataForward();

            if(profiling)
            {
                start = make_hip_event();
                stop  = make_hip_event();
                handle_.EnableProfiling(false);
                hipEventRecord(start.get(), handle_.GetStream());
            }

            // zero amax output data to use atomics
            hipMemsetAsync(dataFwd.amaxSData, 0, sizeof(float), handle_.GetStream());
            hipMemsetAsync(dataFwd.amaxOData, 0, sizeof(float), handle_.GetStream());

            void* fp32_ws = getBuffPart(params.GetWorkspace(), 0);
            void* fp8_ws  = getBuffPart(params.GetWorkspace(), 1);

            gemm(handle_,
                 false,
                 true,
                 S,
                 S,
                 D,
                 D,
                 D,
                 S,
                 N * H,
                 S * D,
                 S * D,
                 S * S,
                 scale,
                 ABType,
                 dataFwd.qData,
                 ABType,
                 dataFwd.kData,
                 fp32_ws,
                 true);

            decltype(auto) softmax_kernel = handle_.Run(kernels.front());
            softmax_kernel(fp32_ws,
                           fp8_ws,
                           dataFwd.mData,
                           dataFwd.zInvData,
                           dataFwd.amaxSData,
                           dataFwd.descaleQData,
                           dataFwd.descaleKData,
                           dataFwd.scaleSData,
                           dataFwd.dropoutSeedData,
                           dataFwd.dropoutOffsetData,
                           dataFwd.dropoutProbabilityData,
                           seq_len,
                           nhs);

            gemm(handle_,
                 false,
                 false,
                 S,
                 D,
                 S,
                 S,
                 D,
                 D,
                 N * H,
                 S * S,
                 S * D,
                 S * D,
                 1.0f,
                 ABType,
                 fp8_ws,
                 ABType,
                 dataFwd.vData,
                 fp32_ws,
                 true);

            decltype(auto) scale_reduce_kernel = handle_.Run(kernels.back());
            scale_reduce_kernel(fp32_ws,
                                dataFwd.oData,
                                dataFwd.amaxOData,
                                dataFwd.descaleSData,
                                dataFwd.descaleVData,
                                dataFwd.scaleOData,
                                nhsd);

            if(profiling)
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                handle_.EnableProfiling(true);
                hipEventSynchronize(stop.get());
                float mS = 0;
                hipEventElapsedTime(&mS, start.get(), stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(mS);
            }
        };
    };

    return result;
}

bool MhaForward::MayNeedWorkspace() const { return true; }

} // namespace mha

} // namespace solver

} // namespace miopen
