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

#include <miopen/mha/solvers.hpp>

#include <miopen/mha/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/gemm_v2.hpp>

#include <algorithm>
#include <vector>
#include <tuple>

#include "../../kernels/miopen_rocrand.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_NAIVE)

namespace miopen {

namespace solver {

namespace mha {

namespace { // TODO: Issue #2748
template <typename T, typename S = std::enable_if_t<std::is_unsigned_v<T>, T>>
constexpr inline S Ceil(const T val, const T div)
{
    return (val - 1 + div) / div;
}

template <typename T, typename S = std::enable_if_t<std::is_unsigned_v<T>, T>>
constexpr S RoundUpToMultiple(T val, T mul)
{
    return Ceil(val, mul) * mul;
}

constexpr uint32_t nextPow2(uint32_t v)
{
    if(v == 1)
    {
        return (v << 1);
    }
    else
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
}

MultiBufferWorkspaceTraits
SplitBufferToWorkspace(size_t S, size_t D, size_t NHS, size_t global_threads)
{
    // the first MatMul (N*H*S*D) * (N*H*S*D)T = (N*H*S*S)
    // the second MatMul (N*H*S*S) * (N*H*S*D) = (N*H*S*D)
    return MultiBufferWorkspaceTraits{
        NHS * std::max(S, D) * get_data_size(miopenFloat),    // first and second matmuls tensor
        NHS * S * get_data_size(miopenFloat),                 // first matmul tensor
        global_threads * sizeof(miopen::prng::xorwow_state)}; // random state
}

MultiBufferWorkspaceTraits SplitBufferToWorkspace(const std::vector<size_t>& lengths)
{
    const auto [N, H, S, D] = miopen::tien<4>(lengths);
    const auto NHS          = N * H * S;

    size_t local_threads  = std::clamp<size_t>(nextPow2(S), warpSize, 256);
    size_t global_threads = NHS * local_threads;
    if(S <= warpSize)
    {
        global_threads = Ceil(global_threads, local_threads / warpSize);
    }

    return SplitBufferToWorkspace(S, D, NHS, global_threads);
}
} // namespace

bool Mha::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                       const miopen::mha::ProblemDescription& problem) const
{
    const miopen::mha::MhaInputDescsForward& descsForward = problem.GetDescsForward();

    auto [N, H, S, D] = miopen::tien<4>(descsForward.kDesc.GetLengths());

    return !miopen::IsDisabled(ENV(MIOPEN_DEBUG_ATTN_NAIVE)) && //
           S <= std::numeric_limits<uint32_t>::max() &&         //
           descsForward.kDesc.IsPacked() &&                     //
           descsForward.qDesc.IsPacked() &&                     //
           descsForward.vDesc.IsPacked() &&                     //
           MIOPEN_USE_GEMM;
}

std::size_t Mha::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                  const miopen::mha::ProblemDescription& problem) const
{
    return SplitBufferToWorkspace(problem.GetDescsForward().kDesc.GetLengths()).GetSize();
}

ConvSolution Mha::GetSolution(const ExecutionContext& context,
                              const miopen::mha::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto [N, H, S, D] = miopen::tien<4>(problem.GetDescsForward().kDesc.GetLengths());
    uint32_t seq_len  = S;
    uint64_t nhs      = N * H * S;
    uint64_t nhsd     = N * H * S * D;

    auto warpSize = context.GetStream().GetWavefrontWidth();

    size_t local_threads  = std::clamp<uint32_t>(nextPow2(seq_len), warpSize, 256);
    size_t global_threads = nhs * local_threads;

    auto softmax_kernel = KernelInfo{};
    softmax_kernel.comp_options =
        KernelBuildParameters{{"THREADS", local_threads}}.GenerateFor(kbp::HIP{});
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

    auto getBuffPart = [ws = SplitBufferToWorkspace(S, D, nhs, global_threads)](void* buffer,
                                                                                size_t part_idx) {
        return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
    };

    local_threads  = RoundUpToMultiple<uint64_t>(std::min<uint64_t>(nhsd, 256), 32);
    global_threads = RoundUpToMultiple(nhsd, local_threads);

    auto scale_reduce_kernel = KernelInfo{};
    scale_reduce_kernel.comp_options =
        KernelBuildParameters{{"THREADS", local_threads}}.GenerateFor(kbp::HIP{});
    scale_reduce_kernel.kernel_file = "MIOpenSoftmaxAttn.cpp";
    scale_reduce_kernel.kernel_name = "ScaleReduce";
    scale_reduce_kernel.l_wk        = {local_threads, 1, 1};
    scale_reduce_kernel.g_wk        = {global_threads, 1, 1};
    result.construction_params.push_back(scale_reduce_kernel);

#if MIOPEN_USE_GEMM
    GemmDescriptor QK_desc(false,
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
                           problem.GetDescsForward().scale,
                           0.0f,
                           problem.GetDescsForward().kDesc.GetType(),
                           true);

    GemmDescriptor SV_desc(false,
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
                           0.0f,
                           problem.GetDescsForward().vDesc.GetType(),
                           true);
#endif

    result.invoker_factory =
        [QK_desc, SV_desc, seq_len, nhs, nhsd, getBuffPart](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) params = raw_params.CastTo<miopen::mha::InvokeParams>();

                HipEventPtr start    = nullptr;
                HipEventPtr stop     = nullptr;
                const bool profiling = handle_.IsProfilingEnabled();

                if(profiling)
                {
                    start = make_hip_event();
                    stop  = make_hip_event();
                    handle_.EnableProfiling(false);
                    hipEventRecord(start.get(), handle_.GetStream());
                }

                // zero amax output data to use atomics
                hipMemsetAsync(params.GetData().amaxSData, 0, sizeof(float), handle_.GetStream());
                hipMemsetAsync(params.GetData().amaxOData, 0, sizeof(float), handle_.GetStream());

                void* fp32_ws = getBuffPart(params.GetWorkspace(), 0);
                void* fp8_ws  = getBuffPart(params.GetWorkspace(), 1);
                void* prng_ws = getBuffPart(params.GetWorkspace(), 2);

#if MIOPEN_USE_GEMM
                CallGemmStridedBatched(handle_,
                                       QK_desc,
                                       params.GetData().qData,
                                       0,
                                       params.GetData().kData,
                                       0,
                                       fp32_ws,
                                       0,
                                       GemmBackend_t::rocblas);
#endif
                decltype(auto) softmax_kernel = handle_.Run(kernels.front());
                softmax_kernel(fp32_ws,
                               fp8_ws,
                               params.GetData().mData,
                               params.GetData().zInvData,
                               params.GetData().amaxSData,
                               params.GetData().descaleQData,
                               params.GetData().descaleKData,
                               params.GetData().scaleSData,
                               prng_ws,
                               params.GetData().dropoutProbabilityData,
                               seq_len,
                               nhs);

#if MIOPEN_USE_GEMM
                CallGemmStridedBatched(handle_,
                                       SV_desc,
                                       fp8_ws,
                                       0,
                                       params.GetData().vData,
                                       0,
                                       fp32_ws,
                                       0,
                                       GemmBackend_t::rocblas);
#endif

                decltype(auto) scale_reduce_kernel = handle_.Run(kernels.back());
                scale_reduce_kernel(fp32_ws,
                                    params.GetData().oData,
                                    params.GetData().amaxOData,
                                    params.GetData().descaleSData,
                                    params.GetData().descaleVData,
                                    params.GetData().scaleOData,
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

bool Mha::MayNeedWorkspace() const { return true; }

} // namespace mha

} // namespace solver

} // namespace miopen
