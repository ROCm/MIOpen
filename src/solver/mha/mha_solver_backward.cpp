/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/buffer_info.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#include <algorithm>
#include <vector>
#include <tuple>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_NAIVE_BWD)

namespace miopen {

namespace solver {

namespace mha {

namespace { // TODO: Issue #2748

MultiBufferWorkspaceTraits
SplitBufferToWorkspace(size_t S, size_t D, size_t NHS, miopenDataType_t out_type)
{
    // the first MatMuls (N*H*S*D) * (N*H*S*D)T = (N*H*S*S)
    // the second MatMuls (N*H*S*S)[T] * (N*H*S*D) = (N*H*S*D)
    // dOxO row reduction (N*H*S*1)
    return MultiBufferWorkspaceTraits{
        NHS * S * get_data_size(miopenFloat),                             // fp32 QxK and fp32 S
        NHS * S * get_data_size(miopenFloat),                             // fp32 dOxV and fp32 dS
        NHS * std::max(S, D) * get_data_size(miopenFloat),                // fp32 dOxO and fp32 SxdO
        NHS * D * get_data_size(miopenFloat),                             // fp32 dSxK
        NHS * D * get_data_size(miopenFloat),                             // fp32 dSxQ
        out_type == miopenFloat ? 0 : NHS * S * get_data_size(out_type)}; // fp8 dS
}

MultiBufferWorkspaceTraits SplitBufferToWorkspace(const std::vector<size_t>& lengths,
                                                  miopenDataType_t out_type)
{
    const auto [N, H, S, D] = miopen::tien<4>(lengths);
    return SplitBufferToWorkspace(S, D, N * H * S, out_type);
}

miopen::HipEventPtr make_hip_fast_event()
{
    hipEvent_t result = nullptr;
    hipEventCreateWithFlags(&result, hipEventDisableTiming);
    return miopen::HipEventPtr{result};
}
} // namespace

bool MhaBackward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                               const miopen::mha::ProblemDescription& problem) const
{
#if MIOPEN_USE_ROCBLAS
    // It's important to have this check before problem.GetDescsBackward() call
    if(problem.IsForward())
    {
        return false;
    }

    const miopen::mha::MhaInputDescsBackward& descsBwd = problem.GetDescsBackward();

    auto [N, H, S, D] = miopen::tien<4>(descsBwd.kDesc.GetLengths());

    return !env::disabled(MIOPEN_DEBUG_ATTN_NAIVE_BWD)                //
           && S <= std::numeric_limits<uint32_t>::max()               //
           && D <= std::numeric_limits<uint32_t>::max()               //
           && descsBwd.kDesc.IsPacked()                               //
           && descsBwd.qDesc.IsPacked()                               //
           && descsBwd.vDesc.IsPacked()                               //
           && descsBwd.oDesc.IsPacked()                               //
           && descsBwd.doDesc.IsPacked()                              //
           && descsBwd.mDesc.IsPacked()                               //
           && descsBwd.zInvDesc.IsPacked()                            //
           && descsBwd.dkDesc.IsPacked()                              //
           && descsBwd.dqDesc.IsPacked()                              //
           && descsBwd.dvDesc.IsPacked()                              //
           && descsBwd.mDesc.GetType() == miopenFloat                 //
           && descsBwd.zInvDesc.GetType() == miopenFloat              //
           && descsBwd.kDesc.GetType() == descsBwd.qDesc.GetType()    //
           && descsBwd.kDesc.GetType() == descsBwd.vDesc.GetType()    //
           && descsBwd.kDesc.GetType() == descsBwd.oDesc.GetType()    //
           && descsBwd.kDesc.GetType() == descsBwd.dqDesc.GetType()   //
           && descsBwd.kDesc.GetType() == descsBwd.dkDesc.GetType()   //
           && descsBwd.kDesc.GetType() == descsBwd.dvDesc.GetType()   //
           && ((descsBwd.kDesc.GetType() == miopenFloat)              //
               || (USE_ROCBLAS_EX3                                    //
                   && (MIOPEN_FP8_IEEE_EXPONENT_BIAS == 0)            //
                   && (descsBwd.kDesc.GetType() == miopenFloat8)))    //
           && ((descsBwd.doDesc.GetType() == miopenFloat)             //
               || (USE_ROCBLAS_EX3                                    //
                   && (MIOPEN_FP8_IEEE_EXPONENT_BIAS == 0)            //
                   && (descsBwd.doDesc.GetType() == miopenBFloat8))); //
#else
    return false;
#endif
}

std::size_t MhaBackward::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                          const miopen::mha::ProblemDescription& problem) const
{
    const auto& kDesc = problem.GetDescsBackward().kDesc;
    return SplitBufferToWorkspace(kDesc.GetLengths(), kDesc.GetType()).GetSize();
}

ConvSolution MhaBackward::GetSolution(const ExecutionContext& context,
                                      const miopen::mha::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    uint64_t N, H, S, D;
    std::tie(N, H, S, D) = miopen::tien<4>(problem.GetDescsBackward().kDesc.GetLengths());
    uint32_t emb_dim     = static_cast<uint32_t>(D);
    uint32_t seq_len     = static_cast<uint32_t>(S);
    uint64_t nhs         = N * H * S;
    uint64_t nhsd        = N * H * S * D;
    float scale          = problem.GetDescsBackward().scale; // just to capture it into lambda

    auto ABType_K  = problem.GetDescsBackward().kDesc.GetType();
    auto ABType_dO = problem.GetDescsBackward().doDesc.GetType();

    auto warpSize = context.GetStream().GetWavefrontWidth();

    size_t local_threads  = std::clamp(nextPow2(D), warpSize, static_cast<size_t>(256));
    size_t global_threads = nhs * local_threads;

    auto dOxO_reduction_kernel         = KernelInfo{};
    dOxO_reduction_kernel.comp_options = KernelBuildParameters{
        {"THREADS", local_threads},
        {"OUT_TYPE", GetDataType(ABType_K)},
        {"dO_TYPE", GetDataType(ABType_dO)}}.GenerateFor(kbp::HIP{});
    dOxO_reduction_kernel.kernel_file = "MIOpenSoftmaxAttn.cpp";
    dOxO_reduction_kernel.kernel_name = D > local_threads ? "ScaleRowReduceCommon"
                                        : D > warpSize    ? "ScaleRowReduceBlock"
                                                          : "ScaleRowReduceWarp";
    if(D <= warpSize)
    {
        global_threads = Ceil(global_threads, local_threads / warpSize);
    }
    dOxO_reduction_kernel.l_wk = {local_threads, 1, 1};
    dOxO_reduction_kernel.g_wk = {global_threads, 1, 1};
    result.construction_params.push_back(dOxO_reduction_kernel);

    local_threads  = std::clamp(nextPow2(S), warpSize, static_cast<size_t>(256));
    global_threads = nhs * local_threads;

    auto bwd_attention_kernel         = KernelInfo{};
    bwd_attention_kernel.comp_options = KernelBuildParameters{
        {"THREADS", local_threads},
        {"OUT_TYPE", GetDataType(ABType_K)}}.GenerateFor(kbp::HIP{});
    bwd_attention_kernel.kernel_file = "MIOpenSoftmaxAttn.cpp";
    bwd_attention_kernel.kernel_name = S > local_threads ? "BwdAttentionCommon"
                                       : S > warpSize    ? "BwdAttentionBlock"
                                                         : "BwdAttentionWarp";
    if(S <= warpSize)
    {
        global_threads = Ceil(global_threads, local_threads / warpSize);
    }
    bwd_attention_kernel.l_wk = {local_threads, 1, 1};
    bwd_attention_kernel.g_wk = {global_threads, 1, 1};
    result.construction_params.push_back(bwd_attention_kernel);

    auto getBuffPart = [ws = SplitBufferToWorkspace(S, D, nhs, ABType_K)](void* buffer,
                                                                          size_t part_idx) {
        return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
    };

    local_threads  = std::clamp(nextPow2(nhsd), warpSize, static_cast<size_t>(256));
    global_threads = RoundUpToMultiple(nhsd, local_threads);

    auto scale_reduce_kernel         = KernelInfo{};
    scale_reduce_kernel.comp_options = KernelBuildParameters{
        {"THREADS", local_threads},
        {"OUT_TYPE",
         GetDataType(ABType_K)}}.GenerateFor(kbp::HIP{});
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
            const auto& dataBwd  = params.GetDataBackward();

            handle_.ReserveExtraStreamsInPool(2);

            auto recordSyncEvent = [&handle_]() {
                auto event = make_hip_fast_event();
                hipEventRecord(event.get(), handle_.GetStream());
                return event;
            };

            auto waitSyncEvent = [&handle_](HipEventPtr&& event) {
                auto tmp_for_deletion(std::move(event));
                hipStreamWaitEvent(handle_.GetStream(), tmp_for_deletion.get(), 0);
            };

            if(profiling)
            {
                start = make_hip_event();
                stop  = make_hip_event();
                handle_.EnableProfiling(false);
                hipEventRecord(start.get(), handle_.GetStream());
            }

            void* fp32_QxK_S_ws = getBuffPart(params.GetWorkspace(), 0);
            void* fp32_dOxV_ws  = getBuffPart(params.GetWorkspace(), 1);
            void* fp32_dS_ws =
                ABType_K == miopenFloat ? fp32_dOxV_ws : getBuffPart(params.GetWorkspace(), 5);
            void* fp32_dOxO_SxdO_ws = getBuffPart(params.GetWorkspace(), 2);
            void* fp32_dSxK_ws      = getBuffPart(params.GetWorkspace(), 3);
            void* fp32_dSxQ_ws      = getBuffPart(params.GetWorkspace(), 4);

            decltype(auto) dOxO_reduction_kernel = handle_.Run(kernels[0]);
            dOxO_reduction_kernel(dataBwd.doData,
                                  dataBwd.oData,
                                  fp32_dOxO_SxdO_ws,
                                  dataBwd.descaleDOData,
                                  dataBwd.descaleOData,
                                  dataBwd.dropoutProbabilityData,
                                  emb_dim,
                                  nhs);
            hipMemsetAsync(dataBwd.amaxDSData, 0, sizeof(float), handle_.GetStream());
            hipMemsetAsync(dataBwd.amaxDVData, 0, sizeof(float), handle_.GetStream());

            handle_.SetStreamFromPool(1);
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
                 ABType_K,
                 dataBwd.qData,
                 ABType_K,
                 dataBwd.kData,
                 fp32_QxK_S_ws,
                 true);

            HipEventPtr event_QxK = recordSyncEvent();
            hipMemsetAsync(dataBwd.amaxDQData, 0, sizeof(float), handle_.GetStream());

            handle_.SetStreamFromPool(2);
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
                 1.0f,
                 ABType_dO,
                 dataBwd.doData,
                 ABType_K,
                 dataBwd.vData,
                 fp32_dOxV_ws,
                 true);

            HipEventPtr event_dOxV = recordSyncEvent();
            hipMemsetAsync(dataBwd.amaxDKData, 0, sizeof(float), handle_.GetStream());

            handle_.SetStreamFromPool(0);
            waitSyncEvent(std::move(event_QxK));
            waitSyncEvent(std::move(event_dOxV));

            decltype(auto) bwd_attention_kernel = handle_.Run(kernels[1]);
            bwd_attention_kernel(fp32_QxK_S_ws,
                                 fp32_dOxV_ws,
                                 fp32_dS_ws,
                                 dataBwd.mData,
                                 dataBwd.zInvData,
                                 fp32_dOxO_SxdO_ws,
                                 dataBwd.amaxDSData,
                                 dataBwd.descaleQData,
                                 dataBwd.descaleKData,
                                 dataBwd.descaleDOData,
                                 dataBwd.descaleVData,
                                 dataBwd.scaleSData,
                                 dataBwd.scaleDSData,
                                 dataBwd.dropoutSeedData,
                                 dataBwd.dropoutOffsetData,
                                 dataBwd.dropoutProbabilityData,
                                 scale,
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
                 ABType_K,
                 fp32_dS_ws,
                 ABType_K,
                 dataBwd.kData,
                 fp32_dSxK_ws,
                 true);
            HipEventPtr event_bwd1 = recordSyncEvent();

            gemm(handle_,
                 true,
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
                 ABType_K,
                 fp32_dS_ws,
                 ABType_K,
                 dataBwd.qData,
                 fp32_dSxQ_ws,
                 true);
            HipEventPtr event_bwd2 = recordSyncEvent();

            handle_.SetStreamFromPool(1);
            decltype(auto) scale_reduce_kerneldSxK = handle_.Run(kernels[2]);
            waitSyncEvent(std::move(event_bwd1));
            scale_reduce_kerneldSxK(fp32_dSxK_ws,
                                    dataBwd.dqData,
                                    dataBwd.amaxDQData,
                                    dataBwd.descaleDSData,
                                    dataBwd.descaleKData,
                                    dataBwd.scaleDQData,
                                    nhsd);
            HipEventPtr event_bwd3 = recordSyncEvent();

            handle_.SetStreamFromPool(2);
            decltype(auto) scale_reduce_kerneldSxQ = handle_.Run(kernels[2]);
            waitSyncEvent(std::move(event_bwd2));
            scale_reduce_kerneldSxQ(fp32_dSxQ_ws,
                                    dataBwd.dkData,
                                    dataBwd.amaxDKData,
                                    dataBwd.descaleDSData,
                                    dataBwd.descaleQData,
                                    dataBwd.scaleDKData,
                                    nhsd);
            HipEventPtr event_bwd4 = recordSyncEvent();

            handle_.SetStreamFromPool(0);
            gemm(handle_,
                 true,
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
                 miopenFloat,
                 fp32_QxK_S_ws,
                 ABType_dO,
                 dataBwd.doData,
                 fp32_dOxO_SxdO_ws,
                 true);

            decltype(auto) scale_reduce_kernelSxdO = handle_.Run(kernels[2]);
            scale_reduce_kernelSxdO(fp32_dOxO_SxdO_ws,
                                    dataBwd.dvData,
                                    dataBwd.amaxDVData,
                                    dataBwd.descaleSData,
                                    dataBwd.descaleDOData,
                                    dataBwd.scaleDVData,
                                    nhsd);

            waitSyncEvent(std::move(event_bwd3));
            waitSyncEvent(std::move(event_bwd4));

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

bool MhaBackward::MayNeedWorkspace() const { return true; }

} // namespace mha

} // namespace solver

} // namespace miopen
