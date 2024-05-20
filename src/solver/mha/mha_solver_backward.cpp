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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_NAIVE_BWD)

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

template <typename T>
constexpr T nextPow2(T v)
{
    static_assert(std::is_unsigned_v<T>);

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
        if constexpr(sizeof(T) > 1)
            v |= v >> 8;
        if constexpr(sizeof(T) > 2)
            v |= v >> 16;
        if constexpr(sizeof(T) > 4)
            v |= v >> 32;
        v++;
        return v;
    }
}

MultiBufferWorkspaceTraits SplitBufferToWorkspace(size_t S, size_t D, size_t NHS)
{
    // the first MatMuls (N*H*S*D) * (N*H*S*D)T = (N*H*S*S)
    // the second MatMuls (N*H*S*S)[T] * (N*H*S*D) = (N*H*S*D)
    // dOxO row reduction (N*H*S*1)
    return MultiBufferWorkspaceTraits{NHS * S * get_data_size(miopenFloat), // first matmuls
                                      NHS * S * get_data_size(miopenFloat), // first matmuls
                                      NHS * std::max(S, D) *
                                          get_data_size(miopenFloat), // reduction and second matmul
                                      NHS * D * get_data_size(miopenFloat),  // second matmuls
                                      NHS * D * get_data_size(miopenFloat)}; // second matmuls
}

MultiBufferWorkspaceTraits SplitBufferToWorkspace(const std::vector<size_t>& lengths)
{
    const auto [N, H, S, D] = miopen::tien<4>(lengths);
    return SplitBufferToWorkspace(S, D, N * H * S);
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
    // It's important to have this check before problem.GetDescsBackward() call
    if(problem.IsForward())
    {
        return false;
    }

    const miopen::mha::MhaInputDescsBackward& descsBackward = problem.GetDescsBackward();

    auto [N, H, S, D] = miopen::tien<4>(descsBackward.kDesc.GetLengths());

    return !miopen::IsDisabled(ENV(MIOPEN_DEBUG_ATTN_NAIVE_BWD)) //
           && S <= std::numeric_limits<uint32_t>::max()                 //
           && D <= std::numeric_limits<uint32_t>::max()                 //
           && descsBackward.kDesc.IsPacked()                            //
           && descsBackward.qDesc.IsPacked()                            //
           && descsBackward.vDesc.IsPacked()                            //
           && descsBackward.oDesc.IsPacked()                            //
           && descsBackward.doDesc.IsPacked()                           //
           && descsBackward.mDesc.IsPacked()                            //
           && descsBackward.zInvDesc.IsPacked()                         //
           && descsBackward.dkDesc.IsPacked()                           //
           && descsBackward.dqDesc.IsPacked()                           //
           && descsBackward.dvDesc.IsPacked()                           //
           && MIOPEN_USE_GEMM;
}

std::size_t MhaBackward::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                          const miopen::mha::ProblemDescription& problem) const
{
    return SplitBufferToWorkspace(problem.GetDescsBackward().kDesc.GetLengths()).GetSize();
}

ConvSolution MhaBackward::GetSolution(const ExecutionContext& context,
                                      const miopen::mha::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto [N, H, S, D] = miopen::tien<4>(problem.GetDescsBackward().kDesc.GetLengths());
    uint32_t emb_dim  = static_cast<uint32_t>(D);
    uint32_t seq_len  = static_cast<uint32_t>(S);
    uint64_t nhs      = N * H * S;
    uint64_t nhsd     = N * H * S * D;
    float scale       = problem.GetDescsBackward().scale; // just to capture it into lambda

    auto warpSize = context.GetStream().GetWavefrontWidth();

    size_t local_threads  = std::clamp(nextPow2(D), warpSize, static_cast<size_t>(256));
    size_t global_threads = nhs * local_threads;

    auto dOxO_reduction_kernel = KernelInfo{};
    dOxO_reduction_kernel.comp_options =
        KernelBuildParameters{{"THREADS", local_threads}}.GenerateFor(kbp::HIP{});
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

    auto bwd_attention_kernel = KernelInfo{};
    bwd_attention_kernel.comp_options =
        KernelBuildParameters{{"THREADS", local_threads}}.GenerateFor(kbp::HIP{});
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

    auto getBuffPart = [ws = SplitBufferToWorkspace(S, D, nhs)](void* buffer, size_t part_idx) {
        return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
    };

    local_threads  = std::clamp(nextPow2(nhsd), warpSize, static_cast<size_t>(256));
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
                           problem.GetDescsBackward().scale,
                           0.0f,
                           problem.GetDescsBackward().kDesc.GetType(),
                           true);

    GemmDescriptor dOV_desc(false,
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
                            0.0f,
                            problem.GetDescsBackward().vDesc.GetType(),
                            true);

    GemmDescriptor xK_desc(false,
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
                           problem.GetDescsBackward().kDesc.GetType(),
                           true);

    GemmDescriptor xQdO_desc(false,
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
                             0.0f,
                             problem.GetDescsBackward().oDesc.GetType(),
                             true);
#endif

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

            void* fp32_QxK_S_ws     = getBuffPart(params.GetWorkspace(), 0);
            void* fp32_dOxV_dS_ws   = getBuffPart(params.GetWorkspace(), 1);
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
#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(
                handle_, QK_desc, dataBwd.qData, 0, dataBwd.kData, 0, fp32_QxK_S_ws, 0);
#endif
            HipEventPtr event_QxK = recordSyncEvent();
            hipMemsetAsync(dataBwd.amaxDQData, 0, sizeof(float), handle_.GetStream());

            handle_.SetStreamFromPool(2);
#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(
                handle_, dOV_desc, dataBwd.doData, 0, dataBwd.vData, 0, fp32_dOxV_dS_ws, 0);
#endif
            HipEventPtr event_dOxV = recordSyncEvent();
            hipMemsetAsync(dataBwd.amaxDKData, 0, sizeof(float), handle_.GetStream());

            handle_.SetStreamFromPool(0);
            waitSyncEvent(std::move(event_QxK));
            waitSyncEvent(std::move(event_dOxV));

            decltype(auto) bwd_attention_kernel = handle_.Run(kernels[1]);
            bwd_attention_kernel(fp32_QxK_S_ws,
                                 fp32_dOxV_dS_ws,
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

#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(
                handle_, xK_desc, fp32_dOxV_dS_ws, 0, dataBwd.kData, 0, fp32_dSxK_ws, 0);
#endif
            HipEventPtr event_bwd1 = recordSyncEvent();

#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(
                handle_, xQdO_desc, fp32_dOxV_dS_ws, 0, dataBwd.qData, 0, fp32_dSxQ_ws, 0);
#endif
            HipEventPtr event_bwd2 = recordSyncEvent();

            decltype(auto) scale_reduce_kernel = handle_.Run(kernels[2]);

            handle_.SetStreamFromPool(1);
            waitSyncEvent(std::move(event_bwd1));
            scale_reduce_kernel(fp32_dSxK_ws,
                                dataBwd.dqData,
                                dataBwd.amaxDQData,
                                dataBwd.descaleDSData,
                                dataBwd.descaleKData,
                                dataBwd.scaleDQData,
                                nhsd);
            HipEventPtr event_bwd3 = recordSyncEvent();

            handle_.SetStreamFromPool(2);
            waitSyncEvent(std::move(event_bwd2));
            scale_reduce_kernel(fp32_dSxQ_ws,
                                dataBwd.dkData,
                                dataBwd.amaxDKData,
                                dataBwd.descaleDSData,
                                dataBwd.descaleQData,
                                dataBwd.scaleDKData,
                                nhsd);
            HipEventPtr event_bwd4 = recordSyncEvent();

            handle_.SetStreamFromPool(0);
#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(
                handle_, xQdO_desc, fp32_QxK_S_ws, 0, dataBwd.doData, 0, fp32_dOxO_SxdO_ws, 0);
#endif

            scale_reduce_kernel(fp32_dOxO_SxdO_ws,
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
