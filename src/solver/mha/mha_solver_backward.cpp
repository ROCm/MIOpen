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

#include "../../kernels/miopen_rocrand.hpp"

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
    // the first MatMuls (N*H*S*D) * (N*H*S*D)T = (N*H*S*S)
    // the second MatMuls (N*H*S*S)[T] * (N*H*S*D) = (N*H*S*D)
    // dOxO row reduction (N*H*S*1)
    return MultiBufferWorkspaceTraits{
        NHS * S * get_data_size(miopenFloat),                 // first matmuls
        NHS * S * get_data_size(miopenFloat),                 // first matmuls
        NHS * std::max(S, D) * get_data_size(miopenFloat),    // reduction and second matmul
        NHS * D * get_data_size(miopenFloat),                 // second matmuls
        NHS * D * get_data_size(miopenFloat),                 // second matmuls
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
    const miopen::mha::MhaInputDescsBackward& descsForward = problem.GetDescsBackward();

    auto [N, H, S, D] = miopen::tien<4>(descsForward.kDesc.GetLengths());

    return !miopen::IsDisabled(ENV(MIOPEN_DEBUG_ATTN_NAIVE_BWD)) && //
           !problem.IsForward() &&                                  //
           S <= std::numeric_limits<uint32_t>::max() &&             //
           D <= std::numeric_limits<uint32_t>::max() &&             //
           descsForward.kDesc.IsPacked() &&                         //
           descsForward.qDesc.IsPacked() &&                         //
           descsForward.vDesc.IsPacked() &&                         //
           descsForward.oDesc.IsPacked() &&                         //
           descsForward.doDesc.IsPacked() &&                        //
           MIOPEN_USE_GEMM;
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
    uint32_t emb_dim  = D;
    uint32_t seq_len  = S;
    uint64_t nhs      = N * H * S;
    uint64_t nhsd     = N * H * S * D;
    float scale       = problem.GetDescsBackward().scale; // just to capture it into lambda

    auto warpSize = context.GetStream().GetWavefrontWidth();

    size_t local_threads  = std::clamp<uint32_t>(nextPow2(D), warpSize, 256);
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

    local_threads  = std::clamp<uint32_t>(nextPow2(S), warpSize, 256);
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

    auto getBuffPart = [ws = SplitBufferToWorkspace(S, D, nhs, global_threads)](void* buffer,
                                                                                size_t part_idx) {
        return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
    };

    local_threads  = std::clamp<uint32_t>(nextPow2(nhsd), warpSize, 256);
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

            handle_.ReserveExtraStreamsInPool(2);

            if(profiling)
            {
                start = make_hip_event();
                stop  = make_hip_event();
                handle_.EnableProfiling(false);
                hipEventRecord(start.get(), handle_.GetStream());
            }

            void* fp32_QxK_ws  = getBuffPart(params.GetWorkspace(), 0);
            void* fp32_dOxV_ws = getBuffPart(params.GetWorkspace(), 1);
            void* fp32_dOxO_ws = getBuffPart(params.GetWorkspace(), 2);
            void* fp32_dSxK_ws = getBuffPart(params.GetWorkspace(), 3);
            void* fp32_dSxQ_ws = getBuffPart(params.GetWorkspace(), 4);
            void* prng_ws      = getBuffPart(params.GetWorkspace(), 5);

            decltype(auto) dOxO_reduction_kernel = handle_.Run(kernels[0]);
            dOxO_reduction_kernel(params.GetDataBackward().doData,
                                  params.GetDataBackward().oData,
                                  fp32_dOxO_ws,
                                  params.GetDataBackward().descaleDOData,
                                  params.GetDataBackward().descaleOData,
                                  params.GetDataBackward().dropoutProbabilityData,
                                  emb_dim,
                                  nhs);
            hipMemsetAsync(
                params.GetDataBackward().amaxDSData, 0, sizeof(float), handle_.GetStream());
            hipMemsetAsync(
                params.GetDataBackward().amaxDVData, 0, sizeof(float), handle_.GetStream());

            const miopen::HipEventPtr event_QxK = make_hip_fast_event();
            handle_.SetStreamFromPool(1);
#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(handle_,
                                   QK_desc,
                                   params.GetDataBackward().qData,
                                   0,
                                   params.GetDataBackward().kData,
                                   0,
                                   fp32_QxK_ws,
                                   0,
                                   GemmBackend_t::rocblas);
#endif
            hipEventRecord(event_QxK.get(), handle_.GetStream());
            hipMemsetAsync(
                params.GetDataBackward().amaxDQData, 0, sizeof(float), handle_.GetStream());

            const miopen::HipEventPtr event_dOxV = make_hip_fast_event();
            handle_.SetStreamFromPool(2);
#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(handle_,
                                   dOV_desc,
                                   params.GetDataBackward().doData,
                                   0,
                                   params.GetDataBackward().vData,
                                   0,
                                   fp32_dOxV_ws,
                                   0,
                                   GemmBackend_t::rocblas);
#endif
            hipEventRecord(event_dOxV.get(), handle_.GetStream());
            hipMemsetAsync(
                params.GetDataBackward().amaxDKData, 0, sizeof(float), handle_.GetStream());

            handle_.SetStreamFromPool(0);
            hipStreamWaitEvent(handle_.GetStream(), event_QxK.get(), 0);
            hipStreamWaitEvent(handle_.GetStream(), event_dOxV.get(), 0);

            decltype(auto) bwd_attention_kernel = handle_.Run(kernels[1]);
            bwd_attention_kernel(fp32_QxK_ws,
                                 fp32_dOxV_ws,
                                 params.GetDataBackward().mData,
                                 params.GetDataBackward().zInvData,
                                 fp32_dOxO_ws,
                                 params.GetDataBackward().amaxDSData,
                                 params.GetDataBackward().descaleQData,
                                 params.GetDataBackward().descaleKData,
                                 params.GetDataBackward().descaleOData,
                                 params.GetDataBackward().descaleVData,
                                 params.GetDataBackward().scaleSData,
                                 params.GetDataBackward().scaleDSData,
                                 prng_ws,
                                 params.GetDataBackward().dropoutProbabilityData,
                                 scale,
                                 seq_len,
                                 nhs);
            const miopen::HipEventPtr event_bwd1 = make_hip_fast_event();
            hipEventRecord(event_bwd1.get(), handle_.GetStream());
            const miopen::HipEventPtr event_bwd2 = make_hip_fast_event();
            hipEventRecord(event_bwd2.get(), handle_.GetStream());

#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(handle_,
                                   xQdO_desc,
                                   fp32_QxK_ws,
                                   0,
                                   params.GetDataBackward().doData,
                                   0,
                                   fp32_dOxO_ws,
                                   0,
                                   GemmBackend_t::rocblas);
#endif
            decltype(auto) scale_reduce_kernel = handle_.Run(kernels[2]);
            scale_reduce_kernel(fp32_dOxO_ws,
                                params.GetDataBackward().dvData,
                                params.GetDataBackward().amaxDVData,
                                params.GetDataBackward().descaleSData,
                                params.GetDataBackward().descaleDOData,
                                params.GetDataBackward().scaleDVData,
                                nhsd);

            handle_.SetStreamFromPool(1);
            hipStreamWaitEvent(handle_.GetStream(), event_bwd1.get(), 0);
#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(handle_,
                                   xK_desc,
                                   fp32_dOxV_ws,
                                   0,
                                   params.GetDataBackward().kData,
                                   0,
                                   fp32_dSxK_ws,
                                   0,
                                   GemmBackend_t::rocblas);
#endif
            scale_reduce_kernel(fp32_dSxK_ws,
                                params.GetDataBackward().dqData,
                                params.GetDataBackward().amaxDQData,
                                params.GetDataBackward().descaleDSData,
                                params.GetDataBackward().descaleKData,
                                params.GetDataBackward().scaleDQData,
                                nhsd);
            const miopen::HipEventPtr event_bwd3 = make_hip_fast_event();
            hipEventRecord(event_bwd3.get(), handle_.GetStream());

            handle_.SetStreamFromPool(2);
            hipStreamWaitEvent(handle_.GetStream(), event_bwd1.get(), 0);
#if MIOPEN_USE_GEMM
            CallGemmStridedBatched(handle_,
                                   xQdO_desc,
                                   fp32_dOxV_ws,
                                   0,
                                   params.GetDataBackward().qData,
                                   0,
                                   fp32_dSxQ_ws,
                                   0,
                                   GemmBackend_t::rocblas);
#endif
            scale_reduce_kernel(fp32_dSxQ_ws,
                                params.GetDataBackward().dkData,
                                params.GetDataBackward().amaxDKData,
                                params.GetDataBackward().descaleDSData,
                                params.GetDataBackward().descaleQData,
                                params.GetDataBackward().scaleDKData,
                                nhsd);
            const miopen::HipEventPtr event_bwd4 = make_hip_fast_event();
            hipEventRecord(event_bwd4.get(), handle_.GetStream());

            handle_.SetStreamFromPool(0);
            hipStreamWaitEvent(handle_.GetStream(), event_bwd3.get(), 0);
            hipStreamWaitEvent(handle_.GetStream(), event_bwd4.get(), 0);

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
