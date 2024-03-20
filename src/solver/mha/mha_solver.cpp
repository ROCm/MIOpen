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

using WS        = std::tuple<Data_t, size_t>;
using Workspace = std::tuple<WS, WS, WS, size_t>;

Workspace SpitBufferToWorkspace(const std::vector<size_t>& lengths,
                                float dropout_p,
                                Data_t buffer,
                                size_t buffer_size)
{
    static constexpr size_t alignment = 256;

    auto [N, H, S, D] = miopen::tien<4>(lengths);

    auto GetNextBuffPtr = [](const WS& ws) {
        return static_cast<std::byte*>(std::get<Data_t>(ws)) + std::get<size_t>(ws);
    };

    // the first MatMul (N*H*S*D) * (N*H*S*D)T = (N*H*S*S)
    // the second MatMul (N*H*S*S) * (N*H*S*D) = (N*H*S*D)

    // temporary tensor for the first and second matmuls
    WS fp32_tensor{
        buffer,
        RoundUpToMultiple(N * H * S * (std::max(S, D)) * get_data_size(miopenFloat), alignment)};

    // temporary tensor for the first matmul
    WS fp8_tensor{GetNextBuffPtr(fp32_tensor),
                  RoundUpToMultiple(N * H * S * S * get_data_size(miopenFloat), alignment)};

    WS random_states{nullptr, 0};

    if(dropout_p > 0.0f)
    {
        size_t local_threads  = std::clamp<size_t>(nextPow2(S), warpSize, 256);
        size_t global_threads = N * H * S * local_threads;
        if(S <= warpSize)
        {
            global_threads = Ceil(global_threads, local_threads / warpSize);
        }

        random_states =
            WS{GetNextBuffPtr(fp8_tensor),
               RoundUpToMultiple(global_threads * sizeof(miopen::prng::xorwow_state), alignment)};
    }

    size_t total_size = std::get<size_t>(fp32_tensor) + std::get<size_t>(fp8_tensor) +
                        std::get<size_t>(random_states);

    if(buffer != nullptr && buffer_size < total_size)
    {
        if(buffer_size >= std::get<size_t>(fp32_tensor) + std::get<size_t>(fp8_tensor))
        {
            MIOPEN_THROW("Provded MHA workspace is less than required, "
                         "most probably because solver was created with zero dropout and called "
                         "with non-zero dropout.");
        }
        else
        {
            MIOPEN_THROW("Provded MHA workspace is less than required.");
        }
    }

    return {fp32_tensor, fp8_tensor, random_states, total_size};
}
} // namespace

bool MHA::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                       const miopen::mha::ProblemDescription& problem) const
{
    auto [N, H, S, D] = miopen::tien<4>(problem.GetDescs().descaleQDesc.GetLengths());

    return !miopen::IsDisabled(ENV(MIOPEN_DEBUG_ATTN_NAIVE)) && //
           S <= std::numeric_limits<uint32_t>::max() &&         //
           problem.GetDescs().kDesc.IsPacked() &&               //
           problem.GetDescs().qDesc.IsPacked() &&               //
           problem.GetDescs().vDesc.IsPacked() &&               //
           MIOPEN_USE_ROCBLAS;
}

std::size_t MHA::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                  const miopen::mha::ProblemDescription& problem) const
{
    return std::get<size_t>(SpitBufferToWorkspace(
        problem.GetDescs().kDesc.GetLengths(), problem.GetDescs().dropoutProbability, nullptr, 0));
}

ConvSolution MHA::GetSolution(const ExecutionContext& context,
                              const miopen::mha::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto [N, H, S, D] = miopen::tien<4>(problem.GetDescs().kDesc.GetLengths());
    uint32_t seq_len  = S;
    uint64_t nhs      = N * H * S;
    uint64_t nhsd     = N * H * S * D;

    auto warpSize = context.GetStream().GetWavefrontWidth();

    size_t local_threads  = std::clamp<uint32_t>(nextPow2(seq_len), warpSize, 256);
    size_t global_threads = nhs * local_threads;

    KernelBuildParameters build_params = KernelBuildParameters{{"THREADS", local_threads}};

    auto softmax_kernel         = KernelInfo{};
    softmax_kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    softmax_kernel.kernel_file  = "MIOpenSoftmaxAttn.cpp";
    softmax_kernel.kernel_name  = S > local_threads ? "SoftMaxCommon"
                                  : S > warpSize    ? "SoftMaxBlock"
                                                    : "SoftMaxWarp";
    if(S <= warpSize)
    {
        global_threads = Ceil(global_threads, local_threads / warpSize);
    }
    softmax_kernel.l_wk = {local_threads, 1, 1};
    softmax_kernel.g_wk = {global_threads, 1, 1};

    auto scale_reduce_kernel         = KernelInfo{};
    scale_reduce_kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    scale_reduce_kernel.kernel_file  = "MIOpenSoftmaxAttn.cpp";
    scale_reduce_kernel.kernel_name  = "ScaleReduce";
    local_threads            = RoundUpToMultiple<uint64_t>(std::min<uint64_t>(nhsd, 256), 32);
    global_threads           = RoundUpToMultiple(nhsd, local_threads);
    scale_reduce_kernel.l_wk = {local_threads, 1, 1};
    scale_reduce_kernel.g_wk = {global_threads, 1, 1};

    result.invoker_factory =
        [seq_len, nhs, nhsd, nh = N * H, s = S, d = D](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) softmax_kernel      = handle_.Run(kernels.front());
                decltype(auto) scale_reduce_kernel = handle_.Run(kernels.back());
                decltype(auto) params              = raw_params.CastTo<miopen::mha::InvokeParams>();

                hipMemsetAsync(params.GetData().amaxSData, 0, sizeof(float), handle_.GetStream());
                hipMemsetAsync(params.GetData().amaxOData, 0, sizeof(float), handle_.GetStream());

                auto [fp32_ws, fp8_ws, prng_ws, total_size] =
                    SpitBufferToWorkspace(params.GetDescs().descaleQDesc.GetLengths(),
                                          params.GetData().dropoutProbability,
                                          params.GetWorkspace(),
                                          params.GetWorkspaceSize());

                float alpha = 1.0f;
                float beta  = 0.0f;

#if MIOPEN_USE_ROCBLAS
                rocblas_status status = rocblas_set_atomics_mode(
                    handle_.rhandle().get(), rocblas_atomics_mode::rocblas_atomics_not_allowed);
                if(status != rocblas_status::rocblas_status_success)
                {
                    MIOPEN_THROW("rocblas_set_atomics_mode failed");
                }

                status =
                    (rocblas_gemm_strided_batched_ex)(handle_.rhandle().get(),
                                                      rocblas_operation_transpose,
                                                      rocblas_operation_none,
                                                      s,
                                                      s,
                                                      d,
                                                      &params.GetData().scale,
                                                      params.GetData().kData,
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      d,
                                                      s * d,
                                                      params.GetData().qData,
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      d,
                                                      d * s,
                                                      &beta,
                                                      std::get<Data_t>(fp32_ws),
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      s,
                                                      s * s,
                                                      std::get<Data_t>(fp32_ws),
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      s,
                                                      s * s,
                                                      nh,
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      rocblas_gemm_algo::rocblas_gemm_algo_standard,
                                                      0,
                                                      0);
                if(status != rocblas_status::rocblas_status_success)
                {
                    MIOPEN_THROW("Q*KT rocblas_gemm_strided_batched_ex failed");
                }
#endif
                softmax_kernel(std::get<Data_t>(fp32_ws),
                               std::get<Data_t>(fp8_ws),
                               params.GetData().mData,
                               params.GetData().zInvData,
                               params.GetData().amaxSData,
                               params.GetData().descaleQData,
                               params.GetData().descaleKData,
                               params.GetData().scaleSData,
                               std::get<Data_t>(prng_ws),
                               params.GetData().dropoutProbability,
                               seq_len,
                               nhs);
#if MIOPEN_USE_ROCBLAS

                status =
                    (rocblas_gemm_strided_batched_ex)(handle_.rhandle().get(),
                                                      rocblas_operation_none,
                                                      rocblas_operation_none,
                                                      d,
                                                      s,
                                                      s,
                                                      &alpha,
                                                      params.GetData().vData,
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      d,
                                                      d * s,
                                                      std::get<Data_t>(fp8_ws),
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      s,
                                                      s * s,
                                                      &beta,
                                                      std::get<Data_t>(fp32_ws),
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      d,
                                                      d * s,
                                                      std::get<Data_t>(fp32_ws),
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      d,
                                                      d * s,
                                                      nh,
                                                      rocblas_datatype::rocblas_datatype_f32_r,
                                                      rocblas_gemm_algo::rocblas_gemm_algo_standard,
                                                      0,
                                                      0);
                if(status != rocblas_status::rocblas_status_success)
                {
                    MIOPEN_THROW("S*V rocblas_gemm_strided_batched_ex failed");
                }
#endif
                scale_reduce_kernel(std::get<Data_t>(fp32_ws),
                                    params.GetData().oData,
                                    params.GetData().amaxOData,
                                    params.GetData().descaleSData,
                                    params.GetData().descaleVData,
                                    params.GetData().scaleOData,
                                    nhsd);
            };
        };

    result.construction_params.push_back(softmax_kernel);
    result.construction_params.push_back(scale_reduce_kernel);

    return result;
}

bool MHA::MayNeedWorkspace() const { return true; }

} // namespace mha

} // namespace solver

} // namespace miopen
