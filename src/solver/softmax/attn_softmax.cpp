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

#include <miopen/softmax/solvers.hpp>

#include <miopen/softmax/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/softmax.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/float_equal.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ATTN_SOFTMAX)

namespace miopen::solver::softmax {

namespace {
template <typename T, typename S = std::enable_if_t<std::is_unsigned_v<T>, T>>
constexpr inline S Ceil(const T val, const T div)
{
    return (val - 1 + div) / div;
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
} // namespace

bool AttnSoftmax::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                               const miopen::softmax::ProblemDescription& problem) const
{
    const size_t seq_len = problem.GetXDesc().GetStrides().front(); // c * h * w
    const size_t nhs     = problem.GetXDesc().GetLengths().front(); // n

    return !env::disabled(MIOPEN_DEBUG_ATTN_SOFTMAX) &&         //
           seq_len <= std::numeric_limits<uint32_t>::max() &&   //
           problem.GetAlgorithm() == MIOPEN_SOFTMAX_ACCURATE && //
           problem.IsForward() &&                               //
           problem.GetXDesc().IsPacked() &&                     //
           problem.GetYDesc().IsPacked() &&                     //
           problem.GetXDesc().GetType() == miopenFloat &&       //
           problem.GetYDesc().GetType() == miopenFloat &&       //
           problem.GetMode() == MIOPEN_SOFTMAX_MODE_INSTANCE && //
           float_equal(problem.GetAlpha(), 1.0f) &&             //
           float_equal(problem.GetBeta(), 0.f) &&               //
           (seq_len > 16 || nhs <= 1024);                       // heuristic
}

std::size_t AttnSoftmax::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::softmax::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution AttnSoftmax::GetSolution(const ExecutionContext& context,
                                      const miopen::softmax::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    // instance mode
    uint32_t seq_len = problem.GetXDesc().GetStrides().front(); // c * h * w
    uint64_t nhs     = problem.GetXDesc().GetLengths().front(); // n

    auto warpSize = context.GetStream().GetWavefrontWidth();

    size_t local_threads  = std::clamp<size_t>(nextPow2(seq_len), warpSize, 256);
    size_t global_threads = nhs * local_threads;

    KernelBuildParameters build_params = KernelBuildParameters{{"THREADS", local_threads}};

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.kernel_file = "MIOpenSoftmaxAttn.cpp";

    kernel.kernel_name = seq_len > local_threads ? "SoftMaxCommon"
                         : seq_len > warpSize    ? "SoftMaxBlock"
                                                 : "SoftMaxWarp";

    if(seq_len <= warpSize)
    {
        global_threads = Ceil(global_threads, local_threads / warpSize);
    }

    kernel.l_wk = {local_threads, 1, 1};
    kernel.g_wk = {global_threads, 1, 1};

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::softmax::InvokeParams>();

            kernel(params.x,
                   params.forward_y,
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   nullptr, // attention related parameters
                   seq_len,
                   nhs);
        };
    };

    result.construction_params.push_back(kernel);

    return result;
}

} // namespace miopen::solver::softmax
