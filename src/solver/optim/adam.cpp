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

#include <miopen/optim/adam/solvers.hpp>

#include <miopen/adam.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/optim/adam/invoke_params.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace adam {

bool IsImprovementOverROCm([[maybe_unused]] const miopen::adam::ProblemDescription& problem)
{
    return true;
}

bool Adam::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                        const miopen::adam::ProblemDescription& problem) const
{
    if(!problem.IsAllPacked())
        return false;
    if(!IsImprovementOverROCm(problem))
        return false;
    return true;
}

inline size_t AlignUp(size_t num, size_t align) { return (num + align - 1) / align * align; }

ConvSolution Adam::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::adam::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};
    auto dtype  = problem.GetParamDesc().GetType();

    {
        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        };

        auto kernel = KernelInfo{};

        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(1);

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.kernel_file = "MIOpenAdam.cpp";
        kernel.kernel_name = problem.IsAmp() ? "AmpAdamPacked" : "AdamPacked";

        result.construction_params.push_back(kernel);

        if(problem.IsAmp())
        {
            auto update_step_kernel        = kernel;
            update_step_kernel.kernel_name = "AmpAdamUpdateStep";

            result.construction_params.push_back(update_step_kernel);
        }
    }

    constexpr size_t local_size = 256;
    if(problem.IsAmp())
    {
        result.invoker_factory = [local_size](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel_adam = handle_.Run(kernels[0]);
                decltype(auto) kernel_step = handle_.Run(kernels[1]);
                decltype(auto) params      = raw_params.CastTo<miopen::adam::InvokeParams>();
                decltype(auto) numel       = params.paramDesc->GetElementSize();

                kernel_adam.ldims = {std::min(numel, local_size), 1, 1};
                kernel_adam.gdims = {AlignUp(numel, local_size), 1, 1};

                kernel_adam(params.param,
                            params.grad,
                            params.expAvg,
                            params.expAvgSq,
                            params.maxExpAvgSq,
                            params.gradScale,
                            params.foundInf,
                            params.step,
                            params.lr,
                            params.beta1,
                            params.beta2,
                            params.weight_decay,
                            params.eps,
                            params.amsgrad,
                            params.maximize,
                            numel);

                kernel_step(params.foundInf, params.step);
            };
        };
    }
    else
    {
        result.invoker_factory = [local_size](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::adam::InvokeParams>();
                decltype(auto) numel  = params.paramDesc->GetElementSize();

                kernel.ldims = {std::min(numel, local_size), 1, 1};
                kernel.gdims = {AlignUp(numel, local_size), 1, 1};

                kernel(params.param,
                       params.grad,
                       params.expAvg,
                       params.expAvgSq,
                       params.maxExpAvgSq,
                       params.step,
                       params.lr,
                       params.beta1,
                       params.beta2,
                       params.weight_decay,
                       params.eps,
                       params.amsgrad,
                       params.maximize,
                       numel);
            };
        };
    }

    return result;
}

} // namespace adam

} // namespace solver

} // namespace miopen
