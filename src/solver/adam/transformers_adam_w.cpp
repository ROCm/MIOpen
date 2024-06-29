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

#include <miopen/adam/solvers.hpp>

#include <miopen/adam.hpp>
#include <miopen/adam/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace adam {

bool TransformersAdamW::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::adam::ProblemDescription& problem) const
{
    if(!problem.IsAllContiguous())
        return false;
    return true;
}

ConvSolution TransformersAdamW::GetSolution(const ExecutionContext& context,
                                            const miopen::adam::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto param_dtype = miopen::GetDataType(problem.GetParamDesc().GetType());
        auto ptype_size  = miopen::get_data_size(problem.GetParamDesc().GetType());
        auto grad_dtype  = (problem.IsAmp() || problem.ExistStepTensor())
                               ? miopen::GetDataType(problem.GetGradDesc().GetType())
                               : "float";

        const auto build_params =
            KernelBuildParameters{
                {"PTYPE", param_dtype},
                {"GTYPE", grad_dtype},
                {"CTYPE", ptype_size > 4 ? "double" : "float"},
            }
            << GetDataTypeKBP(problem.GetParamDesc().GetType());

        constexpr size_t local_size = 256;
        auto& handle                = context.GetStream();
        auto numCu                  = handle.GetMaxComputeUnits();
        auto grid_size              = numCu * 4 * local_size;

        auto kernel = KernelInfo{};

        kernel.l_wk.push_back(local_size);
        kernel.g_wk.push_back(grid_size);

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.kernel_file = "MIOpenAdam.cpp";
        if(problem.ExistStepTensor())
        {
            kernel.kernel_name = "TransformersAmpAdamWContiguousWithStep";
        }
        else
        {
            kernel.kernel_name =
                problem.IsAmp() ? "TransformersAmpAdamWContiguous" : "TransformersAdamWContiguous";
        }

        result.construction_params.push_back(kernel);

        if(problem.ExistStepTensor())
        {
            auto kernel_update_step        = kernel;
            kernel_update_step.kernel_name = "AdamUpdateStep";

            result.construction_params.push_back(kernel_update_step);
        }
    }

    if(problem.ExistStepTensor())
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel_adam = handle_.Run(kernels[0]);
                decltype(auto) kernel_step = handle_.Run(kernels[1]);
                decltype(auto) params =
                    raw_params.CastTo<miopen::adam::TransformersAdamWInvokeParams>();
                decltype(auto) numel  = params.paramDesc->GetElementSize();
                float lr_weight_decay = params.lr * params.weight_decay;
                auto elapsed          = 0.f;

                kernel_adam(params.paramIn,
                            params.paramOut,
                            params.paramOutFloat16,
                            params.gradIn,
                            params.expAvgIn,
                            params.expAvgOut,
                            params.expAvgSqIn,
                            params.expAvgSqOut,
                            params.gradScale,
                            params.foundInf,
                            params.stepIn,
                            params.lr,
                            params.beta1,
                            params.beta2,
                            params.eps,
                            lr_weight_decay,
                            params.step_size,
                            params.correct_bias,
                            numel);

                if(handle_.IsProfilingEnabled())
                    elapsed = handle_.GetKernelTime();

                kernel_step(params.foundInf, params.stepIn, params.stepOut);

                if(handle_.IsProfilingEnabled())
                {
                    elapsed += handle_.GetKernelTime();
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                }
            };
        };
    }
    else
    {
        if(problem.IsAmp())
        {
            result.invoker_factory = [](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                    decltype(auto) kernel = handle_.Run(kernels.front());
                    decltype(auto) params =
                        raw_params.CastTo<miopen::adam::TransformersAdamWInvokeParams>();
                    decltype(auto) numel  = params.paramDesc->GetElementSize();
                    float lr_weight_decay = params.lr * params.weight_decay;
                    auto step_size        = params.step_size;

                    if(step_size < 0)
                    {
                        if(params.correct_bias)
                        {
                            float bias_correction1 = 1 - pow(params.beta1, params.step);
                            float bias_correction2 = 1 - pow(params.beta2, params.step);
                            step_size = params.lr * sqrt(bias_correction2) / bias_correction1;
                        }
                        else
                        {
                            step_size = params.lr;
                        }
                    }

                    kernel(params.paramIn,
                           params.paramOut,
                           params.paramOutFloat16,
                           params.gradIn,
                           params.expAvgIn,
                           params.expAvgOut,
                           params.expAvgSqIn,
                           params.expAvgSqOut,
                           params.gradScale,
                           params.foundInf,
                           params.beta1,
                           params.beta2,
                           params.eps,
                           lr_weight_decay,
                           step_size,
                           numel);
                };
            };
        }
        else
        {
            result.invoker_factory = [](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                    decltype(auto) kernel = handle_.Run(kernels.front());
                    decltype(auto) params =
                        raw_params.CastTo<miopen::adam::TransformersAdamWInvokeParams>();
                    decltype(auto) numel  = params.paramDesc->GetElementSize();
                    float lr_weight_decay = params.lr * params.weight_decay;
                    auto step_size        = params.step_size;

                    if(step_size < 0)
                    {
                        if(params.correct_bias)
                        {
                            float bias_correction1 = 1 - pow(params.beta1, params.step);
                            float bias_correction2 = 1 - pow(params.beta2, params.step);
                            step_size = params.lr * sqrt(bias_correction2) / bias_correction1;
                        }
                        else
                        {
                            step_size = params.lr;
                        }
                    }

                    kernel(params.paramIn,
                           params.paramOut,
                           params.gradIn,
                           params.expAvgIn,
                           params.expAvgOut,
                           params.expAvgSqIn,
                           params.expAvgSqOut,
                           params.beta1,
                           params.beta2,
                           params.eps,
                           lr_weight_decay,
                           step_size,
                           numel);
                };
            };
        }
    }

    return result;
}

} // Namespace adam

} // namespace solver

} // namespace miopen
