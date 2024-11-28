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
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/gradientdescent/solvers.hpp>

#include <miopen/gradientdescent/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/gradientdescent.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace GradientDescent {

bool GradientDescent::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                   const miopen::GradientDescent::ProblemDescription& problem) const
{
    if(!(problem.GetvarInDesc().GetType() == miopenFloat ||
         problem.GetvarInDesc().GetType() == miopenHalf ||
         problem.GetvarInDesc().GetType() == miopenBFloat16))
        return false;
    return true;
}

ConvSolution
GradientDescent::GetSolution([[maybe_unused]] const ExecutionContext& context,
                             const miopen::GradientDescent::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto is_contiguous = problem.IsAllContiguous();
    auto dtype         = problem.GetvarInDesc().GetType();
    auto d_dtype       = miopen::GetDataType(dtype);
    auto nelems        = problem.GetvarInDesc().GetElementSize();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"D_TYPE", d_dtype == "bfloat16" ? "ushort" : d_dtype},
    };

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(nelems, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel         = KernelInfo{};
    kernel.kernel_file  = "MIOpenGradientDescent.cpp";
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    if(is_contiguous)
    {
        kernel.kernel_name = "ResourceApplyGradientDescentContiguous";
        result.construction_params.push_back(kernel);

        result.invoker_factory = [nelems](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::GradientDescent::InvokeParams>();

                kernel(params.var_in, params.var_out, params.alpha_in, params.delta_in, nelems);
            };
        };
    }
    else
    {
        kernel.kernel_name = "ResourceApplyGradientDescent";
        result.construction_params.push_back(kernel);

        result.invoker_factory = [nelems](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::GradientDescent::InvokeParams>();

                auto var_in_tv   = get_inner_expanded_tv<5>(deref(params.varInDesc));
                auto var_out_tv  = get_inner_expanded_tv<5>(deref(params.varOutDesc));
                auto alpha_in_tv = get_inner_expanded_tv<1>(deref(params.alphaInDesc));
                auto delta_in_tv = get_inner_expanded_tv<5>(deref(params.deltaInDesc));

                kernel(params.var_in,
                       params.var_out,
                       params.alpha_in,
                       params.delta_in,
                       nelems,
                       var_in_tv,
                       var_out_tv,
                       alpha_in_tv,
                       delta_in_tv);
            };
        };
    }
    return result;
}

} // namespace GradientDescent

} // namespace solver

} // namespace miopen
