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
#include <miopen/buffer_info.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/median.hpp>
#include <miopen/median/solvers.hpp>
#include <miopen/median/invoke_params.hpp>
#include <miopen/median/problem_description.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace median {

bool IsImprovementOverROCm(const miopen::median::BwdProblemDescription& problem)
{
    auto dim                = problem.GetDim();
    auto input_grad_lengths = problem.GetInputGradDesc().GetLengths();
    auto dim_size           = input_grad_lengths[dim];
    auto dim_stride         = problem.GetInputGradDesc().GetStrides()[dim];

    return input_grad_lengths.size() > 1 && !problem.IsAllContiguous() && dim_size > 250 &&
           dim_stride == 1;
}

bool MedianBackward::IsApplicable(const ExecutionContext& /*context*/,
                                  const miopen::median::BwdProblemDescription& problem) const
{
    if(!problem.IsValidNumDims())
        return false;

    if(!problem.IsValidFloat())
        return false;

    if(!IsImprovementOverROCm(problem))
        return false;

    return true;
}

ConvSolution MedianBackward::GetSolution(const ExecutionContext& context,
                                         const miopen::median::BwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetInputGradDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);

    auto input_grad_lengths = problem.GetInputGradDesc().GetLengths();

    auto dim_size = input_grad_lengths[problem.GetDim()];

    auto size        = problem.GetInputGradDesc().GetElementSize();
    auto output_size = size / dim_size;

    // Start building result.construction_params
    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = output_size * xlocalsize;
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenKthvalue.cpp";
    kernel.kernel_name = "KthvalueBwd";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
        {"LOCAL_SIZE", xlocalsize},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);
    // End building result.construction_params

    // Start building result.invoker_factory
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::median::BwdInvokeParams>();
            size_t dim_stride     = params.inputGradDesc->GetStrides()[params.dim];

            auto input_grad_tv = get_inner_expanded_tv<5>(deref(params.inputGradDesc));
            auto input_grad_tv_without_selected_dim =
                get_tv_without_dim<5>(input_grad_tv, params.dim);

            auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.outputGradDesc));
            auto indices_tv     = get_inner_expanded_tv<5>(deref(params.indicesDesc));

            kernel(params.inputGrad,
                   params.outputGrad,
                   params.indices,
                   dim_size,
                   dim_stride,
                   input_grad_tv_without_selected_dim,
                   output_grad_tv,
                   indices_tv);
        };
    };

    return result;
}

} // namespace median

} // namespace solver

} // namespace miopen
