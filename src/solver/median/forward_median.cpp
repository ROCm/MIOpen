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
#include <miopen/median.hpp>
#include <miopen/median/solvers.hpp>
#include <miopen/median/invoke_params.hpp>
#include <miopen/median/problem_description.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace median {

bool IsImprovementOverROCm(const miopen::median::FwdProblemDescription& problem)
{
    auto dim           = problem.GetDim();
    auto input_lengths = problem.GetInputDesc().GetLengths();
    auto dim_size      = input_lengths[dim];
    auto is_contiguous = problem.GetInputDesc().IsContiguous();
    auto dim_stride    = problem.GetInputDesc().GetStrides()[dim];

    return input_lengths.size() > 1 && !is_contiguous && dim_size > 300 && dim_stride == 1;
}

bool MedianForward::IsApplicable(const ExecutionContext& /*context*/,
                                 const miopen::median::FwdProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(problem))
        return false;

    if(!problem.IsValidFloat())
        return false;

    return true;
}

ConvSolution MedianForward::GetSolution(const ExecutionContext& context,
                                        const miopen::median::FwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetInputDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);

    auto input_lengths  = problem.GetInputDesc().GetLengths();
    auto output_lengths = problem.GetOutputDesc().GetLengths();

    auto dim_size   = input_lengths[problem.GetDim()]; // reduce_size
    auto dim_stride = problem.GetInputDesc().GetStrides()[problem.GetDim()];

    auto size        = problem.GetInputDesc().GetElementSize();
    auto output_size = size / dim_size;

    // Start building result.construction_params
    size_t xlocalsize = dim_size >= 8192 ? LOCAL_SIZE * 2 : LOCAL_SIZE;
    size_t xgridsize  = output_size * xlocalsize;
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenKthvalue.cpp";
    kernel.kernel_name = "KthvalueFwd";

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
            decltype(auto) params = raw_params.CastTo<miopen::median::FwdInvokeParams>();

            auto input_tv                      = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto input_tv_without_selected_dim = get_tv_without_dim<5>(input_tv, params.dim);

            auto output_tv  = get_inner_expanded_tv<5>(deref(params.outputDesc));
            auto indices_tv = get_inner_expanded_tv<5>(deref(params.indicesDesc));

            auto dim = params.dim;
            auto k   = (input_lengths[dim] + 1) / 2;

            kernel(params.input,
                   params.output,
                   params.indices,
                   k,
                   dim_size,
                   dim_stride,
                   output_size,
                   input_tv_without_selected_dim,
                   output_tv,
                   indices_tv);
        };
    };

    return result;
}

} // namespace median

} // namespace solver

} // namespace miopen
