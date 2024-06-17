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

#include "miopen/kernel_info.hpp"
#include "miopen/mlo_internal.hpp"
#include "miopen/tensor.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "miopen/where/problem_description.hpp"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/where/invoke_params.hpp>
#include <miopen/where/solvers.hpp>
#include <miopen/where.hpp>
#include <miopen/target_properties.hpp>
#include "../kernels/tensor_view.hpp"

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace where {

template <int N>
int64_t check_broadcasted_contiguous(const tensor_view_t<N>& tensorView)
{
    int64_t num_elems = 1;

    for(int i = N - 1; i >= 0; i--)
    {
        if(tensorView.stride[i] != 0 && tensorView.stride[i] != num_elems)
            return 0;
        if(tensorView.stride[i] == 0)
        {
            for(int j = i; j >= 0; j--)
                if(tensorView.stride[j] != 0)
                    return 0;
            return num_elems;
        }
        num_elems *= tensorView.size[i];
    }

    return num_elems;
}

bool WhereBackward::IsApplicable(const ExecutionContext& context,
                                 const miopen::where::BackwardProblemDescription& problem) const
{
    std::ignore                     = context;
    tensor_view_t<5> input_grad_tv  = get_inner_expanded_tv<5>(problem.GetInputGradDesc());
    tensor_view_t<5> other_grad_tv  = get_inner_expanded_tv<5>(problem.GetOtherGradDesc());
    tensor_view_t<5> cond_tv        = get_inner_expanded_tv<5>(problem.GetConditionDesc());
    tensor_view_t<5> output_grad_tv = get_inner_expanded_tv<5>(problem.GetOutputGradDesc());

    input_grad_tv = broadcast_to(input_grad_tv, output_grad_tv);
    other_grad_tv = broadcast_to(other_grad_tv, output_grad_tv);
    cond_tv       = broadcast_to(cond_tv, output_grad_tv);

    auto cond_contig_size        = check_broadcasted_contiguous(cond_tv);
    auto input_grad_contig_size  = check_broadcasted_contiguous(input_grad_tv);
    auto other_grad_contig_size  = check_broadcasted_contiguous(other_grad_tv);
    auto output_grad_contig_size = check_broadcasted_contiguous(output_grad_tv);

    auto is_all_contiguous =
        isTensorViewContiguous(input_grad_tv) && isTensorViewContiguous(other_grad_tv) &&
        isTensorViewContiguous(cond_tv) && isTensorViewContiguous(output_grad_tv);

    bool is_all_broadcasted_contiguous = cond_contig_size && output_grad_contig_size &&
                                         input_grad_contig_size && other_grad_contig_size;

    if(!is_all_broadcasted_contiguous && !is_all_contiguous)
        return false;
    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllPacked())
        return false;
    return true;
}

ConvSolution
WhereBackward::GetSolution(const ExecutionContext& context,
                           const miopen::where::BackwardProblemDescription& problem) const
{
    std::ignore = context;

    tensor_view_t<5> input_grad_tv  = get_inner_expanded_tv<5>(problem.GetInputGradDesc());
    tensor_view_t<5> other_grad_tv  = get_inner_expanded_tv<5>(problem.GetOtherGradDesc());
    tensor_view_t<5> cond_tv        = get_inner_expanded_tv<5>(problem.GetConditionDesc());
    tensor_view_t<5> output_grad_tv = get_inner_expanded_tv<5>(problem.GetOutputGradDesc());

    input_grad_tv = broadcast_to(input_grad_tv, output_grad_tv);
    other_grad_tv = broadcast_to(other_grad_tv, output_grad_tv);
    cond_tv       = broadcast_to(cond_tv, output_grad_tv);

    auto cond_contig_size        = check_broadcasted_contiguous(cond_tv);
    auto input_grad_contig_size  = check_broadcasted_contiguous(input_grad_tv);
    auto other_grad_contig_size  = check_broadcasted_contiguous(other_grad_tv);
    auto output_grad_contig_size = check_broadcasted_contiguous(output_grad_tv);

    auto is_all_contiguous =
        isTensorViewContiguous(input_grad_tv) && isTensorViewContiguous(other_grad_tv) &&
        isTensorViewContiguous(cond_tv) && isTensorViewContiguous(output_grad_tv);

    bool is_all_broadcasted_contiguous = cond_contig_size && output_grad_contig_size &&
                                         input_grad_contig_size && other_grad_contig_size;

    bool is_condition_broadcasted =
        cond_contig_size && ((input_grad_contig_size % cond_contig_size) == 0 ||
                             (other_grad_contig_size % cond_contig_size) == 0);

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype           = problem.GetOtherGradDesc().GetType();
    auto input_dtype     = miopen::GetDataType(problem.GetOutputGradDesc().GetType());
    auto output_dtype    = miopen::GetDataType(problem.GetInputGradDesc().GetType());
    auto outputGradNumel = problem.GetOutputGradDesc().GetElementSize();
    auto kernel          = KernelInfo{};
    kernel.kernel_file   = "MIOpenWhere.cpp";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    if(is_all_contiguous)
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(outputGradNumel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "WhereContiguousBackward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::where::BwdInvokeParams>();

                size_t output_grad_numel = problem.GetOutputGradDesc().GetElementSize();

                kernel(params.condition,
                       params.outputGrad,
                       params.inputGrad,
                       params.otherGrad,
                       output_grad_numel);
            };
        };
    }
    else if(is_condition_broadcasted && is_all_broadcasted_contiguous)
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(cond_contig_size, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "WhereConditionBroadcastedContiguousBackward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::where::BwdInvokeParams>();

                size_t output_grad_numel      = problem.GetOutputGradDesc().GetElementSize();
                size_t cond_contig_size       = check_broadcasted_contiguous(cond_tv);
                size_t input_grad_contig_size = check_broadcasted_contiguous(input_grad_tv);
                size_t other_grad_contig_size = check_broadcasted_contiguous(other_grad_tv);

                kernel(params.condition,
                       params.outputGrad,
                       params.inputGrad,
                       params.otherGrad,
                       output_grad_numel,
                       cond_contig_size,
                       input_grad_contig_size,
                       other_grad_contig_size);
            };
        };
    }
    else if(is_all_broadcasted_contiguous)
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(outputGradNumel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "WhereBroadcastedContiguousBackward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::where::BwdInvokeParams>();

                size_t output_grad_numel      = problem.GetOutputGradDesc().GetElementSize();
                size_t cond_contig_size       = check_broadcasted_contiguous(cond_tv);
                size_t input_grad_contig_size = check_broadcasted_contiguous(input_grad_tv);
                size_t other_grad_contig_size = check_broadcasted_contiguous(other_grad_tv);

                kernel(params.condition,
                       params.outputGrad,
                       params.inputGrad,
                       params.otherGrad,
                       output_grad_numel,
                       cond_contig_size,
                       input_grad_contig_size,
                       other_grad_contig_size);
            };
        };
    }

    return result;
}

} // namespace where

} // namespace solver

} // namespace miopen
