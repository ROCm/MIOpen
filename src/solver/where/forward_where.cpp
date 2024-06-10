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
#include <../kernels/tensor_view.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace where {

template <int N>
void printTv(const tensor_view_t<N> tv)
{
    std::cout << "Size = ";
    for(int i = 0; i < N; i++)
        std::cout << tv.size[i] << " ";
    std::cout << std::endl;

    std::cout << "Stride = ";
    for(int i = 0; i < N; i++)
        std::cout << tv.stride[i] << " ";
    std::cout << std::endl;
}

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

bool WhereForward::IsApplicable(const ExecutionContext& context,
                                const miopen::where::ForwardProblemDescription& problem) const
{
    std::ignore    = context;
    auto inputDims = problem.GetInputDesc().GetLengths();

    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllPacked())
        return false;
    return true;
}

ConvSolution
WhereForward::GetSolution(const ExecutionContext& context,
                          const miopen::where::ForwardProblemDescription& problem) const
{
    std::ignore = context;

    tensor_view_t<5> input_tv  = get_inner_expanded_tv<5>(problem.GetInputDesc());
    tensor_view_t<5> other_tv  = get_inner_expanded_tv<5>(problem.GetOtherDesc());
    tensor_view_t<5> cond_tv   = get_inner_expanded_tv<5>(problem.GetConditionDesc());
    tensor_view_t<5> output_tv = get_inner_expanded_tv<5>(problem.GetOutputDesc());

    printTv(input_tv);
    printTv(other_tv);
    printTv(cond_tv);
    printTv(output_tv);

    input_tv = broadcast_to(input_tv, output_tv);
    other_tv = broadcast_to(other_tv, output_tv);
    cond_tv  = broadcast_to(cond_tv, output_tv);

    printTv(input_tv);
    printTv(other_tv);
    printTv(cond_tv);
    printTv(output_tv);

    auto cond_contig_size  = check_broadcasted_contiguous(cond_tv);
    auto input_contig_size = check_broadcasted_contiguous(input_tv);
    auto other_contig_size = check_broadcasted_contiguous(other_tv);

    bool is_all_broadcasted_contiguous = cond_contig_size && input_contig_size &&
                                         other_contig_size &&
                                         miopen::where::isContiguous(problem.GetOutputDesc());

    bool is_condition_broadcasted =
        cond_contig_size && ((input_contig_size % cond_contig_size) == 0 ||
                             (other_contig_size % cond_contig_size) == 0);
    std::cout << "cond_contig_size = " << cond_contig_size << " input_contig_size = " << input_contig_size << " other_contig_size = " << other_contig_size << std::endl;
    std::cout << "is_condition_broadcasted = " << is_condition_broadcasted << "is_all_broadcasted_contiguous = " << is_all_broadcasted_contiguous << std::endl;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto outputDims   = problem.GetOutputDesc().GetLengths();

    auto cond_numel   = problem.GetConditionDesc().GetElementSize();
    auto output_numel = problem.GetOutputDesc().GetElementSize();

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenWhere.cpp";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    if(is_all_broadcasted_contiguous && is_condition_broadcasted)
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(cond_numel, xlocalsize);
        std::cout << "cond_contig_size " << cond_contig_size << std::endl;
        std::cout << "input size" << input_contig_size << std::endl;
        std::cout << "other size" << other_contig_size << std::endl;
        std::cout << "output_numel" << output_numel << std::endl;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "WhereConditionBroadcastedContiguousForward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);
    }
    else if(is_all_broadcasted_contiguous)
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "WhereBroadcastedContiguousForward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);
    }

    result.construction_params.push_back(kernel);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::where::InvokeParams>();

            size_t output_numel      = problem.GetOutputDesc().GetElementSize();
            size_t condition_off     = 0;
            size_t input_off         = 0;
            size_t other_off         = 0;
            size_t output_off        = 0;
            size_t cond_contig_size  = check_broadcasted_contiguous(cond_tv);
            size_t input_contig_size = check_broadcasted_contiguous(input_tv);
            size_t other_contig_size = check_broadcasted_contiguous(other_tv);

            kernel(params.condition,
                   params.input,
                   params.other,
                   params.output,
                   output_numel,
                   condition_off,
                   input_off,
                   other_off,
                   output_off,
                   cond_contig_size,
                   input_contig_size,
                   other_contig_size);
        };
    };

    return result;
}

} // namespace where

} // namespace solver

} // namespace miopen
