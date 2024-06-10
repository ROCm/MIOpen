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
#include "miopen/where/problem_description.hpp"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/where/invoke_params.hpp>
#include <miopen/where/solvers.hpp>
#include <miopen/where.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace where {

int64_t check_broadcasted_contiguous(const TensorDescriptor& tensorDesc)
{
    int64_t num_elems = 1;
    auto len          = tensorDesc.GetLengths();
    auto strides      = tensorDesc.GetStrides();

    for(int i = len.size() - 1; i >= 0; i--)
    {
        if(strides[i] != 0 && strides[i] != num_elems)
            return 0;
        if(strides[i] == 0)
        {
            for(int j = i; j >= 0; j--)
                if(strides[j] != 0)
                    return 0;
            return num_elems;
        }
        num_elems *= len[i];
    }

    return num_elems;
}

/*
bool WhereBackward::IsApplicable(const ExecutionContext& context,
                                const miopen::where::BackwardProblemDescription& problem) const
{
    std::ignore    = context;
    auto inputDims = problem.GetInputGradDesc().GetLengths();

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

    auto cond_contig_size  = check_broadcasted_contiguous(problem.GetConditionDesc());
    auto input_contig_size = check_broadcasted_contiguous(problem.GetInputDesc());
    auto other_contig_size = check_broadcasted_contiguous(problem.GetOtherDesc());

    bool is_all_broadcasted_contiguous = cond_contig_size && input_contig_size &&
                                         other_contig_size &&
                                         miopen::where::isContiguous(problem.GetOutputDesc());

    bool is_condition_broadcasted =
        cond_contig_size && ((input_contig_size % cond_contig_size) == 0 ||
                             (other_contig_size % cond_contig_size) == 0);

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto outputDims   = problem.GetOutputDesc().GetLengths();
    auto output_numel =
        std::accumulate(outputDims.begin(), outputDims.end(), 1ULL, std::multiplies<size_t>());

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
        size_t xgridsize  = AlignUp(cond_contig_size, xlocalsize);
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

    result.invoker_factory =
        [=, &cond_contig_size, &input_contig_size, &other_contig_size, &output_numel](
            const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::where::InvokeParams>();

                kernel(params.condition,
                       params.input,
                       params.other,
                       params.output,
                       output_numel,
                       0,
                       0,
                       0,
                       0,
                       cond_contig_size,
                       input_contig_size,
                       other_contig_size);
            };
        };

    return result;
}
*/

} // namespace where

} // namespace solver

} // namespace miopen
