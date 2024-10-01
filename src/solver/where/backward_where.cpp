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

#include <cstddef>

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/where/invoke_params.hpp>
#include <miopen/where/problem_description.hpp>
#include <miopen/where/solvers.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace where {

bool WhereBackward::IsApplicable(const ExecutionContext& context,
                                 const miopen::where::BackwardProblemDescription& problem) const
{
    auto ignore = context;

    bool is_all_contiguous             = problem.isAllContiguous();
    bool is_all_broadcasted_contiguous = problem.isAllBroadcastedContiguous();

    if(!is_all_broadcasted_contiguous && !is_all_contiguous)
        return false;
    return true;
}

ConvSolution
WhereBackward::GetSolution(const ExecutionContext& context,
                           const miopen::where::BackwardProblemDescription& problem) const
{
    std::ignore = context;

    auto cond_contig_size  = miopen::where::checkBroadcastedContiguous(problem.GetCondition_tv());
    bool is_all_same_shape = problem.isAllSameShape();
    bool is_all_broadcasted_contiguous = problem.isAllBroadcastedContiguous();
    bool is_condition_broadcasted      = problem.isConditionBroadcasted();

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype           = problem.GetInputGradDesc().GetType();
    auto io_dtype        = miopen::GetDataType(dtype);
    auto outputGradNumel = problem.GetOutputGradDesc().GetElementSize();
    auto kernel          = KernelInfo{};
    kernel.kernel_file   = "MIOpenWhere.cpp";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    if(is_all_same_shape) // no broadcast
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(outputGradNumel, xlocalsize);

        kernel.kernel_name = "WhereContiguousBackward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [outputGradNumel](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::where::BwdInvokeParams>();

                kernel(params.condition,
                       params.outputGrad,
                       params.inputGrad,
                       params.otherGrad,
                       outputGradNumel);
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

                size_t output_grad_numel = problem.GetOutputGradDesc().GetElementSize();
                size_t cond_contig_size =
                    miopen::where::checkBroadcastedContiguous(problem.GetCondition_tv());
                size_t input_grad_contig_size =
                    miopen::where::checkBroadcastedContiguous(problem.GetInputGrad_tv());
                size_t other_grad_contig_size =
                    miopen::where::checkBroadcastedContiguous(problem.GetOtherGrad_tv());

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

                size_t output_grad_numel = problem.GetOutputGradDesc().GetElementSize();
                size_t cond_contig_size =
                    miopen::where::checkBroadcastedContiguous(problem.GetCondition_tv());
                size_t input_grad_contig_size =
                    miopen::where::checkBroadcastedContiguous(problem.GetInputGrad_tv());
                size_t other_grad_contig_size =
                    miopen::where::checkBroadcastedContiguous(problem.GetOtherGrad_tv());

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
