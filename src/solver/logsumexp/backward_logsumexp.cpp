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

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/logsumexp/invoke_params.hpp>
#include <miopen/logsumexp/solvers.hpp>
#include <miopen/logsumexp.hpp>
#include "miopen/logsumexp/problem_description.hpp"

#define LOCAL_SIZE 1024

#define VIEW_DIMS 5

namespace miopen {

namespace solver {

namespace logsumexp {

namespace {
bool IsImprovementOverROCmBackward(const miopen::logsumexp::ProblemDescriptionBackward& problem)
{
    constexpr size_t max_input_numel = 1000000;
    constexpr size_t min_input_numel = 300;
    if(problem.GetInputDesc().GetElementSize() > max_input_numel)
        return false;
    if(problem.GetInputDesc().GetElementSize() < min_input_numel)
        return false;

    if(!problem.IsAllPacked())
        return false;

    return true;
}
} // namespace

bool LogSumExpBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::logsumexp::ProblemDescriptionBackward& problem) const
{
    if(!(problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
        return false;
    if(problem.GetInputDesc().GetNumDims() > VIEW_DIMS)
        return false;
    if(!IsImprovementOverROCmBackward(problem))
        return false;
    return true;
}

ConvSolution
LogSumExpBackward::GetSolution(const ExecutionContext& /*context*/,
                               const miopen::logsumexp::ProblemDescriptionBackward& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto dtype = problem.GetInputDesc().GetType();

        auto input_numel = problem.GetInputDesc().GetElementSize();

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(input_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLogSumExp.cpp";
        kernel.kernel_name = "LogSumExpBackward";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"VIEW_DIMS", VIEW_DIMS},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::logsumexp::LogSumExpBackwardInvokeParams>();

            auto input_dims       = deref(params.inputDesc).GetLengths();
            auto input_grad_dims  = deref(params.inputGradDesc).GetLengths();
            auto output_dims      = deref(params.outputDesc).GetLengths();
            auto output_grad_dims = deref(params.outputGradDesc).GetLengths();

            auto input_numel = deref(params.inputDesc).GetElementSize();

            std::vector<int> dims_vector(params.dims, params.dims + params.num_dims);

            auto input_tv       = get_inner_expanded_tv<VIEW_DIMS>(*(params.inputDesc));
            auto input_grad_tv  = get_inner_expanded_tv<VIEW_DIMS>(*(params.inputGradDesc));
            auto output_tv      = get_inner_expanded_tv<VIEW_DIMS>(*(params.outputDesc));
            auto output_grad_tv = get_inner_expanded_tv<VIEW_DIMS>(*(params.outputGradDesc));

            kernel(params.input,
                   params.inputGrad,
                   params.output,
                   params.outputGrad,
                   static_cast<uint64_t>(input_numel),
                   input_tv,
                   input_grad_tv,
                   output_tv,
                   output_grad_tv);
        };
    };

    return result;
}

} // namespace logsumexp

} // namespace solver

} // namespace miopen
