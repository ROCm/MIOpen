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
#include <miopen/marginrankingloss/solvers.hpp>

#include <miopen/marginrankingloss/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/marginrankingloss.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace marginrankingloss {

bool MarginRankingLossForward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::marginrankingloss::ProblemDescriptionForward& problem) const
{
    if(!(problem.GetOutputDesc().GetType() == miopenHalf ||
         problem.GetOutputDesc().GetType() == miopenFloat ||
         problem.GetOutputDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    return true;
}

ConvSolution MarginRankingLossForward::GetSolution(
    const ExecutionContext& context,
    const miopen::marginrankingloss::ProblemDescriptionForward& problem) const
{
    std::ignore = context;
    auto result = ConvSolution(miopenStatusSuccess);
    auto dtype  = problem.GetInput1Desc().GetType();
    auto dims   = problem.GetInput1Desc().GetLengths();

    size_t total_elements = problem.GetInput1Desc().GetElementSize();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
    };

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(total_elements, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel         = KernelInfo{};
    kernel.kernel_file  = "MIOpenMarginRankingLoss.cpp";
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    if(problem.GetReductionMode() != MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE)
    {
        float divisor = 1.0f;
        if(problem.GetReductionMode() == MIOPEN_MARGINRANKINGLOSS_REDUCTION_MEAN)
        {
            divisor = static_cast<float>(problem.GetTargetDesc().GetElementSize());
        }

        kernel.kernel_name     = "MarginRankingLossReducedForward5d";
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params =
                    raw_params.CastTo<miopen::marginrankingloss::FwdInvokeParams>();
                auto input1_tv = get_inner_expanded_tv<5>(deref(params.input1Desc));
                auto input2_tv = get_inner_expanded_tv<5>(deref(params.input2Desc));
                auto target_tv = get_inner_expanded_tv<5>(deref(params.targetDesc));
                auto output_tv = get_inner_expanded_tv<5>(deref(params.outputDesc));

                kernel(params.input1,
                       params.input2,
                       params.target,
                       params.output,
                       params.margin,
                       divisor,
                       input1_tv,
                       input2_tv,
                       target_tv,
                       output_tv);
            };
        };
    }
    else
    {
        kernel.kernel_name     = "MarginRankingLossUnreducedForward5d";
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params =
                    raw_params.CastTo<miopen::marginrankingloss::FwdInvokeParams>();
                auto input1_tv = get_inner_expanded_tv<5>(deref(params.input1Desc));
                auto input2_tv = get_inner_expanded_tv<5>(deref(params.input2Desc));
                auto target_tv = get_inner_expanded_tv<5>(deref(params.targetDesc));
                auto output_tv = get_inner_expanded_tv<5>(deref(params.outputDesc));

                kernel(params.input1,
                       params.input2,
                       params.target,
                       params.output,
                       params.margin,
                       input1_tv,
                       input2_tv,
                       target_tv,
                       output_tv);
            };
        };
    }

    result.construction_params.push_back(kernel);
    return result;
}

} // namespace marginrankingloss

} // namespace solver

} // namespace miopen
