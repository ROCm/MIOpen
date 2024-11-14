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
#include <miopen/marginrankingloss.hpp>
#include <miopen/marginrankingloss/invoke_params.hpp>
#include <miopen/marginrankingloss/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace marginrankingloss {

namespace {

bool IsOverRocmBwd(const miopen::marginrankingloss::ProblemDescriptionBackward& problem)
{
    auto dims             = problem.GetInput1Desc().GetLengths();
    size_t total_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    for(size_t dim : dims)
    {
        if(total_elements == dim)
            return false;
    }
    return true;
}

} // namespace

bool MarginRankingLossBackward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::marginrankingloss::ProblemDescriptionBackward& problem) const
{
    if(!IsOverRocmBwd(problem))
    {
        return false;
    }
    if(!(problem.GetOutGradDesc().GetType() == miopenHalf ||
         problem.GetOutGradDesc().GetType() == miopenFloat ||
         problem.GetOutGradDesc().GetType() == miopenBFloat16))
    {
        return false;
    }
    return true;
}

ConvSolution MarginRankingLossBackward::GetSolution(
    const ExecutionContext& context,
    const miopen::marginrankingloss::ProblemDescriptionBackward& problem) const
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
        {"DTYPE", miopen::GetDataType(dtype) == "bfloat16" ? "ushort" : miopen::GetDataType(dtype)},
        {"REDUCTION_TYPE", static_cast<int>(problem.GetReductionMode())},
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

    uint64_t divisor = 1;
    if(problem.GetReductionMode() == MIOPEN_LOSS_REDUCTION_MEAN)
    {
        divisor = problem.GetTargetDesc().GetElementSize();
    }

    kernel.kernel_name     = "MarginRankingLossBackward5d";
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::marginrankingloss::BwdInvokeParams>();
            auto input1_tv        = get_inner_expanded_tv<5>(deref(params.input1Desc));
            auto input2_tv        = get_inner_expanded_tv<5>(deref(params.input2Desc));
            auto target_tv        = get_inner_expanded_tv<5>(deref(params.targetDesc));
            auto outGrad_tv       = get_inner_expanded_tv<5>(deref(params.outGradDesc));
            auto in1Grad_tv       = get_inner_expanded_tv<5>(deref(params.in1GradDesc));
            auto in2Grad_tv       = get_inner_expanded_tv<5>(deref(params.in2GradDesc));

            kernel(params.input1,
                   params.input2,
                   params.target,
                   params.outGrad,
                   params.in1Grad,
                   params.in2Grad,
                   params.margin,
                   divisor,
                   input1_tv,
                   input2_tv,
                   target_tv,
                   outGrad_tv,
                   in1Grad_tv,
                   in2Grad_tv);
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

} // namespace marginrankingloss

} // namespace solver

} // namespace miopen
