/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/kldivloss.hpp>
#include <miopen/kldivloss/invoke_params.hpp>
#include <miopen/kldivloss/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_REDUCED_BWD 1024

namespace miopen {

namespace solver {

namespace kldivloss {

bool KLDivLossBackward5d::IsApplicable(
    const ExecutionContext&, const miopen::kldivloss::BwdProblemDescription& problem) const
{
    if(!(problem.GetOutputGradDesc().GetType() == miopenHalf ||
         problem.GetOutputGradDesc().GetType() == miopenFloat ||
         problem.GetOutputGradDesc().GetType() == miopenBFloat16))
    {
        return false;
    }
    return true;
}

ConvSolution
KLDivLossBackward5d::GetSolution(const ExecutionContext& context,
                                 const miopen::kldivloss::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto output_dtype = miopen::GetDataType(problem.GetOutputGradDesc().GetType());

    {
        auto dtype     = problem.GetOutputGradDesc().GetType();
        size_t N_total = problem.GetNtotal();

        auto kernel = KernelInfo{};

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"REDUCTION_TYPE", static_cast<int>(problem.GetReductionMode())},
        };

        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCED_BWD},
                                                             {N_total},
                                                             "MIOpenKLDivLoss.cpp",
                                                             "KLDivLossBackward5d",
                                                             build_params));
    }

    uint64_t divisor = 1;
    if(problem.GetReductionMode() == MIOPEN_LOSS_REDUCTION_MEAN)
    {
        divisor = problem.GetTargetDesc().GetElementSize();
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::kldivloss::BwdInvokeParams>();

            auto input_tv       = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto target_tv      = get_inner_expanded_tv<5>(deref(params.targetDesc));
            auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.outputGradDesc));
            auto input_grad_tv  = get_inner_expanded_tv<5>(deref(params.inputGradDesc));
            auto target_grad_tv = get_inner_expanded_tv<5>(deref(params.targetGradDesc));

            kernel(params.input,
                   params.target,
                   params.output_grad,
                   params.input_grad,
                   params.target_grad,
                   divisor,
                   params.log_target,
                   input_tv,
                   target_tv,
                   output_grad_tv,
                   input_grad_tv,
                   target_grad_tv);
        };
    };

    return result;
}

} // namespace kldivloss

} // namespace solver

} // namespace miopen
