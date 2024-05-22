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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include <miopen/kldivloss/solvers.hpp>

#include <miopen/kldivloss/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kldivloss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view.hpp>

#define LOCAL_SIZE_UNREDUCED_FWD 1024

namespace miopen {

namespace solver {

namespace kldivloss {

bool KLDivLossUnreducedForward5d::IsApplicable(
    const ExecutionContext& context,
    const miopen::kldivloss::UnreducedProblemDescription& problem) const
{
    if(!KLDivLossUnreducedSolver::IsApplicable(context, problem))
        return false;

    return false;
}

ConvSolution KLDivLossUnreducedForward5d::GetSolution(
    const ExecutionContext& context,
    const miopen::kldivloss::UnreducedProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    {
        auto dtype     = problem.GetOutputDesc().GetType();
        size_t N_total = problem.GetNtotal();

        auto kernel = KernelInfo{};

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_UNREDUCED_FWD},
                                                             {N_total},
                                                             "MIOpenKLDivLoss.cpp",
                                                             "KLDivLossUnreducedForward5d",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::kldivloss::FwdInvokeParams>();

            auto input_tv  = get_inner_expanded_tv_5d(deref(params.inputDesc));
            auto target_tv = get_inner_expanded_tv_5d(deref(params.targetDesc));
            auto output_tv = get_inner_expanded_tv_5d(deref(params.outputDesc));

            kernel(params.input,
                   params.target,
                   params.output,
                   params.log_target,
                   input_tv,
                   target_tv,
                   output_tv);
        };
    };

    return result;
}

} // namespace kldivloss

} // namespace solver

} // namespace miopen
