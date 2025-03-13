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

#include "miopen/hingeembeddingloss/problem_description.hpp"
#include <miopen/mlo_internal.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/hingeembeddingloss/invoke_params.hpp>
#include <miopen/hingeembeddingloss/solvers.hpp>
#include <miopen/hinge_embedding_loss.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_HINGE_EMBEDDING_LOSS 256

namespace miopen {

namespace solver {

namespace hingeembeddingloss {

bool HingeEmbeddingLossBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::hingeembeddingloss::BackwardProblemDescription& problem) const
{
    if(!(problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
        return false;
    return true;
}

ConvSolution HingeEmbeddingLossBackward::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::hingeembeddingloss::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    // Start building result.construction_params
    auto xgrid         = problem.GetInputDesc().GetElementSize();
    auto dtype         = problem.GetInputDesc().GetType();
    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenHingeEmbeddingLoss.cpp";
    kernel.kernel_name = "HingeEmbeddingLossBackward";

    size_t xlocalsize = LOCAL_SIZE_HINGE_EMBEDDING_LOSS;
    size_t xgridsize  = AlignUp(xgrid, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);
    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
        {"REDUCTION_TYPE", static_cast<int>(problem.GetReduction())},
    };
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::hingeembeddingloss::InvokeParams>();

            auto i_tv  = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto t_tv  = get_inner_expanded_tv<5>(deref(params.targetDesc));
            auto do_tv = get_inner_expanded_tv<5>(deref(params.doutputDesc));
            auto di_tv = get_inner_expanded_tv<5>(deref(params.dinputDesc));

            kernel(params.input,
                   params.target,
                   params.doutput,
                   params.dinput,
                   params.inputDesc->GetElementSize(),
                   params.margin,
                   i_tv,
                   t_tv,
                   do_tv,
                   di_tv);
        };
    };

    return result;
}

} // namespace hingeembeddingloss

} // namespace solver

} // namespace miopen
