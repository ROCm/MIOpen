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

#include "miopen/mlo_internal.hpp"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/softmarginloss/invoke_params.hpp>
#include <miopen/softmarginloss/solvers.hpp>
#include <miopen/softmarginloss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace softmarginloss {

bool SoftMarginLossBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::softmarginloss::BackwardProblemDescription& problem) const
{
    if(!(problem.GetiDesc().GetType() == miopenFloat ||
         problem.GetiDesc().GetType() == miopenHalf ||
         problem.GetiDesc().GetType() == miopenBFloat16))
        return false;
    return true;
}

ConvSolution SoftMarginLossBackward::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::softmarginloss::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto elem = problem.GetiDesc().GetElementSize();

    {
        auto dtype        = problem.GetiDesc().GetType();
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(elem, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenSoftMarginLoss.cpp";
        kernel.kernel_name = "SoftMarginLossBackward5d";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            {"REDUCTION_TYPE", static_cast<int>(problem.Getreduction())},
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
            decltype(auto) params = raw_params.CastTo<miopen::softmarginloss::InvokeParams>();

            auto i_tv  = get_inner_expanded_tv<5>(deref(params.iDesc));
            auto t_tv  = get_inner_expanded_tv<5>(deref(params.tDesc));
            auto dO_tv = get_inner_expanded_tv<5>(deref(params.dODesc));
            auto dI_tv = get_inner_expanded_tv<5>(deref(params.dIDesc));
            kernel(params.i,
                   params.t,
                   params.dO,
                   params.dI,
                   params.iDesc->GetElementSize(),
                   i_tv,
                   t_tv,
                   dO_tv,
                   dI_tv);
        };
    };

    return result;
}

} // namespace softmarginloss

} // namespace solver

} // namespace miopen
