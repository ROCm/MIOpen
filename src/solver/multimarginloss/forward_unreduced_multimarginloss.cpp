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

#include "miopen/miopen.h"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/multimarginloss/invoke_params.hpp>
#include <miopen/multimarginloss/solvers.hpp>
#include <miopen/multimarginloss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace multimarginloss {

bool MultiMarginLossUnreducedForward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::multimarginloss::ForwardProblemDescription& problem) const
{
    // TODO: edit later
    // if(problem.GetiDesc().GetLengths()[1] > 24)
    //     return false;
    return true;
}

ConvSolution MultiMarginLossUnreducedForward::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::multimarginloss::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto xgrid = problem.GetiDesc().GetLengths()[0];

    {
        auto dtype        = problem.GetiDesc().GetType();
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(xgrid, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenMultiMarginLoss.cpp";
        kernel.kernel_name = "MultiMarginLossUnreducedForward2d";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
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
            decltype(auto) params = raw_params.CastTo<miopen::multimarginloss::InvokeParams>();

            auto i_tv = get_inner_expanded_tv<2>(deref(params.iDesc));
            auto t_tv = get_inner_expanded_tv<1>(deref(params.tDesc));
            auto w_tv = get_inner_expanded_tv<1>(deref(params.wDesc));
            auto o_tv = get_inner_expanded_tv<1>(deref(params.oDesc));

            kernel(params.i,
                   params.t,
                   params.w,
                   params.o,
                   params.p,
                   params.margin,
                   i_tv,
                   t_tv,
                   w_tv,
                   o_tv);
        };
    };

    return result;
}

} // namespace multimarginloss

} // namespace solver

} // namespace miopen
