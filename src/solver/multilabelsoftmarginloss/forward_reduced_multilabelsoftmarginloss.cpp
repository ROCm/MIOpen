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
#include <miopen/multilabelsoftmarginloss/invoke_params.hpp>
#include <miopen/multilabelsoftmarginloss/solvers.hpp>
#include <miopen/multilabelsoftmarginloss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_MULTILABELSOFTMARGINLOSS 256
#define LOCAL_SIZE_REDUCE 256

namespace miopen {

namespace solver {

namespace multilabelsoftmarginloss {

bool MultilabelSoftMarginLossForward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::multilabelsoftmarginloss::ForwardProblemDescription& problem) const
{
    if(problem.GetiDesc().GetLengths()[1] > 24)
        return false;
    return true;
}

ConvSolution MultilabelSoftMarginLossForward::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::multilabelsoftmarginloss::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto xgrid = problem.GetiDesc().GetLengths()[0];

    {
        /* Phase 1: Calc loss for each element. */
        auto dtype        = problem.GetiDesc().GetType();
        size_t xlocalsize = LOCAL_SIZE_MULTILABELSOFTMARGINLOSS;
        size_t xgridsize  = AlignUp(xgrid, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenMultilabelSoftMarginLoss.cpp";
        kernel.kernel_name = "MultilabelSoftMarginLossForward2d";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            {"INPUT_TYPE",
             (dtype == miopenBFloat16) ? "ushort"
             : (dtype == miopenHalf)   ? "_Float16"
                                       : "float"},
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

    {
        /* Phase 2: Reduce */
        auto _size              = xgrid;
        auto dtype              = problem.GetiDesc().GetType();
        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            {"INPUT_TYPE",
             (dtype == miopenBFloat16) ? "ushort"
             : (dtype == miopenHalf)   ? "_Float16"
                                       : "float"},
            {"REDUCE_SIZE", LOCAL_SIZE_REDUCE},
        };
        do
        {
            size_t xlocalsize = LOCAL_SIZE_REDUCE;
            size_t xgridsize  = AlignUp(_size, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenLossReduce.cpp";
            kernel.kernel_name = "ReduceSumLoss";

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            result.construction_params.push_back(kernel);
            _size = AlignUp(_size, LOCAL_SIZE_REDUCE) / LOCAL_SIZE_REDUCE;
        } while(_size > 1);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::multilabelsoftmarginloss::InvokeParams>();
            auto i_tv = get_inner_expanded_tv<2>(deref(params.iDesc));
            auto t_tv = get_inner_expanded_tv<2>(deref(params.tDesc));
            auto w_tv = get_inner_expanded_tv<1>(deref(params.wDesc));

            float elapsed = 0.0f;
            HipEventPtr start;
            HipEventPtr stop;

            bool reset_profiling_state = false;
            if(handle_.IsProfilingEnabled())
            {
                reset_profiling_state = true;
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }
            /* Phase 1: Calc loss for each element. */
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                kernel(params.i,
                       params.t,
                       params.w,
                       params.workspace,
                       params.divisor,
                       i_tv,
                       t_tv,
                       w_tv);
            }

            /* Phase 2: Reduce */
            auto size      = deref(params.iDesc).GetLengths()[0];
            auto reduce_in = params.workspace;
            auto reduce_out =
                static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                    size * get_data_size(deref(params.oDesc).GetType()));

            for(int i = 1; i < kernels.size(); ++i)
            {
                decltype(auto) kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(reduce_in, reduce_out, size);
                    std::swap(reduce_in, reduce_out);
                }
                else
                {
                    kernel(reduce_in, params.o, size);
                }
                size = AlignUp(size, LOCAL_SIZE_REDUCE) / LOCAL_SIZE_REDUCE;
            }

            if(reset_profiling_state)
                handle_.EnableProfiling(true);
            if(handle_.IsProfilingEnabled())
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());

                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

std::size_t MultilabelSoftMarginLossForward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::multilabelsoftmarginloss::ForwardProblemDescription& problem) const
{
    auto elem = problem.GetiDesc().GetLengths()[0];
    return (elem + (elem + LOCAL_SIZE_REDUCE) / LOCAL_SIZE_REDUCE) *
           get_data_size(problem.GetoDesc().GetType());
}

} // namespace multilabelsoftmarginloss

} // namespace solver

} // namespace miopen
