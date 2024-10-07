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

#include "miopen/buffer_info.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/multimarginloss/invoke_params.hpp>
#include <miopen/multimarginloss/solvers.hpp>
#include <miopen/multimarginloss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_MULTIMARGINLOSS 256
#define LOCAL_SIZE_REDUCE 256

namespace miopen {

namespace solver {

namespace multimarginloss {

bool MultiMarginLossForward::IsImprovementOverROCm(
    const ExecutionContext& /*context*/,
    const miopen::multimarginloss::ForwardProblemDescription& problem) const
{
    int C = problem.GetiDesc().GetLengths()[1];
    if(problem.allContiguousTensor())
    {
        switch(problem.GetiDesc().GetType())
        {
        case miopenFloat: return C <= 33;
        case miopenHalf: return C <= 43;
        case miopenBFloat16: return C <= 44;
        default: return true;
        }
    }
    else
    {
        switch(problem.GetiDesc().GetType())
        {
        case miopenFloat: return C <= 31;
        case miopenHalf: return C <= 38;
        case miopenBFloat16: return C <= 40;
        default: return true;
        }
    }
}

bool MultiMarginLossForward::IsApplicable(
    const ExecutionContext& context,
    const miopen::multimarginloss::ForwardProblemDescription& problem) const
{
    if(!(problem.GetiDesc().GetType() == miopenFloat ||
         problem.GetiDesc().GetType() == miopenHalf ||
         problem.GetiDesc().GetType() == miopenBFloat16))
        return false;
    if(!IsImprovementOverROCm(context, problem))
        return false;
    return true;
}

ConvSolution MultiMarginLossForward::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::multimarginloss::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    // Start building result.construction_params
    auto xgrid = problem.GetiDesc().GetLengths()[0];
    auto dtype = problem.GetiDesc().GetType();
    {
        /* Phase 1: Calc loss for each element. */
        size_t xlocalsize = LOCAL_SIZE_MULTIMARGINLOSS;
        size_t xgridsize  = AlignUp(xgrid, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenMultiMarginLoss.cpp";
        kernel.kernel_name = "MultiMarginLossForward2d";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
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

    if(problem.Getreduction() != MIOPEN_LOSS_REDUCTION_NONE)
    {
        // If Reduction = NONE, then we should run second kernel to calculate mean/sum of result
        // from first kernel above
        /* Phase 2: Reduce FLOAT_ACCUM -> FLOAT_ACCUM */
        auto _size              = xgrid;
        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            {"REDUCE_SIZE", LOCAL_SIZE_REDUCE},
        };
        while(_size > LOCAL_SIZE_REDUCE)
        {
            size_t xlocalsize = LOCAL_SIZE_REDUCE;
            size_t xgridsize  = AlignUp(_size, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenReduceSum.cpp";
            kernel.kernel_name = "ReduceSumFLOATACCUM";

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            result.construction_params.push_back(kernel);
            _size = (_size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE;
        }

        // Last kernel reduce: FLOAT_ACCUM -> FLOAT
        size_t xlocalsize = LOCAL_SIZE_REDUCE;
        size_t xgridsize  = AlignUp(_size, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenReduceSum.cpp";
        kernel.kernel_name = "ReduceSum";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }
    // End building result.construction_params

    // Start building result.invoker_factory
    if(problem.Getreduction() == MIOPEN_LOSS_REDUCTION_NONE)
    {
        // Reduction = None -> invoke 1 kernel
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
    }
    else
    {
        // Reduction != None -> invoke 2 kernels
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) params = raw_params.CastTo<miopen::multimarginloss::InvokeParams>();
                auto i_tv             = get_inner_expanded_tv<2>(deref(params.iDesc));
                auto t_tv             = get_inner_expanded_tv<1>(deref(params.tDesc));
                auto w_tv             = get_inner_expanded_tv<1>(deref(params.wDesc));
                auto o_tv             = get_inner_expanded_tv<1>(deref(params.oDesc));

                float elapsed = 0.0f;
                HipEventPtr start;
                HipEventPtr stop;

                const bool profiling = handle_.IsProfilingEnabled();
                if(profiling)
                {
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
                           params.p,
                           params.margin,
                           i_tv,
                           t_tv,
                           w_tv,
                           o_tv);
                }

                /* Phase 2: Reduce */
                auto size       = deref(params.iDesc).GetLengths()[0];
                auto data_size  = get_data_size(miopenFloat);
                auto wt         = MultiBufferWorkspaceTraits{size * data_size,
                                                     (size + LOCAL_SIZE_REDUCE - 1) /
                                                         LOCAL_SIZE_REDUCE * data_size};
                auto reduce_in  = params.workspace;
                auto reduce_out = static_cast<Data_t>(static_cast<std::byte*>(params.workspace) +
                                                      wt.GetOffset(1));

                int kernelCnt = 1;
                for(kernelCnt; kernelCnt < kernels.size() - 1; kernelCnt++)
                {
                    decltype(auto) kernel = handle_.Run(kernels[kernelCnt]);
                    kernel(reduce_in, reduce_out, size);
                    std::swap(reduce_in, reduce_out);
                    size = (size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE;
                }

                decltype(auto) kernel = handle_.Run(kernels[kernelCnt]);
                kernel(reduce_in, params.o, size, o_tv);

                if(profiling)
                {
                    hipEventRecord(stop.get(), handle_.GetStream());
                    hipEventSynchronize(stop.get());
                    hipEventElapsedTime(&elapsed, start.get(), stop.get());

                    // Clean up
                    hipEventDestroy(start.get());
                    hipEventDestroy(stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);

                    handle_.EnableProfiling(true);
                };
            };
        };
    }
    // End building result.invoker_factory

    return result;
}

std::size_t MultiMarginLossForward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::multimarginloss::ForwardProblemDescription& problem) const
{
    if(problem.Getreduction() == MIOPEN_LOSS_REDUCTION_NONE)
        return 0;

    auto size      = problem.GetiDesc().GetLengths()[0];
    auto data_size = get_data_size(miopenFloat);
    return MultiBufferWorkspaceTraits{
        size * data_size, (size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE * data_size}
        .GetSize();
}

} // namespace multimarginloss

} // namespace solver

} // namespace miopen
