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

#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include "miopen/tensor.hpp"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/normalize/invoke_params.hpp>
#include <miopen/normalize/solvers.hpp>
#include <miopen/normalize.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace normalize {

bool IsImprovementOverROCm(const ExecutionContext& context,
                           const miopen::normalize::BackwardProblemDescription& problem)
{
    auto outer_size = problem.GetInputDesc().GetElementSize() / problem.GetInnerSize();
    return (problem.IsLastDim() && (problem.GetInnerSize() % LOCAL_SIZE == 0) &&
            problem.IsAllContiguous() && outer_size >= context.GetStream().GetMaxComputeUnits());
}

bool NormalizeBackward::IsApplicable(
    const ExecutionContext& context,
    const miopen::normalize::BackwardProblemDescription& problem) const
{
    if(!(problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
        return false;
    if(!IsImprovementOverROCm(context, problem))
        return false;
    return true;
}

ConvSolution
NormalizeBackward::GetSolution(const ExecutionContext& /*context*/,
                               const miopen::normalize::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    // Start building result.construction_params
    auto num_elem   = problem.GetInputDesc().GetElementSize();
    auto inner_size = problem.GetInnerSize();
    auto outer_size = num_elem / inner_size;
    auto dtype      = problem.GetInputDesc().GetType();

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
                              {"LOCAL_SIZE", LOCAL_SIZE}};
    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenNormalize.cpp";
        kernel.kernel_name = "NormalizeReduceContiguous";

        size_t xlocalsize = 1;
        size_t xgridsize  = outer_size;
        size_t ylocalsize = LOCAL_SIZE;
        size_t ygridsize  = LOCAL_SIZE;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        result.construction_params.push_back(kernel);
    }

    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenNormalize.cpp";
        kernel.kernel_name = "NormalizeBackwardOpt";

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(num_elem, xlocalsize);
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

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        result.construction_params.push_back(kernel);
    }
    // End building result.construction_params

    // Start building result.invoker_factory
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) params = raw_params.CastTo<miopen::normalize::InvokeParams>();
                auto input_tv         = get_inner_expanded_tv<5>(deref(params.inputDesc));
                auto divisor_tv       = get_inner_expanded_tv<5>(deref(params.divisorDesc));

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

                {
                    decltype(auto) kernel = handle_.Run(kernels[0]);
                    auto inner_size       = params.inputDesc->GetLengths()[params.dim];

                    kernel(params.input, params.outputGrad, params.workspace, inner_size);
                }

                {
                    decltype(auto) kernel = handle_.Run(kernels[1]);
                    auto num_elem         = params.inputDesc->GetElementSize();
                    kernel(params.input,
                           params.divisor,
                           params.outputGrad,
                           params.inputGrad,
                           params.workspace,
                           params.p,
                           params.eps,
                           num_elem,
                           params.dim,
                           input_tv,
                           divisor_tv);
                }

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

    return result;
}

std::size_t NormalizeBackward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::normalize::BackwardProblemDescription& problem) const
{
    return problem.GetInputDesc().GetElementSize() / problem.GetInnerSize() *
           get_data_size(problem.GetInputDesc().GetType());
}

} // namespace normalize

} // namespace solver

} // namespace miopen
