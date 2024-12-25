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

#include <miopen/buffer_info.hpp>
#include <miopen/datatype.hpp>
#include <miopen/allclose.hpp>
#include <miopen/allclose/invoke_params.hpp>
#include <miopen/allclose/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_FWD 256
#define LOCAL_SIZE_REDUCE 256

namespace miopen {

namespace solver {

namespace allclose {

bool AllCloseForward::IsApplicable(const ExecutionContext&,
                                   const miopen::allclose::ProblemDescription& problem) const
{
    if(!(problem.GetInput1Desc().GetType() == miopenHalf ||
         problem.GetInput1Desc().GetType() == miopenFloat ||
         problem.GetInput1Desc().GetType() == miopenBFloat16))
    {
        return false;
    }
    return true;
}

ConvSolution AllCloseForward::GetSolution(const ExecutionContext& context,
                                          const miopen::allclose::ProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto input1_dtype = miopen::GetDataType(problem.GetInput1Desc().GetType());
    auto dtype        = problem.GetInput1Desc().GetType();
    auto numel        = problem.GetInput1Desc().GetElementSize();

    {
        /* Phase 1: Calc AllClose for each element. */
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"D_TYPE", input1_dtype == "bfloat16" ? "ushort" : input1_dtype},
            {"LOCAL_SIZE", LOCAL_SIZE_FWD},
        };

        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_FWD}, {numel}, "MIOpenAllClose.cpp", "AllCloseForward", build_params));
    }

    {
        /* Phase 2: Reduce the results. */
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"D_TYPE", input1_dtype == "bfloat16" ? "ushort" : input1_dtype},
            {"LOCAL_SIZE", LOCAL_SIZE_REDUCE},
        };

        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCE},
                                                             {numel},
                                                             "MIOpenReduceCalculation.cpp",
                                                             "CalculationParallelFwdContiguous",
                                                             build_params));
    }

    result.invoker_factory = [numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::allclose::InvokeParams>();
            auto input1_tv        = get_inner_expanded_tv<5>(deref(params.input1Desc));
            auto input2_tv        = get_inner_expanded_tv<5>(deref(params.input2Desc));

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

            /* Phase 1: Calc AllClose for each element. */
            {
                decltype(auto) kernel = handle_.Run(kernels[0]);

                kernel(params.input1,
                       params.input2,
                       params.atol,
                       params.rtol,
                       params.equal_nan,
                       params.workspace,
                       numel,
                       input1_tv,
                       input2_tv);
            }

            /* Phase 2: Reduce */
            {
                auto size      = numel;
                auto data_size = get_data_size(miopenInt32);
                auto wt        = MultiBufferWorkspaceTraits{size * data_size,
                                                     (size + LOCAL_SIZE_REDUCE - 1) /
                                                         LOCAL_SIZE_REDUCE * data_size};
                auto reduce_in = params.workspace;

                decltype(auto) kernel = handle_.Run(kernels[1]);
                kernel(reduce_in, params.output, size);
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

    return result;
};

std::size_t
AllCloseForward::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                  const miopen::allclose::ProblemDescription& problem) const
{
    auto size      = problem.GetInput1Desc().GetElementSize();
    auto data_size = get_data_size(miopenInt32);
    return MultiBufferWorkspaceTraits{
        size * data_size, (size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE * data_size}
        .GetSize();
}

} // namespace allclose

} // namespace solver

} // namespace miopen
