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
#include <miopen/prelu/invoke_params.hpp>
#include <miopen/prelu/solvers.hpp>
#include <miopen/prelu/utils.hpp>

#define FLOAT_ACCUM float
#define VIEW_DIMS 5

#define LOCAL_SIZE_SW_BWD 256
#define LOCAL_SIZE_SW_REDUCE_BWD 256
#define LOCAL_SIZE_MW_REDUCE_BWD warpSize

namespace miopen {

namespace solver {

namespace prelu {

bool SingleWeightBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::prelu::BackwardProblemDescription& problem) const
{
    if(!problem.IsSingleWeight())
        return false;
    return true;
}

ConvSolution
SingleWeightBackward::GetSolution(const ExecutionContext& /*context*/,
                                  const miopen::prelu::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetdInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetdInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetdOuputDesc().GetType());

    /* Phase 1: Calc gradient for each elements. */
    {
        auto size         = problem.GetdInputDesc().GetElementSize();
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"VIEW_DIMS", VIEW_DIMS},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_SW_BWD}, {size}, "MIOpenPReLU.cpp", "PReLUSWBackward", build_params));
    }

    /* Phase 2: Reduce gradient. */
    {
        auto size         = problem.GetdInputDesc().GetElementSize();
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"OUTPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"REDUCE_SIZE", LOCAL_SIZE_SW_REDUCE_BWD},
        };
        while(size > LOCAL_SIZE_SW_REDUCE_BWD)
        {
            result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_SW_REDUCE_BWD},
                                                                 {size},
                                                                 "MIOpenReduceSum.cpp",
                                                                 "ReduceSumFLOATACCUM",
                                                                 build_params));
            size = (size + LOCAL_SIZE_SW_REDUCE_BWD - 1) / LOCAL_SIZE_SW_REDUCE_BWD;
        }
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_SW_REDUCE_BWD}, {size}, "MIOpenReduceSum.cpp", "ReduceSum", build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::prelu::InvokeParams>();

            auto elapsed = 0.0f;
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

            int kernelCnt = 0;
            auto work_a   = params.workspace;
            auto work_b   = reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                                   deref(params.inputDesc).GetElementSize() *
                                                       sizeof(FLOAT_ACCUM));

            /* Phase 1: Calc gradient for each elements. */
            {
                auto input_tv       = get_inner_expanded_tv<VIEW_DIMS>(deref(params.inputDesc));
                auto output_grad_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.doutputDesc));
                auto input_grad_tv  = get_inner_expanded_tv<VIEW_DIMS>(deref(params.dinputDesc));
                auto kernel         = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input,
                       params.weight,
                       params.doutput,
                       params.dinput,
                       work_a,
                       (ulong)deref(params.inputDesc).GetElementSize(),
                       input_tv,
                       output_grad_tv,
                       input_grad_tv);
            }

            /* Phase 2: Reduce gradient. */
            {
                auto size = deref(params.inputDesc).GetElementSize();
                while(size > LOCAL_SIZE_SW_REDUCE_BWD)
                {
                    auto kernel = handle_.Run(kernels[kernelCnt++]);
                    kernel(work_a, work_b, size);
                    size = (size + LOCAL_SIZE_SW_REDUCE_BWD - 1) / LOCAL_SIZE_SW_REDUCE_BWD;
                    std::swap(work_a, work_b);
                }
                handle_.Run(kernels[kernelCnt++])(work_a, params.dweight, size);
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

std::size_t SingleWeightBackward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::prelu::BackwardProblemDescription& problem) const
{
    auto size = problem.GetdInputDesc().GetElementSize();
    if(size > LOCAL_SIZE_SW_REDUCE_BWD)
        size += (size + LOCAL_SIZE_SW_REDUCE_BWD - 1) / LOCAL_SIZE_SW_REDUCE_BWD;
    size *= sizeof(FLOAT_ACCUM);
    return size;
}

bool MultiWeightsBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::prelu::BackwardProblemDescription& problem) const
{
    if(problem.IsSingleWeight())
        return false;
    return true;
}

ConvSolution
MultiWeightsBackward::GetSolution(const ExecutionContext& context,
                                  const miopen::prelu::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetdInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetdInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetdOuputDesc().GetType());

    /* Phase 1: Calc gradient for each elements. */
    {
        auto size         = problem.GetdInputDesc().GetElementSize();
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"VIEW_DIMS", VIEW_DIMS},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_SW_BWD}, {size}, "MIOpenPReLU.cpp", "PReLUMWBackward", build_params));
    }

    /* Phase 2: Reduce gradient. */
    {
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"OUTPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"REDUCE_SIZE", LOCAL_SIZE_MW_REDUCE_BWD},
        };
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_MW_REDUCE_BWD},
                            {problem.GetdWeightDesc().GetElementSize() * LOCAL_SIZE_MW_REDUCE_BWD},
                            "MIOpenReduceSum.cpp",
                            "Reduce1dSum",
                            build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::prelu::InvokeParams>();

            auto elapsed = 0.f;
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

            int kernelCnt = 0;

            /* Phase 1: Calc gradient for each elements. */
            {
                auto input_tv       = get_inner_expanded_tv<VIEW_DIMS>(deref(params.inputDesc));
                auto output_grad_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.doutputDesc));
                auto input_grad_tv  = get_inner_expanded_tv<VIEW_DIMS>(deref(params.dinputDesc));
                auto kernel         = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input,
                       params.weight,
                       params.doutput,
                       params.dinput,
                       params.workspace,
                       (ulong)deref(params.inputDesc).GetElementSize(),
                       input_tv,
                       output_grad_tv,
                       input_grad_tv);
            }

            /* Phase 2: Reduce gradient. */
            {
                size_t output_numel = deref(params.weightDesc).GetElementSize();
                size_t outer_size   = deref(params.inputDesc).GetLengths()[0];
                size_t inner_size =
                    deref(params.inputDesc).GetElementSize() / outer_size / output_numel;
                handle_.Run(kernels[kernelCnt++])(
                    params.workspace, params.dweight, output_numel, inner_size, outer_size);
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

std::size_t MultiWeightsBackward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::prelu::BackwardProblemDescription& problem) const
{
    auto size = problem.GetdInputDesc().GetElementSize();
    size *= sizeof(FLOAT_ACCUM);
    return size;
}

} // namespace prelu
} // namespace solver
} // namespace miopen
