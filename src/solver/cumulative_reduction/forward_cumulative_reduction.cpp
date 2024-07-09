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
#include <miopen/cumulative_reduction/invoke_params.hpp>
#include <miopen/cumulative_reduction/solvers.hpp>
#include <miopen/cumulative_reduction/utils.hpp>

#define FLOAT_ACCUM float
#define VIEW_DIMS 5

#define LOCAL_SIZE 128

namespace miopen {

namespace solver {

namespace cumulative_reduction {

bool IsImprovementOverROCm(const ExecutionContext& /*context*/,
                           const miopen::cumulative_reduction::ForwardProblemDescription& problem)
{
    // if(problem.GetDim() != problem.GetInputDesc().GetSize() - 1)
    //     return false;
    // if(problem.GetInputDesc().GetLengths()[problem.GetDim()] > LOCAL_SIZE)
    //     return false;
    return true;
}

bool Forward::IsApplicable(
    const ExecutionContext& context,
    const miopen::cumulative_reduction::ForwardProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(context, problem))
        return false;
    if(problem.GetInputDesc().GetSize() > VIEW_DIMS)
        return false;
    return true;
}

ConvSolution
Forward::GetSolution(const ExecutionContext& /*context*/,
                     const miopen::cumulative_reduction::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto cum_op       = problem.GetCumOp();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"VIEW_DIMS", VIEW_DIMS},
        {"OP_TYPE", cum_op},
        {"REDUCE_SIZE", LOCAL_SIZE},
    };

    if(problem.GetIndicesDesc().GetLengths()[problem.GetDim()] > LOCAL_SIZE)
    {
        auto size             = problem.GetInputDesc().GetElementSize();
        auto inner_size       = problem.GetInputDesc().GetLengths()[problem.GetDim()];
        auto local_inner_size = AlignUp(inner_size, LOCAL_SIZE) / LOCAL_SIZE;
        auto local_size       = size / inner_size * local_inner_size;
        result.construction_params.push_back(make_hip_kernel({1, LOCAL_SIZE},
                                                             {local_size, LOCAL_SIZE},
                                                             "MIOpenCumulativeReduction.cpp",
                                                             "LocalCumulativeReduction",
                                                             build_params));
    }

    if(problem.GetIndicesDesc().GetLengths()[problem.GetDim()] > LOCAL_SIZE)
    {
        auto size       = problem.GetInputDesc().GetElementSize();
        auto inner_size = problem.GetInputDesc().GetLengths()[problem.GetDim()];
        auto outer_size = size / inner_size;
        result.construction_params.push_back(make_hip_kernel({1, LOCAL_SIZE},
                                                             {outer_size, LOCAL_SIZE},
                                                             "MIOpenCumulativeReduction.cpp",
                                                             "CumulativeReductionNaiveForward",
                                                             build_params));
    }

    {
        auto size       = problem.GetInputDesc().GetElementSize();
        auto inner_size = problem.GetInputDesc().GetLengths()[problem.GetDim()];
        auto outer_size = size / inner_size;
        result.construction_params.push_back(make_hip_kernel({1, LOCAL_SIZE},
                                                             {outer_size, inner_size},
                                                             "MIOpenCumulativeReduction.cpp",
                                                             "CumulativeReductionForward",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::cumulative_reduction::InvokeParams>();

            auto elapsed = 0.f;
            HipEventPtr start;
            HipEventPtr stop;

            bool reset_profiling_state = false;
            if(handle_.IsProfilingEnabled())
            {
                reset_profiling_state = true;
                handle_.EnableProfiling(false);
                hipStreamSynchronize(handle_.GetStream());
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            int kernelCnt = 0;

            const auto ndims            = deref(params.inputDesc).GetSize();
            const unsigned int true_dim = ((params.dim % ndims) + ndims) % ndims;

            Data_t local_output  = nullptr;
            Data_t local_indices = nullptr;
            if(deref(params.inputDesc).GetLengths()[true_dim] > LOCAL_SIZE)
            {
                local_output = params.workspace;
                if(params.indices != nullptr)
                {
                    local_indices = static_cast<Data_t>(
                        static_cast<char*>(local_output) +
                        deref(params.indicesDesc).GetElementSize() /
                            deref(params.indicesDesc).GetLengths()[true_dim] *
                            AlignUp(deref(params.indicesDesc).GetLengths()[true_dim], LOCAL_SIZE) /
                            LOCAL_SIZE * sizeof(FLOAT_ACCUM));
                }
            }

            if(deref(params.inputDesc).GetLengths()[true_dim] > LOCAL_SIZE)
            {
                auto input_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.inputDesc));
                auto kernel   = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input,
                       local_output,
                       local_indices,
                       true_dim,
                       params.exclusive,
                       params.reverse,
                       input_tv);
            }

            if(deref(params.inputDesc).GetLengths()[true_dim] > LOCAL_SIZE)
            {
                auto reduce_size =
                    AlignUp(deref(params.inputDesc).GetLengths()[true_dim], LOCAL_SIZE) /
                    LOCAL_SIZE;
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(
                    local_output, local_output, local_indices, (size_t)reduce_size, params.reverse);
            }

            {
                auto input_tv   = get_inner_expanded_tv<VIEW_DIMS>(deref(params.inputDesc));
                auto output_tv  = get_inner_expanded_tv<VIEW_DIMS>(deref(params.outputDesc));
                auto indices_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.indicesDesc));
                auto kernel     = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input,
                       params.output,
                       params.indices,
                       local_output,
                       local_indices,
                       true_dim,
                       params.exclusive,
                       params.reverse,
                       input_tv,
                       output_tv,
                       indices_tv);
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

std::size_t Forward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::cumulative_reduction::ForwardProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetLengths()[problem.GetDim()] > LOCAL_SIZE)
    {
        size_t size = 0;
        size += problem.GetInputDesc().GetElementSize() * sizeof(FLOAT_ACCUM);
        size += problem.GetIndicesDesc().GetElementSize() * sizeof(int);
        return size;
    }
    else
        return 0;
}

} // namespace cumulative_reduction
} // namespace solver
} // namespace miopen
