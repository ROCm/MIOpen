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

#define warpSizeCTX (context.GetStream().GetWavefrontWidth())
#define LOCAL_SIZE_MAX 256
#define LOCAL_SIZE_MIN warpSizeCTX

namespace miopen {

namespace solver {

namespace cumulative_reduction {

bool IsImprovementOverROCm(const ExecutionContext& /*context*/,
                           const miopen::cumulative_reduction::ForwardProblemDescription& problem)
{
    if(problem.GetInputDesc().GetLengths()[problem.GetDim()] > LOCAL_SIZE_MAX)
        return false;
    return true;
}

bool ForwardContiguousLastDim::IsApplicable(
    const ExecutionContext& context,
    const miopen::cumulative_reduction::ForwardProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(context, problem))
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsAllDimStride1())
        return false;
    return true;
}

ConvSolution ForwardContiguousLastDim::GetSolution(
    const ExecutionContext& context,
    const miopen::cumulative_reduction::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto cum_op       = problem.GetCumOp();

    auto size       = problem.GetInputDesc().GetElementSize();
    auto inner_size = problem.GetInputDesc().GetLengths()[problem.GetDim()];
    auto outer_size = size / inner_size;

    auto local_size = LOCAL_SIZE_MIN;
    while(local_size < inner_size)
        local_size *= 2;

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"OP_TYPE", cum_op},
        {"REDUCE_SIZE", local_size},
    };

    {
        result.construction_params.push_back(
            make_hip_kernel({1, local_size},
                            {outer_size, inner_size},
                            "MIOpenCumulativeReduction.cpp",
                            "CumulativeReductionForwardContiguousLastDim",
                            build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::cumulative_reduction::InvokeParams>();

            const int ndims             = deref(params.inputDesc).GetNumDims();
            const unsigned int true_dim = ((params.dim % ndims) + ndims) % ndims;
            auto kernel                 = handle_.Run(kernels[0]);
            kernel(params.input,
                   params.output,
                   params.indices,
                   deref(params.indicesDesc).GetLengths()[true_dim],
                   params.exclusive,
                   params.reverse);
        };
    };

    return result;
}

} // namespace cumulative_reduction
} // namespace solver
} // namespace miopen
