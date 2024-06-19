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

#define VIEW_DIMS 5

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace cumulative_reduction {

bool Forward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::cumulative_reduction::ForwardProblemDescription& /*problem*/) const
{
    return true;
}

ConvSolution
Forward::GetSolution(const ExecutionContext& /*context*/,
                     const miopen::cumulative_reduction::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOuputDesc().GetType());
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

    auto size       = problem.GetInputDesc().GetElementSize();
    auto inner_size = problem.GetInputDesc().GetLengths()[problem.GetDim()];
    auto outer_size = size / inner_size;
    result.construction_params.push_back(make_hip_kernel({1, LOCAL_SIZE},
                                                         {outer_size, inner_size},
                                                         "MIOpenCumulativeReduction.cpp",
                                                         "CumulativeReductionNaiveFwdNd",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::cumulative_reduction::InvokeParams>();
            auto kernel = handle_.Run(kernels.front());

            auto input_tv   = get_inner_expanded_tv<VIEW_DIMS>(deref(params.inputDesc));
            auto output_tv  = get_inner_expanded_tv<VIEW_DIMS>(deref(params.outputDesc));
            auto indices_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.indicesDesc));

            kernel(params.input,
                   params.output,
                   params.indices,
                   params.dim,
                   params.exclusive,
                   params.reverse,
                   input_tv,
                   output_tv,
                   indices_tv);
        };
    };

    return result;
}

} // namespace cumulative_reduction
} // namespace solver
} // namespace miopen
