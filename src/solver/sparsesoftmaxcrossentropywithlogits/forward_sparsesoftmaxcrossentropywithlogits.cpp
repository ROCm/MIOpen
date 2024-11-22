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

#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/sparsesoftmaxcrossentropywithlogits/solvers.hpp>

#include <miopen/sparsesoftmaxcrossentropywithlogits/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/sparsesoftmaxcrossentropywithlogits.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/par_for.hpp>

#define LOCAL_SIZE_FWD 128

namespace miopen {

namespace solver {

namespace sparsesoftmaxcrossentropywithlogits {

bool SparseSoftmaxCrossEntropyWithLogitsForward::IsApplicable(
    const ExecutionContext&,
    const miopen::sparsesoftmaxcrossentropywithlogits::FwdProblemDescription& problem) const
{
    // if(!(problem.GetOutputDesc().GetType() == miopenHalf ||
    //      problem.GetOutputDesc().GetType() == miopenFloat ||
    //      problem.GetOutputDesc().GetType() == miopenBFloat16))
    // {
    //     return false;
    // }
    return true;
}

ConvSolution SparseSoftmaxCrossEntropyWithLogitsForward::GetSolution(
    const ExecutionContext& context,
    const miopen::sparsesoftmaxcrossentropywithlogits::FwdProblemDescription& problem) const
{
    std::ignore       = context;
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto target_dtype = miopen::GetDataType(problem.GetTargetDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    auto dtype_lowest = std::numeric_limits<float>::lowest();

    auto result       = ConvSolution{miopenStatusSuccess};
    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE_FWD},
        {"T_TYPE", target_dtype},
        {"DTYPE_LOWEST", dtype_lowest},
    };

    if(!problem.IsAllContiguous())
    {
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_FWD},
                            {LOCAL_SIZE_FWD * problem.GetOutputDesc().GetLengths()[0]},
                            "MIOpenSparseSoftmaxCrossEntropyWithLogits.cpp",
                            "SparseSoftmaxCrossEntropyWithLogitsForward",
                            build_params));

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params =
                    raw_params
                        .CastTo<miopen::sparsesoftmaxcrossentropywithlogits::FwdInvokeParams>();
                auto input_tv    = get_inner_expanded_tv<2>(deref(params.inputDesc));
                auto target_tv   = get_inner_expanded_tv<1>(deref(params.targetDesc));
                auto output_tv   = get_inner_expanded_tv<1>(deref(params.outputDesc));
                auto backprop_tv = get_inner_expanded_tv<2>(deref(params.backpropDesc));
                auto num_class   = deref(params.inputDesc).GetLengths()[1];

                kernel(params.input,
                       params.target,
                       params.output,
                       params.backprop,
                       num_class,
                       input_tv,
                       target_tv,
                       output_tv,
                       backprop_tv);
            };
        };
    }
    else
    {
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_FWD},
                            {LOCAL_SIZE_FWD * problem.GetOutputDesc().GetLengths()[0]},
                            "MIOpenSparseSoftmaxCrossEntropyWithLogits.cpp",
                            "SparseSoftmaxCrossEntropyWithLogitsForwardContiguous",
                            build_params));

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params =
                    raw_params
                        .CastTo<miopen::sparsesoftmaxcrossentropywithlogits::FwdInvokeParams>();
                auto num_class = deref(params.inputDesc).GetLengths()[1];

                kernel(params.input, params.target, params.output, params.backprop, num_class);
            };
        };
    }

    return result;
};

} // namespace sparsesoftmaxcrossentropywithlogits

} // namespace solver

} // namespace miopen
