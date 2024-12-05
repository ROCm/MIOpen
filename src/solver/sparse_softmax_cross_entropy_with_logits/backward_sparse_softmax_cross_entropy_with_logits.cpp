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
#include <miopen/sparse_softmax_cross_entropy_with_logits.hpp>
#include <miopen/sparse_softmax_cross_entropy_with_logits/invoke_params.hpp>
#include <miopen/sparse_softmax_cross_entropy_with_logits/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_BWD 256

namespace miopen {

namespace solver {

namespace sparse_softmax_cross_entropy_with_logits {

bool SparseSoftmaxCrossEntropyWithLogitsBackward::IsApplicable(
    const ExecutionContext&,
    const miopen::sparse_softmax_cross_entropy_with_logits::BwdProblemDescription& problem) const
{
    if(!(problem.GetOutputGradDesc().GetType() == miopenHalf ||
         problem.GetOutputGradDesc().GetType() == miopenFloat ||
         problem.GetOutputGradDesc().GetType() == miopenBFloat16))
    {
        return false;
    }
    return true;
}

ConvSolution SparseSoftmaxCrossEntropyWithLogitsBackward::GetSolution(
    const ExecutionContext& context,
    const miopen::sparse_softmax_cross_entropy_with_logits::BwdProblemDescription& problem) const
{
    std::ignore       = context;
    auto output_dtype = miopen::GetDataType(problem.GetOutputGradDesc().GetType());
    auto dtype        = problem.GetOutputGradDesc().GetType();

    auto result       = ConvSolution{miopenStatusSuccess};
    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"LOCAL_SIZE", LOCAL_SIZE_BWD},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
    };

    if(!problem.IsAllContiguous())
    {
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_BWD},
                            {LOCAL_SIZE_BWD * problem.GetOutputGradDesc().GetLengths()[0]},
                            "MIOpenSparseSoftmaxCrossEntropyWithLogits.cpp",
                            "SparseSoftmaxCrossEntropyWithLogitsBackward",
                            build_params));

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<
                    miopen::sparse_softmax_cross_entropy_with_logits::BwdInvokeParams>();
                auto output_grad_tv = get_inner_expanded_tv<1>(deref(params.outputGradDesc));
                auto backprop_tv    = get_inner_expanded_tv<2>(deref(params.backpropDesc));
                auto input_grad_tv  = get_inner_expanded_tv<2>(deref(params.inputGradDesc));
                auto num_class      = deref(params.inputGradDesc).GetLengths()[1];

                kernel(params.output_grad,
                       params.backprop,
                       params.input_grad,
                       num_class,
                       output_grad_tv,
                       backprop_tv,
                       input_grad_tv);
            };
        };
    }
    else
    {
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_BWD},
                            {LOCAL_SIZE_BWD * problem.GetOutputGradDesc().GetLengths()[0]},
                            "MIOpenSparseSoftmaxCrossEntropyWithLogits.cpp",
                            "SparseSoftmaxCrossEntropyWithLogitsBackwardContiguous",
                            build_params));

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<
                    miopen::sparse_softmax_cross_entropy_with_logits::BwdInvokeParams>();
                auto num_class = deref(params.inputGradDesc).GetLengths()[1];

                kernel(params.output_grad, params.backprop, params.input_grad, num_class);
            };
        };
    }

    return result;
}

} // namespace sparse_softmax_cross_entropy_with_logits

} // namespace solver

} // namespace miopen
