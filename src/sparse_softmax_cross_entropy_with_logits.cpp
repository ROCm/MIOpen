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
#include <miopen/sparse_softmax_cross_entropy_with_logits.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/sparse_softmax_cross_entropy_with_logits/invoke_params.hpp>
#include <miopen/sparse_softmax_cross_entropy_with_logits/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace sparse_softmax_cross_entropy_with_logits {

miopenStatus_t SparseSoftmaxCrossEntropyWithLogitsForward(Handle& handle,
                                                          const TensorDescriptor& inputDesc,
                                                          ConstData_t input,
                                                          const TensorDescriptor& targetDesc,
                                                          ConstData_t target,
                                                          const TensorDescriptor& outputDesc,
                                                          Data_t output,
                                                          const TensorDescriptor& backpropDesc,
                                                          Data_t backprop)
{
    const auto problem = sparse_softmax_cross_entropy_with_logits::FwdProblemDescription{
        inputDesc, targetDesc, outputDesc, backpropDesc};
    const auto invoke_params = [&]() {
        auto tmp         = sparse_softmax_cross_entropy_with_logits::FwdInvokeParams{};
        tmp.inputDesc    = &inputDesc;
        tmp.input        = input;
        tmp.targetDesc   = &targetDesc;
        tmp.target       = target;
        tmp.outputDesc   = &outputDesc;
        tmp.output       = output;
        tmp.backpropDesc = &backpropDesc;
        tmp.backprop     = backprop;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"SparseSoftmaxCrossEntropyWithLogitsForward"};
    const auto solvers = solver::SolverContainer<solver::sparse_softmax_cross_entropy_with_logits::
                                                     SparseSoftmaxCrossEntropyWithLogitsForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t SparseSoftmaxCrossEntropyWithLogitsBackward(Handle& handle,
                                                           const TensorDescriptor& outputGradDesc,
                                                           ConstData_t output_grad,
                                                           const TensorDescriptor& backpropDesc,
                                                           ConstData_t backprop,
                                                           const TensorDescriptor& inputGradDesc,
                                                           Data_t input_grad)
{
    const auto problem = sparse_softmax_cross_entropy_with_logits::BwdProblemDescription{
        outputGradDesc, backpropDesc, inputGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = sparse_softmax_cross_entropy_with_logits::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.output_grad    = output_grad;
        tmp.backpropDesc   = &backpropDesc;
        tmp.backprop       = backprop;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.input_grad     = input_grad;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"SparseSoftmaxCrossEntropyWithLogitsBackward"};
    const auto solvers = solver::SolverContainer<solver::sparse_softmax_cross_entropy_with_logits::
                                                     SparseSoftmaxCrossEntropyWithLogitsBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace sparse_softmax_cross_entropy_with_logits

} // namespace miopen
