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

#include <miopen/softmaxcrossentropywithlogits.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/softmaxcrossentropywithlogits/invoke_params.hpp>
#include <miopen/softmaxcrossentropywithlogits/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace softmaxcrossentropywithlogits {

miopenStatus_t SoftmaxCrossEntropyWithLogitsForward(Handle& handle,
                                                    const TensorDescriptor& inputDesc,
                                                    ConstData_t input,
                                                    const TensorDescriptor& targetDesc,
                                                    ConstData_t target,
                                                    const TensorDescriptor& outputDesc,
                                                    Data_t output,
                                                    const TensorDescriptor& backpropDesc,
                                                    Data_t backprop)
{
    const auto problem = softmaxcrossentropywithlogits::FwdProblemDescription{
        inputDesc, targetDesc, outputDesc, backpropDesc};

    const auto invoke_params = [&]() {
        auto tmp         = softmaxcrossentropywithlogits::FwdInvokeParams{};
        tmp.inputDesc    = &inputDesc;
        tmp.targetDesc   = &targetDesc;
        tmp.outputDesc   = &outputDesc;
        tmp.backpropDesc = &backpropDesc;

        tmp.input    = input;
        tmp.target   = target;
        tmp.output   = output;
        tmp.backprop = backprop;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"SoftmaxCrossEntropyWithLogitsForward"};
    const auto solvers = solver::SolverContainer<
        solver::softmaxcrossentropywithlogits::SoftmaxCrossEntropyWithLogitsForwardContiguous>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxCrossEntropyWithLogitsBackward(Handle& handle,
                                                     const TensorDescriptor& outputGradDesc,
                                                     ConstData_t output_grad,
                                                     const TensorDescriptor& backpropDesc,
                                                     ConstData_t backprop,
                                                     const TensorDescriptor& inputDesc,
                                                     ConstData_t input,
                                                     const TensorDescriptor& inputGradDesc,
                                                     Data_t input_grad,
                                                     const TensorDescriptor& targetGradDesc,
                                                     Data_t target_grad)
{
    const auto problem = softmaxcrossentropywithlogits::BwdProblemDescription{
        outputGradDesc, backpropDesc, inputDesc, inputGradDesc, targetGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = softmaxcrossentropywithlogits::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.backpropDesc   = &backpropDesc;
        tmp.inputDesc      = &inputDesc;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.targetGradDesc = &targetGradDesc;

        tmp.output_grad = output_grad;
        tmp.backprop    = backprop;
        tmp.input       = input;
        tmp.input_grad  = input_grad;
        tmp.target_grad = target_grad;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"SoftmaxCrossEntropyWithLogitsBackward"};
    const auto solvers = solver::SolverContainer<
        solver::softmaxcrossentropywithlogits::SoftmaxCrossEntropyWithLogitsBackwardContiguous>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace softmaxcrossentropywithlogits

} // namespace miopen
