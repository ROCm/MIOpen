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
#include <miopen/cartesianprod.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/cartesianprod/invoke_params.hpp>
#include <miopen/cartesianprod/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace cartesianprod {

size_t GetCartesianProdForwardWorkspaceSize(Handle& handle,
                                            const size_t inputCount,
                                            const TensorDescriptor* const* inputDescs,
                                            const TensorDescriptor& outputDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = cartesianprod::FwdProblemDescription{inputCount, inputDescs, outputDesc};

    const auto solvers = solver::SolverContainer<solver::cartesianprod::CartesianProdForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t CartesianProdForward(Handle& handle,
                                    Data_t workspace,
                                    const size_t workspaceSizeInBytes,
                                    const size_t inputCount,
                                    const TensorDescriptor* const* inputDescs,
                                    ConstData_t* inputs,
                                    const TensorDescriptor& outputDesc,
                                    Data_t output)
{
    const auto problem = cartesianprod::FwdProblemDescription{inputCount, inputDescs, outputDesc};
    const auto invoke_params = [&]() {
        auto tmp          = cartesianprod::FwdInvokeParams{};
        tmp.workspace     = workspace;
        tmp.workspaceSize = workspaceSizeInBytes;
        tmp.inputCount    = inputCount;
        tmp.inputDescs    = inputDescs;
        tmp.inputs        = inputs;
        tmp.outputDesc    = &outputDesc;
        tmp.output        = output;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"CartesianProdForward"};
    const auto solvers = solver::SolverContainer<solver::cartesianprod::CartesianProdForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t CartesianProdBackward(Handle& handle,
                                     const size_t inputCount,
                                     const TensorDescriptor& outputGradDesc,
                                     ConstData_t output_grad,
                                     const TensorDescriptor* const* inputGradDescs,
                                     Data_t* input_grads)
{
    const auto problem =
        cartesianprod::BwdProblemDescription{inputCount, outputGradDesc, inputGradDescs};

    const auto invoke_params = [&]() {
        auto tmp           = cartesianprod::BwdInvokeParams{};
        tmp.inputCount     = inputCount;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.output_grad    = output_grad;
        tmp.inputGradDescs = inputGradDescs;
        tmp.input_grads    = input_grads;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"CartesianProdBackward"};
    const auto solvers = solver::SolverContainer<solver::cartesianprod::CartesianProdBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace cartesianprod

} // namespace miopen
