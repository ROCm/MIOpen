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

#include <miopen/find_solution.hpp>
#include <miopen/tensor.hpp>
#include <miopen/prelu.hpp>
#include <miopen/prelu/invoke_params.hpp>
#include <miopen/prelu/solvers.hpp>

namespace miopen {

size_t GetPReLUBackwardWorkspaceSize(Handle& handle,
                                     const TensorDescriptor& inputDesc,
                                     const TensorDescriptor& weightDesc)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        prelu::BackwardProblemDescription{inputDesc, weightDesc, inputDesc, inputDesc, weightDesc};

    const auto solvers = solver::SolverContainer<solver::prelu::SingleWeightBackward,
                                                 solver::prelu::MultiWeightsBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t PReLUBackward(Handle& handle,
                             Data_t workspace,
                             size_t workspaceSizeInBytes,
                             const TensorDescriptor& inputDesc,
                             ConstData_t input,
                             const TensorDescriptor& weightDesc,
                             ConstData_t weight,
                             const TensorDescriptor& doutputDesc,
                             ConstData_t doutput,
                             const TensorDescriptor& dinputDesc,
                             Data_t dinput,
                             const TensorDescriptor& dweightDesc,
                             Data_t dweight)
{
    const auto problem = prelu::BackwardProblemDescription{
        inputDesc, weightDesc, doutputDesc, inputDesc, dweightDesc};

    const auto invoke_params = [&]() {
        auto tmp           = prelu::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputDesc      = &inputDesc;
        tmp.weightDesc     = &weightDesc;
        tmp.doutputDesc    = &doutputDesc;
        tmp.dinputDesc     = &dinputDesc;
        tmp.dweightDesc    = &dweightDesc;
        tmp.input          = input;
        tmp.weight         = weight;
        tmp.doutput        = doutput;
        tmp.dinput         = dinput;
        tmp.dweight        = dweight;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PReLUBackward"};
    const auto solvers = solver::SolverContainer<solver::prelu::SingleWeightBackward,
                                                 solver::prelu::MultiWeightsBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
