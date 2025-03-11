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

#include <miopen/hinge_embedding_loss.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/hingeembeddingloss/invoke_params.hpp>
#include <miopen/hingeembeddingloss/solvers.hpp>

namespace miopen {

size_t GetHingeEmbeddingLossForwardWorkspaceSize(Handle& handle,
                                                 const TensorDescriptor& inputDesc,
                                                 const TensorDescriptor& targetDesc,
                                                 const TensorDescriptor& outputDesc,
                                                 const miopenLossReductionMode_t reduction)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        hingeembeddingloss::ForwardProblemDescription{inputDesc, targetDesc, outputDesc, reduction};

    const auto solvers =
        solver::SolverContainer<solver::hingeembeddingloss::HingeEmbeddingLossForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t HingeEmbeddingLossForward(Handle& handle,
                                         Data_t workspace,
                                         const size_t workspaceSizeInBytes,
                                         const TensorDescriptor& inputDesc,
                                         ConstData_t input,
                                         const TensorDescriptor& targetDesc,
                                         ConstData_t target,
                                         const TensorDescriptor& outputDesc,
                                         Data_t output,
                                         const float margin,
                                         const miopenLossReductionMode_t reduction)
{
    const auto problem =
        hingeembeddingloss::ForwardProblemDescription{inputDesc, targetDesc, outputDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp           = hingeembeddingloss::InvokeParams{};
        tmp.inputDesc      = &inputDesc;
        tmp.targetDesc     = &targetDesc;
        tmp.outputDesc     = &outputDesc;
        tmp.input          = input;
        tmp.target         = target;
        tmp.output         = output;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.margin         = margin;
        return tmp;
    }();

    const auto algo = AlgorithmName{"HingeEmbeddingLossForward"};
    const auto solvers =
        solver::SolverContainer<solver::hingeembeddingloss::HingeEmbeddingLossForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t HingeEmbeddingLossBackward(Handle& handle,
                                          const TensorDescriptor& inputDesc,
                                          ConstData_t input,
                                          const TensorDescriptor& targetDesc,
                                          ConstData_t target,
                                          const TensorDescriptor& doutputDesc,
                                          ConstData_t doutput,
                                          const TensorDescriptor& dinputDesc,
                                          Data_t dinput,
                                          const float margin,
                                          const miopenLossReductionMode_t reduction)
{
    const auto problem = hingeembeddingloss::BackwardProblemDescription{
        inputDesc, targetDesc, doutputDesc, dinputDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp        = hingeembeddingloss::InvokeParams{};
        tmp.inputDesc   = &inputDesc;
        tmp.targetDesc  = &targetDesc;
        tmp.doutputDesc = &doutputDesc;
        tmp.dinputDesc  = &dinputDesc;
        tmp.input       = input;
        tmp.target      = target;
        tmp.doutput     = doutput;
        tmp.dinput      = dinput;
        tmp.margin      = margin;
        return tmp;
    }();

    const auto algo = AlgorithmName{"HingeEmbeddingLossBackward"};
    const auto solvers =
        solver::SolverContainer<solver::hingeembeddingloss::HingeEmbeddingLossBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
