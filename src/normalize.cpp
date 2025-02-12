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

#include <miopen/normalize.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/normalize/invoke_params.hpp>
#include <miopen/normalize/solvers.hpp>

namespace miopen {

std::size_t GetNormalizeBackwardWorkspaceSize(Handle& handle,
                                              const TensorDescriptor& inputDesc,
                                              const TensorDescriptor& divisorDesc,
                                              const TensorDescriptor& outputGradDesc,
                                              const TensorDescriptor& inputGradDesc,
                                              const uint32_t dim)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = normalize::BackwardProblemDescription{
        inputDesc, divisorDesc, outputGradDesc, inputGradDesc, dim};

    const auto solvers = solver::SolverContainer<solver::normalize::NormalizeBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t NormalizeBackward(Handle& handle,
                                 Data_t workspace,
                                 const size_t workspaceSizeInBytes,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& divisorDesc,
                                 ConstData_t divisor,
                                 const TensorDescriptor& outputGradDesc,
                                 ConstData_t outputGrad,
                                 const TensorDescriptor& inputGradDesc,
                                 Data_t inputGrad,
                                 const float p,
                                 const float eps,
                                 const uint32_t dim)
{
    const auto problem = normalize::BackwardProblemDescription{
        inputDesc, divisorDesc, outputGradDesc, inputGradDesc, dim};

    const auto invoke_params = [&]() {
        auto tmp           = normalize::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputDesc      = &inputDesc;
        tmp.input          = input;
        tmp.divisorDesc    = &divisorDesc;
        tmp.divisor        = divisor;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.outputGrad     = outputGrad;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.inputGrad      = inputGrad;
        tmp.p              = p;
        tmp.eps            = eps;
        tmp.dim            = dim;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"NormalizeBackward"};
    const auto solvers = solver::SolverContainer<solver::normalize::NormalizeBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
