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
#include <miopen/cumulative_reduction.hpp>
#include <miopen/cumulative_reduction/invoke_params.hpp>
#include <miopen/cumulative_reduction/solvers.hpp>

namespace miopen {

size_t GetCumulativeReductionForwardWorkspaceSize(Handle& handle,
                                                  const TensorDescriptor& inputDesc,
                                                  const TensorDescriptor& indicesDesc,
                                                  const int dim)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = cumulative_reduction::ForwardProblemDescription{
        inputDesc, inputDesc, indicesDesc, dim, MIOPEN_CUM_MAX};

    const auto solvers =
        solver::SolverContainer<solver::cumulative_reduction::ForwardContiguousLastDim>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t CumulativeReductionForward(Handle& handle,
                                          Data_t workspace,
                                          size_t workspaceSizeInBytes,
                                          const TensorDescriptor& inputDesc,
                                          ConstData_t input,
                                          const TensorDescriptor& outputDesc,
                                          Data_t output,
                                          const TensorDescriptor& indicesDesc,
                                          Data_t indices,
                                          const int dim,
                                          const bool exclusive,
                                          const bool reverse,
                                          const miopenCumOp_t cumOp)
{
    const auto problem = cumulative_reduction::ForwardProblemDescription{
        inputDesc, outputDesc, indicesDesc, dim, cumOp};

    const auto invoke_params = [&]() {
        auto tmp        = cumulative_reduction::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.inputDesc   = &inputDesc;
        tmp.outputDesc  = &outputDesc;
        tmp.indicesDesc = &indicesDesc;
        tmp.input       = input;
        tmp.output      = output;
        tmp.indices     = indices;

        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;

        tmp.dim       = dim;
        tmp.exclusive = exclusive;
        tmp.reverse   = reverse;

        return tmp;
    }();

    const auto algo = AlgorithmName{"CumulativeReductionForward"};
    const auto solvers =
        solver::SolverContainer<solver::cumulative_reduction::ForwardContiguousLastDim>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
