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
#include <miopen/pdist.hpp>
// #include <miopen/pdist/invoke_params.hpp>
// #include <miopen/pdist/solvers.hpp>
// #include <miopen/pdist/problem_description.hpp>

#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>
#include <miopen/names.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

std::size_t GetPdistBackwardWorkspaceSize(Handle& handle, const TensorDescriptor& inputDesc)
{
    // auto ctx = ExecutionContext(&handle);
    auto ctx = ExecutionContext{&handle};

    const auto problem = pdist::BackwardProblemDescription(inputDesc);
    const auto solvers = solver::SolverContainer<solver::pdist::PdistBackward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t PdistBackward(Handle& handle,
                             Data_t workspace,
                             size_t workspaceSizeInBytes,
                             const TensorDescriptor& inputDesc,
                             ConstData_t input,
                             const TensorDescriptor& outputDesc,
                             ConstData_t output,
                             const TensorDescriptor& douputDesc,
                             ConstData_t douput,
                             const TensorDescriptor& dinputDesc,
                             Data_t dinput,
                             const double p)
{
    const auto problem =
        pdist::BackwardProblemDescription(inputDesc, outputDesc, douputDesc, dinputDesc, p);

    const auto invoke_params = [&]() {
        auto tmp       = pdist::BackwardInvokeParams{};
        tmp.inputDesc  = inputDesc;
        tmp.input      = input;
        tmp.outputDesc = outputDesc;
        tmp.output     = output;
        tmp.douputDesc = douputDesc;
        tmp.doutput    = douput;
        tmp.dinputDesc = dinputDesc;
        tmp.dinput     = dinput;
        tmp.p          = p;

        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;

        return tmp;
    }();

    const auto algo    = AlgorithmName("PdistBackward");
    const auto solvers = solver::SolverContainer<solver::pdist::PdistBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
