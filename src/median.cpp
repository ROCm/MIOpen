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
#include <miopen/kernel_cache.hpp>
#include <miopen/median.hpp>
#include <miopen/median/invoke_params.hpp>
#include <miopen/median/solvers.hpp>
#include <miopen/median/problem_description.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace median {

miopenStatus_t MedianForward(Handle& handle,
                             const TensorDescriptor& inputDesc,
                             ConstData_t input,
                             const TensorDescriptor& outputDesc,
                             Data_t output,
                             const TensorDescriptor& indicesDesc,
                             Data_t indices,
                             int32_t dim)
{
    const auto problem = median::FwdProblemDescription{inputDesc, outputDesc, indicesDesc, dim};

    const auto invoke_params = [&]() {
        auto tmp        = median::FwdInvokeParams{};
        tmp.inputDesc   = &inputDesc;
        tmp.outputDesc  = &outputDesc;
        tmp.indicesDesc = &indicesDesc;
        tmp.input       = input;
        tmp.output      = output;
        tmp.indices     = indices;
        tmp.dim         = dim < 0 ? inputDesc.GetNumDims() + dim : dim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MedianForward"};
    const auto solvers = solver::SolverContainer<solver::median::MedianForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t MedianBackward(Handle& handle,
                              const TensorDescriptor& outputGradDesc,
                              ConstData_t outputGrad,
                              const TensorDescriptor& indicesDesc,
                              ConstData_t indices,
                              const TensorDescriptor& inputGradDesc,
                              Data_t inputGrad,
                              int32_t dim)
{
    const auto problem =
        median::BwdProblemDescription{outputGradDesc, indicesDesc, inputGradDesc, dim};

    const auto invoke_params = [&]() {
        auto tmp           = median::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.indicesDesc    = &indicesDesc;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.outputGrad     = outputGrad;
        tmp.indices        = indices;
        tmp.inputGrad      = inputGrad;
        tmp.dim            = dim < 0 ? inputGradDesc.GetNumDims() + dim : dim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MedianBackward"};
    const auto solvers = solver::SolverContainer<solver::median::MedianBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace median

} // namespace miopen
