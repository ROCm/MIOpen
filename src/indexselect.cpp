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
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/indexselect/invoke_params.hpp>
#include <miopen/indexselect/solvers.hpp>
#include <miopen/indexselect.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace indexselect {

miopenStatus_t IndexSelectForward(Handle& handle,
                                  const TensorDescriptor& inputDesc,
                                  ConstData_t input,
                                  const TensorDescriptor& indicesDesc,
                                  ConstData_t indices,
                                  const TensorDescriptor& outputDesc,
                                  Data_t output,
                                  size_t dim)
{
    const auto problem =
        indexselect::FwdProblemDescription(inputDesc, indicesDesc, outputDesc, dim);

    const auto invoke_params = [&]() {
        auto tmp        = indexselect::FwdInvokeParams{};
        tmp.inputDesc   = &inputDesc;
        tmp.indicesDesc = &indicesDesc;
        tmp.outputDesc  = &outputDesc;
        tmp.input       = input;
        tmp.indices     = indices;
        tmp.output      = output;
        tmp.dim         = dim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"IndexSelectForward"};
    const auto solvers = solver::SolverContainer<solver::indexselect::IndexSelectForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t IndexSelectBackward(Handle& handle,
                                   const TensorDescriptor& inputGradDesc,
                                   Data_t inptuGrad,
                                   const TensorDescriptor& indicesDesc,
                                   ConstData_t indices,
                                   const TensorDescriptor& outputGradDesc,
                                   ConstData_t outputGrad,
                                   size_t dim)
{
    const auto problem =
        indexselect::BwdProblemDescription(inputGradDesc, indicesDesc, outputGradDesc, dim);

    const auto invoke_params = [&]() {
        auto tmp           = indexselect::BwdInvokeParams{};
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.indicesDesc    = &indicesDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.inputGrad      = inptuGrad;
        tmp.indices        = indices;
        tmp.outputGrad     = outputGrad;
        tmp.dim            = dim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"IndexSelectBackward"};
    const auto solvers = solver::SolverContainer<solver::indexselect::IndexSelectBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace indexselect

} // namespace miopen
