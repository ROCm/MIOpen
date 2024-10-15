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

#include <miopen/kthvalue/invoke_params.hpp>
#include <miopen/kthvalue/problem_description.hpp>
#include <miopen/kthvalue/solvers.hpp>
#include <miopen/miopen.h>
#include <miopen/find_solution.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kthvalue.hpp>

namespace miopen {

miopenStatus_t KthvalueForward(Handle& handle,
                               const TensorDescriptor& inputDesc,
                               ConstData_t input,
                               const TensorDescriptor& outputDesc,
                               Data_t output,
                               const TensorDescriptor& indicesDesc,
                               size_t* indices,
                               size_t k,
                               int32_t dim,
                               bool keepDim)
{
    if(dim < 0)
    {
        dim += inputDesc.GetNumDims();
    }

    const auto problem =
        kthvalue::FwdProblemDescription{inputDesc, outputDesc, indicesDesc, dim, k, keepDim};

    const auto invoke_params = [&]() {
        auto tmp        = kthvalue::FwdInvokeParams{};
        tmp.inputDesc   = &inputDesc;
        tmp.outputDesc  = &outputDesc;
        tmp.indicesDesc = &indicesDesc;
        tmp.input       = input;
        tmp.indices     = indices;
        tmp.output      = output;
        tmp.k           = k;
        tmp.dim         = dim;
        tmp.keepDim     = keepDim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"KthvalueFwd"};
    const auto solvers = solver::SolverContainer<solver::kthvalue::KthvalueFwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
