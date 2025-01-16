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

#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/matrix_diag/invoke_params.hpp>
#include <miopen/matrix_diag/problem_description.hpp>
#include <miopen/matrix_diag/solvers.hpp>
#include <miopen/matrix_diag.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t MatrixSetDiagForward(Handle& handle,
                                    const TensorDescriptor& inputDesc,
                                    ConstData_t input,
                                    const TensorDescriptor& diagDesc,
                                    ConstData_t diag,
                                    const TensorDescriptor& outputDesc,
                                    Data_t output,
                                    const int64_t diagOffset0,
                                    const int64_t diagOffset1,
                                    const miopenMatrixDiagAlignMode_t align)
{
    const auto problem = matrix_diag::MatrixSetDiagForwardProblemDescription{
        inputDesc, diagDesc, outputDesc, diagOffset0, diagOffset1, align};

    const auto invoke_params = [&]() {
        auto tmp        = matrix_diag::MatrixSetDiagFwdInvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.inputDesc   = &inputDesc;
        tmp.diagDesc    = &diagDesc;
        tmp.outputDesc  = &outputDesc;
        tmp.input       = input;
        tmp.diag        = diag;
        tmp.output      = output;
        tmp.diagOffset0 = diagOffset0;
        tmp.diagOffset1 = diagOffset1;
        return tmp;
    }();

    const auto algo = AlgorithmName{"MatrixSetDiagForward"};
    const auto solvers =
        solver::SolverContainer<solver::matrix_diag::MatrixSetDiagForwardContiguous>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
