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

miopenStatus_t MatrixDiagForward(Handle& handle,
                                 const TensorDescriptor& diagDesc,
                                 ConstData_t diag,
                                 const TensorDescriptor& padDesc,
                                 ConstData_t pad,
                                 const TensorDescriptor& outputDesc,
                                 Data_t output,
                                 const int64_t diagOffset0,
                                 const int64_t diagOffset1,
                                 const miopenMatrixDiagAlignMode_t align)
{
    const auto problem = matrix_diag::MatrixSetDiagForwardProblemDescription{padDesc,
                                                                             diagDesc,
                                                                             outputDesc,
                                                                             diagOffset0,
                                                                             diagOffset1,
                                                                             align,
                                                                             "MatrixDiagForward",
                                                                             "Padding",
                                                                             "Diagonal",
                                                                             "Output"};

    return MatrixSetDiagForward(
        handle, padDesc, pad, diagDesc, diag, outputDesc, output, diagOffset0, diagOffset1, align);
}

miopenStatus_t MatrixDiagBackward(Handle& handle,
                                  const TensorDescriptor& padDesc,
                                  ConstData_t pad,
                                  const TensorDescriptor& outputGradDesc,
                                  ConstData_t outputGrad,
                                  const TensorDescriptor& diagGradDesc,
                                  Data_t diagGrad,
                                  int64_t diagOffset0,
                                  int64_t diagOffset1,
                                  miopenMatrixDiagAlignMode_t align)
{
    const auto problem = matrix_diag::MatrixDiagPartForwardProblemDescription{outputGradDesc,
                                                                              padDesc,
                                                                              diagGradDesc,
                                                                              diagOffset0,
                                                                              diagOffset1,
                                                                              align,
                                                                              "MatrixDiagBackward",
                                                                              "Output gradient",
                                                                              "Padding",
                                                                              "Diagonal gradient"};

    return MatrixDiagPartForward(handle,
                                 outputGradDesc,
                                 outputGrad,
                                 padDesc,
                                 pad,
                                 diagGradDesc,
                                 diagGrad,
                                 diagOffset0,
                                 diagOffset1,
                                 align);
}

} // namespace miopen
