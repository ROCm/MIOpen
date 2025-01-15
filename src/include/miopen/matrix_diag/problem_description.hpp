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
#pragma once

#include <miopen/miopen.h>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace matrix_diag {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& diagDesc_,
                              const TensorDescriptor& outputDesc_,
                              const int64_t diagOffset0_,
                              const int64_t diagOffset1_,
                              const TensorDescriptor& padDesc_,
                              const miopenMatrixDiagAlignMode_t align_)
        : diagDesc(diagDesc_),
          outputDesc(outputDesc_),
          diagOffset0(diagOffset0_),
          diagOffset1(diagOffset1_),
          padDesc(padDesc_),
          align(align_)
    {
        IsValidType();
        IsValidSize();
    }

    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    miopenMatrixDiagAlignMode_t GetAlign() const { return align; }

    bool IsValidType() const
    {
        if(diagDesc.GetType() != outputDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: Diagonal and Output datatype do not match.");
        if(diagDesc.GetType() != padDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: Diagonal and Padding datatype do not match.");
        return true;
    }

    bool IsValidSize() const
    {
        if(outputDesc.GetNumDims() < 2)
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: Output tensor must have at least 2 dimensions");
        if(-diagOffset0 >=
           static_cast<int64_t>(outputDesc.GetLengths()[outputDesc.GetNumDims() - 2]))
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MatrixDiag: diagOffset0 must be less than the second last Output dimension");
        if(diagOffset1 >=
           static_cast<int64_t>(outputDesc.GetLengths()[outputDesc.GetNumDims() - 1]))
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: diagOffset1 must be less than the last Output dimension");
        if(diagOffset0 == diagOffset1)
        {
            if(diagDesc.GetNumDims() < 1)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor must "
                             "have at least 1 dimension");
            if(diagDesc.GetNumDims() + 1 != outputDesc.GetNumDims())
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Output tensor must have "
                             "1 more dimension than Diagonal tensor");
            if(std::vector<size_t>(diagDesc.GetLengths().begin(),
                                   diagDesc.GetLengths().end() - 1) !=
               std::vector<size_t>(outputDesc.GetLengths().begin(),
                                   outputDesc.GetLengths().end() - 2))
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor has "
                             "shape [I, J, ..., L, M, N] then Output tensor must have shape [I, J, "
                             "..., L, M, num_rows, num_cols]");
            int64_t num_rows = outputDesc.GetLengths()[outputDesc.GetNumDims() - 2];
            int64_t num_cols = outputDesc.GetLengths()[outputDesc.GetNumDims() - 1];
            auto diagLength  = std::min(num_rows, num_cols) -
                              std::max({std::min(num_cols - num_rows, 0L) - diagOffset0,
                                        diagOffset0 - std::max(num_cols - num_rows, 0L),
                                        0L});
            if(diagDesc.GetLengths().back() != diagLength)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor last "
                             "dimension must match with Output tensor diagonal length");
        }
        else
        {
            if(diagOffset0 > diagOffset1)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: diagOffset0 must be less than or equal to diagOffset1");
            if(diagDesc.GetNumDims() < 2)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor must "
                             "have at least 2 dimensions");
            if(diagDesc.GetNumDims() != outputDesc.GetNumDims())
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 != diagOffset1, Output tensor must has "
                             "the same number of dimension with Diagonal tensor");
            if(std::vector<size_t>(diagDesc.GetLengths().begin(),
                                   diagDesc.GetLengths().end() - 2) !=
               std::vector<size_t>(outputDesc.GetLengths().begin(),
                                   outputDesc.GetLengths().end() - 2))
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 != diagOffset1, Diagonal tensor has "
                             "shape [I, J, ..., L, M, N] then Output tensor must have shape [I, J, "
                             "..., L, num_rows, num_cols]");
            if(diagDesc.GetLengths()[diagDesc.GetNumDims() - 2] != diagOffset1 - diagOffset0 + 1)
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    "MatrixDiag: When diagOffset0 != diagOffset1, Diagonal tensor second last "
                    "dimension must equal to diagOffset1 - diagOffset0");
            int64_t num_rows = outputDesc.GetLengths()[outputDesc.GetNumDims() - 2];
            int64_t num_cols = outputDesc.GetLengths()[outputDesc.GetNumDims() - 1];
            auto diagLength0 = std::min(num_rows, num_cols) -
                               std::max(std::min(num_cols - num_rows, 0L) - diagOffset0,
                                        diagOffset0 - std::max(num_cols - num_rows, 0L));
            auto diagLength1 = std::min(num_rows, num_cols) -
                               std::max(std::min(num_cols - num_rows, 0L) - diagOffset1,
                                        diagOffset1 - std::max(num_cols - num_rows, 0L));
            auto diagLength = 0L;
            if(std::abs(diagOffset0 + diagOffset1) < std::abs(diagOffset0) + std::abs(diagOffset1))
                diagLength = std::min(num_rows, num_cols);
            if(diagDesc.GetLengths().back() != std::max({diagLength, diagLength0, diagLength1}))
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 != diagOffset1, Diagonal tensor last "
                             "dimension must match with Output tensor diagonal length");
        }
        if(padDesc.GetElementSize() > 1 && padDesc.GetLengths() != outputDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: When there are more than 1 padding value, Padding and Ouput "
                         "tensor size must be same.");
        return true;
    }

    bool IsAllContiguous() const
    {
        if(!diagDesc.IsContiguous())
            return false;
        if(!outputDesc.IsContiguous())
            return false;
        if(!padDesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor diagDesc;
    const TensorDescriptor outputDesc;
    const int64_t diagOffset0;
    const int64_t diagOffset1;
    const TensorDescriptor padDesc;
    const miopenMatrixDiagAlignMode_t align;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace matrix_diag

namespace matrix_set_diag {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& diagDesc_,
                              const TensorDescriptor& outputDesc_,
                              const int64_t diagOffset0_,
                              const int64_t diagOffset1_,
                              const miopenMatrixDiagAlignMode_t align_)
        : inputDesc(inputDesc_),
          diagDesc(diagDesc_),
          outputDesc(outputDesc_),
          diagOffset0(diagOffset0_),
          diagOffset1(diagOffset1_),
          align(align_)
    {
        IsValidType();
        IsValidSize();
    }

    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    miopenMatrixDiagAlignMode_t GetAlign() const { return align; }

    bool IsValidType() const
    {
        if(diagDesc.GetType() != inputDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: Diagonal and Input datatype do not match.");
        if(diagDesc.GetType() != outputDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: Diagonal and Output datatype do not match.");
        return true;
    }

    bool IsValidSize() const
    {
        if(outputDesc.GetNumDims() < 2)
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: Output tensor must have at least 2 dimensions");
        if(-diagOffset0 >=
           static_cast<int64_t>(outputDesc.GetLengths()[outputDesc.GetNumDims() - 2]))
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MatrixDiag: diagOffset0 must be less than the second last Output dimension");
        if(diagOffset1 >=
           static_cast<int64_t>(outputDesc.GetLengths()[outputDesc.GetNumDims() - 1]))
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixDiag: diagOffset1 must be less than the last Output dimension");
        if(diagOffset0 == diagOffset1)
        {
            if(diagDesc.GetNumDims() < 1)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor must "
                             "have at least 1 dimension");
            if(diagDesc.GetNumDims() + 1 != outputDesc.GetNumDims())
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Output tensor must have "
                             "1 more dimension than Diagonal tensor");
            if(std::vector<size_t>(diagDesc.GetLengths().begin(),
                                   diagDesc.GetLengths().end() - 1) !=
               std::vector<size_t>(outputDesc.GetLengths().begin(),
                                   outputDesc.GetLengths().end() - 2))
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor has "
                             "shape [I, J, ..., L, M, N] then Output tensor must have shape [I, J, "
                             "..., L, M, num_rows, num_cols]");
            int64_t num_rows = outputDesc.GetLengths()[outputDesc.GetNumDims() - 2];
            int64_t num_cols = outputDesc.GetLengths()[outputDesc.GetNumDims() - 1];
            auto diagLength  = std::min(num_rows, num_cols) -
                              std::max({std::min(num_cols - num_rows, 0L) - diagOffset0,
                                        diagOffset0 - std::max(num_cols - num_rows, 0L),
                                        0L});
            if(diagDesc.GetLengths().back() != diagLength)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor last "
                             "dimension must match with Output tensor diagonal length");
        }
        else
        {
            if(diagOffset0 > diagOffset1)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: diagOffset0 must be less than or equal to diagOffset1");
            if(diagDesc.GetNumDims() < 2)
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 == diagOffset1, Diagonal tensor must "
                             "have at least 2 dimensions");
            if(diagDesc.GetNumDims() != outputDesc.GetNumDims())
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 != diagOffset1, Output tensor must has "
                             "the same number of dimension with Diagonal tensor");
            if(std::vector<size_t>(diagDesc.GetLengths().begin(),
                                   diagDesc.GetLengths().end() - 2) !=
               std::vector<size_t>(outputDesc.GetLengths().begin(),
                                   outputDesc.GetLengths().end() - 2))
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 != diagOffset1, Diagonal tensor has "
                             "shape [I, J, ..., L, M, N] then Output tensor must have shape [I, J, "
                             "..., L, num_rows, num_cols]");
            if(diagDesc.GetLengths()[diagDesc.GetNumDims() - 2] != diagOffset1 - diagOffset0 + 1)
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    "MatrixDiag: When diagOffset0 != diagOffset1, Diagonal tensor second last "
                    "dimension must equal to diagOffset1 - diagOffset0");
            int64_t num_rows = outputDesc.GetLengths()[outputDesc.GetNumDims() - 2];
            int64_t num_cols = outputDesc.GetLengths()[outputDesc.GetNumDims() - 1];
            auto diagLength0 = std::min(num_rows, num_cols) -
                               std::max(std::min(num_cols - num_rows, 0L) - diagOffset0,
                                        diagOffset0 - std::max(num_cols - num_rows, 0L));
            auto diagLength1 = std::min(num_rows, num_cols) -
                               std::max(std::min(num_cols - num_rows, 0L) - diagOffset1,
                                        diagOffset1 - std::max(num_cols - num_rows, 0L));
            auto diagLength = 0L;
            if(std::abs(diagOffset0 + diagOffset1) < std::abs(diagOffset0) + std::abs(diagOffset1))
                diagLength = std::min(num_rows, num_cols);
            if(diagDesc.GetLengths().back() != std::max({diagLength, diagLength0, diagLength1}))
                MIOPEN_THROW(miopenStatusBadParm,
                             "MatrixDiag: When diagOffset0 != diagOffset1, Diagonal tensor last "
                             "dimension must match with Output tensor diagonal length");
        }
        if(inputDesc.GetElementSize() > 1 && inputDesc.GetLengths() != outputDesc.GetLengths())
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MatrixDiag: When there are more than 1 element in Input tensor, Input and Output "
                "tensor size must be same.");
        return true;
    }

    bool IsAllContiguous() const
    {
        if(!inputDesc.IsContiguous())
            return false;
        if(!diagDesc.IsContiguous())
            return false;
        if(!outputDesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor inputDesc;
    const TensorDescriptor diagDesc;
    const TensorDescriptor outputDesc;
    const int64_t diagOffset0;
    const int64_t diagOffset1;
    const miopenMatrixDiagAlignMode_t align;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace matrix_set_diag

} // namespace miopen
