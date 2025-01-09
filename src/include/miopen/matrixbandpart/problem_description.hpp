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
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace matrixbandpart {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& numLowerDesc_,
                       const TensorDescriptor& numUpperDesc_,
                       bool is_fwd_)
        : xDesc(xDesc_),
          yDesc(yDesc_),
          numLowerDesc(numLowerDesc_),
          numUpperDesc(numUpperDesc_),
          is_fwd(is_fwd_)
    {
        IsSameType();
        IsValidType();
        IsValidDims();
    }

    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetNumLowerDesc() const { return numLowerDesc; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType() || numLowerDesc.GetType() != numUpperDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm, "MatrixBandPart: Data types do not match.");
        return true;
    }

    bool IsValidType() const
    {
        if(numLowerDesc.GetType() != miopenInt32 && numLowerDesc.GetType() != miopenInt64)
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixBandPart: num_lower and num_upper tensor must be int32 "
                         "or int64.");
        return true;
    }

    bool IsValidDims() const
    {
        if(xDesc.GetLengths() != yDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixBandPart: Input and Output tensor must have the same dims.");
        if(numLowerDesc.GetLengths().size() != 1 || numUpperDesc.GetLengths().size() != 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixBandPart: num_lower and num_upper tensor must be 1D tensor.");
        if(xDesc.GetNumDims() < 2 || xDesc.GetNumDims() > 5)
            MIOPEN_THROW(miopenStatusBadParm,
                         "MatrixBandPart: Input tensor must have 2D, 3D, 4D or 5D tensor.");
        return true;
    }

    bool IsAllContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    TensorDescriptor numLowerDesc;
    TensorDescriptor numUpperDesc;
    bool is_fwd;
};

} // namespace matrixbandpart

} // namespace miopen
