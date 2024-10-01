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

#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace kthvalue {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& indicesDesc_,
                          int32_t dim_,
                          size_t k_,
                          bool keepDim_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          indicesDesc(indicesDesc_),
          dim(dim_),
          k(k_),
          keepDim(keepDim_)
    {
        if(k < 1 || k > inputDesc.GetLengths()[dim])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Kthvalue: selected number k out of range for dimension");
        }
        if(dim < 0 || dim >= inputDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Kthvalue: dim doesn't not exist");
        }
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Input, output tensor types do not match.");
        }
        if(!IsRightLength())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Reduce: Input and output tensor dimension lengths do not match.");
        }
        if(outputDesc.GetLengths() != indicesDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Reduce: Output and indices tensor dimension lengths do not match.");
        }
    }

    bool IsRightLength() const
    {
        if(inputDesc.GetLengths().size() == 1)
            return true;

        if(keepDim && inputDesc.GetNumDims() != outputDesc.GetNumDims())
        {
            return false;
        }
        if(!keepDim && inputDesc.GetNumDims() != outputDesc.GetNumDims() + 1)
        {
            return false;
        }

        int32_t posOut = 0;
        for(int32_t i = 0; i < inputDesc.GetLengths().size(); i++)
        {
            if(i == dim)
            {
                if(!keepDim)
                    continue;
                if(outputDesc.GetLengths()[posOut] != 1)
                    return false;
            }
            else if(inputDesc.GetLengths()[i] != outputDesc.GetLengths()[posOut])
            {
                return false;
            }

            posOut++;
        }
        return true;
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    int32_t GetDim() const { return dim; }
    size_t GetK() const { return k; }
    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor indicesDesc;
    int32_t dim;
    size_t k;
    bool keepDim;
};

} // namespace kthvalue

} // namespace miopen
