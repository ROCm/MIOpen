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

// #include <cstdint>
#include "miopen/errors.hpp"
#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace any {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& outputDesc_,
                       int32_t dim_,
                       bool keepdim_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), dim(dim_), keepdim(keepdim_)
    {
        if(dim < -1 || dim >= static_cast<int32_t>(inputDesc.GetNumDims()))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Any: Dimension out of range.");
        }

        if(!IsRightLength())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Any: The input and output dim size don't match.");
        }

        // if(!IsAllPacked())
        // {
        //     MIOPEN_THROW(miopenStatusBadParm, "Any: The input or output tensor is not packed.");
        // }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    int32_t GetDim() const { return dim; }
    bool GetKeepDim() const { return keepdim; }

    bool IsRightLength() const
    {
        if(dim != -1)
        {
            if(keepdim)
            {
                if(inputDesc.GetNumDims() != outputDesc.GetNumDims())
                {
                    return false;
                }

                for(uint32_t i = 0; i < inputDesc.GetNumDims(); ++i)
                {
                    if(i == dim)
                    {
                        if(outputDesc.GetLengths()[i] != 1)
                        {
                            return false;
                        }
                    }
                    else
                    {
                        if(inputDesc.GetLengths()[i] != outputDesc.GetLengths()[i])
                        {
                            return false;
                        }
                    }
                }
            }
            else
            {
                if(inputDesc.GetNumDims() != outputDesc.GetNumDims() - 1)
                {
                    return false;
                }
            }
        }
        else
        {
            if(keepdim)
            {
                return false;
            }
            else
            {
                if(outputDesc.GetNumDims() != 1 || outputDesc.GetLengths()[0] != 1)
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inputDesc.IsPacked() && outputDesc.IsPacked()))
        {
            return false;
        }
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    int32_t dim;
    bool keepdim;
};

} // namespace any

} // namespace miopen