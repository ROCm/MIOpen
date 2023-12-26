/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace cat {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const int32_t xCount_,
                       const TensorDescriptor* const* xDescs_,
                       const TensorDescriptor& yDesc_,
                       int32_t dim_)
        : xDescs(xDescs_), yDesc(yDesc_), xCount(xCount_), dim(dim_)
    {
    }

    const TensorDescriptor& GetXDesc(int i) const
    {
        if(i >= xCount)
        {
            MIOPEN_THROW(miopenStatusBadParm, "CatForward: Invalid tensor index.");
        }
        return *xDescs[i];
    }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    int32_t GetDim() const { return dim; }
    int32_t GetXCount() const { return xCount; }

    bool IsSameType() const
    {
        const auto dtype = yDesc.GetType();
        for(int i = 0; i < xCount; i++)
        {
            if(xDescs[i]->GetType() != dtype)
            {
                return false;
            }
        }
        return true;
    }

    bool IsRightLength() const
    {
        auto ydims = yDesc.GetLengths();
        ydims[dim] = 0;
        for(int i = 0; i < xCount; i++)
        {
            auto& xdims = xDescs[i]->GetLengths();

            if(ydims.size() != xdims.size())
            {
                return false;
            }

            for(int j = 0; j < ydims.size(); j++)
            {
                if((j != dim) && (ydims[j] != xdims[j]))
                {
                    return false;
                }
            }
            ydims[dim] += xdims[dim];
        }

        if(ydims[dim] != yDesc.GetLengths()[dim])
        {
            return false;
        }

        return true;
    }

    bool IsRightDim() const
    {
        if((dim < 0) || (dim > yDesc.GetLengths().size()))
        {
            return false;
        }
        return true;
    }

    bool IsAllPacked() const
    {
        for(int i = 0; i < xCount; i++)
        {
            if(!xDescs[i]->IsPacked())
            {
                return false;
            }
        }

        if(!yDesc.IsPacked())
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor* const* xDescs;
    const TensorDescriptor& yDesc;
    const int32_t xCount;
    const int32_t dim;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace cat
} // namespace miopen
