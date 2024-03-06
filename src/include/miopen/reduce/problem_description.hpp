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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace reduce {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(miopenSumNanPropagation_t nanPropagation_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& reduceDesc_,
                       int32_t dim_)
        : nanPropagation(nanPropagation_), xDesc(xDesc_), reduceDesc(reduceDesc_), dim(dim_)
    {
    }

    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& reduceDesc_,
                       int32_t dim_,
                       miopenReduceExtremeOp_t reduceExtremeOp_)
        : xDesc(xDesc_), reduceDesc(reduceDesc_), dim(dim_), reduceExtremeOp(reduceExtremeOp_)
    {
    }

    miopenSumNanPropagation_t GetNanPropagation_() const { return nanPropagation; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetReduceDesc() const { return reduceDesc; }
    int32_t GetDim() const { return dim; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != reduceDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightLength() const
    {
        if(xDesc.GetLengths().size() == 1)
            return true;

        int32_t posy = 0;
        for(int32_t i = 0; i < xDesc.GetLengths().size(); i++)
        {
            if(i == dim)
                continue;

            if(xDesc.GetLengths()[i] != reduceDesc.GetLengths()[posy])
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor dimension lengths do not match.");
#else
                return false;
#endif
            }

            posy++;
        }
        return true;
    }

    bool IsRightDim() const
    {
        if((dim < 0) || (dim > xDesc.GetLengths().size()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Reduce: is greater than 0 and less than or equal tensor dimension length.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && reduceDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Unpacked tensors not supported.");
#else
            return false;
#endif
        }

        return true;
    }

    bool IsNotLastDim() const
    {
        if(dim == xDesc.GetLengths().size() - 1)
            return false;
        return true;
    }

    bool IsLargeReduceSize() const
    {
        if(xDesc.GetLengths()[dim] > 64)
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    miopenSumNanPropagation_t nanPropagation;
    TensorDescriptor xDesc;
    TensorDescriptor reduceDesc;

    int32_t dim;
    miopenReduceExtremeOp_t op;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace reduce

} // namespace miopen
