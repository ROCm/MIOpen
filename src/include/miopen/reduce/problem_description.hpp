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
                       const TensorDescriptor& yDesc_,
                       int32_t dim_)
        : nanPropagation(nanPropagation_), xDesc(xDesc_), yDesc(yDesc_), dim(dim_)
    {
    }

    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& indiceDesc_,
                       int32_t dim_,
                       miopenReduceExtremeOp_t reduceExtremeOp_)
        : xDesc(xDesc_),
          yDesc(yDesc_),
          indiceDesc(indiceDesc_),
          dim(dim_),
          reduceExtremeOp(reduceExtremeOp_)
    {
    }

    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& indiceDesc_,
                       int32_t dim_,
                       miopenReduceExtremeOp_t reduceExtremeOp_)
        : xDesc(xDesc_), indiceDesc(indiceDesc_), dim(dim_), reduceExtremeOp(reduceExtremeOp_)
    {
    }

    miopenSumNanPropagation_t GetNanPropagation_() const { return nanPropagation; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetIndiceDesc() const { return indiceDesc; }
    int32_t GetDim() const { return dim; }

    bool IsValidLength() const
    {
        if(xDesc.GetLengths().size() == 1)
            return true;

        int32_t posy = 0;
        for(int32_t i = 0; i < xDesc.GetLengths().size(); ++i)
        {
            if(i == dim)
                continue;

            if(xDesc.GetLengths()[i] != yDesc.GetLengths()[posy])
            {
                MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor dimension lengths do not match.");
            }

            ++posy;
        }
        return true;
    }

    bool IsValidLengthIndice() const
    {
        if(xDesc.GetLengths().size() == 1)
            return true;

        int32_t posy = 0;
        for(int32_t i = 0; i < xDesc.GetLengths().size(); ++i)
        {
            if(i == dim)
                continue;

            if(xDesc.GetLengths()[i] != indiceDesc.GetLengths()[posy])
            {
                MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor dimension lengths do not match.");
            }

            ++posy;
        }
        return true;
    }

    bool IsValidDim() const
    {
        if((dim < 0) || (dim > xDesc.GetLengths().size()))
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Reduce: is greater than 0 and less than or equal tensor dimension length.");
        }
        return true;
    }

    bool IsValidInputNumel() const
    {
        auto xdims = xDesc.GetLengths();
        auto input_numel =
            std::accumulate(xdims.begin(), xdims.end(), 1ULL, std::multiplies<size_t>());
        if(input_numel > INT32_MAX)
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: input numel is bigger than INT_MAX.");

        return true;
    }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
            return false;
        }

        return true;
    }

    bool IsAllPackedWithIndice() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked() && indiceDesc.IsPacked()))
        {
            return false;
        }

        return true;
    }

    bool IsAllPackedIndice() const
    {
        if(!(xDesc.IsPacked() && indiceDesc.IsPacked()))
        {
            return false;
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
    TensorDescriptor yDesc;
    TensorDescriptor indiceDesc;

    int32_t dim;

    miopenReduceExtremeOp_t reduceExtremeOp = MIOPEN_REDUCE_CALCULATION_SUM;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace reduce

} // namespace miopen
