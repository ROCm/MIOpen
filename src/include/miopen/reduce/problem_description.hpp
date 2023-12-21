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

#include <cassert>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace reduce {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(miopenSumNanPropagation_t nanPropagation_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       int32_t* dims_,
                       int32_t dims_size_)
        : nanPropagation(nanPropagation_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          dims(dims_),
          dims_size(dims_size_)
    {
    }

    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       int32_t* dims_,
                       int32_t dims_size_)
        : xDesc(xDesc_), yDesc(yDesc_), dims(dims_), dims_size(dims_size_)
    {
    }

    miopenSumNanPropagation_t GetNanPropagation_() const { return nanPropagation; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    int32_t* GetDims() const { return dims; }
    int32_t GetDims_size() const { return dims_size; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor types do not match.");
        }
        return true;
    }

    bool IsRightLength() const
    {
        auto in_length = xDesc.GetLengths();
        std::vector<std::size_t> out_length;
        for(int i = 0; i < in_length.size(); i++)
        {
            bool not_reduce = true;
            for(int j = 0; j < dims_size; j++)
            {
                if(i == dims[j])
                {
                    not_reduce = false;
                    continue;
                }
            }
            if(not_reduce)
            {
                out_length.push_back(in_length[i]);
                not_reduce = true;
            }
        }

        for(int32_t i = 0; i < out_length.size(); i++)
        {
            if(out_length[i] != yDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor dimension lengths do not match.");
            }
        }
        return true;
    }

    bool IsRightDim() const
    {
        for(int32_t i = 0; i < dims_size; i++)
        {
            int32_t dim = dims[i];
            if((dim < 0) || (dim > xDesc.GetLengths().size()))
            {
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    "Reduce: is greater than 0 and less than or equal tensor dimension length.");
            }
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Unpacked tensors not supported.");
        }
        return true;
    }

    bool IsNotLastDim() const
    {
        for(int32_t i = 0; i < dims_size; i++)
        {
            int32_t dim = dims[i];
            if(dim == xDesc.GetLengths().size() - 1)
                return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    miopenSumNanPropagation_t nanPropagation;
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;

    int32_t* dims;
    int32_t dims_size;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace reduce

} // namespace miopen
