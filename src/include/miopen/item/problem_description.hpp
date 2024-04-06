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

namespace item {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& dyDesc_,
                       int32_t indexCount_,
                       const TensorDescriptor* const* indexDescs_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& dxDesc_,
                       int32_t dimCount_,
                       int32_t* dims_,
                       int32_t sliceCount_,
                       int32_t* slices_,
                       int32_t offset_)
        : dyDesc(dyDesc_),
          indexCount(indexCount_),
          indexDescs(indexDescs_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          dxDesc(dxDesc_),
          dimCount(dimCount_),
          dims(dims_),
          sliceCount(sliceCount_),
          slices(slices_),
          offset(offset_)
    {
    }

    ProblemDescription(const TensorDescriptor* const* indexDescs_,
                       ConstData_t* indexs_,
                       int32_t dimCount_,
                       int32_t* dims_)
        : indexDescs(indexDescs_), indexs(indexs_), dimCount(dimCount_), dims(dims_)
    {
    }

    const TensorDescriptor& GetDyDesc() const { return dyDesc; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    int32_t GetIndexCount() const { return indexCount; }
    const TensorDescriptor& GetIndexDesc(int i) const
    {
        if(i >= indexCount)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Item: Invalid tensor index.");
        }
        return *indexDescs[i];
    }
    int32_t GetXCount() const { return xCount; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetDxDesc() const { return dxDesc; }
    int32_t GetDimCount() const { return dimCount; }
    int32_t GetDim(int i) const
    {
        if(i >= indexCount)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Item: Invalid dim index.");
        }
        return dims[i];
    }
    int32_t GetSliceCount() const { return sliceCount; }
    int32_t GetSlice(int i) const
    {
        if(i >= sliceCount)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Item: Invalid slice index.");
        }
        return slices[i];
    }
    int32_t GetOffset() const { return offset; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Item: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightDim() const
    {
        if((dim < 0) || (dim > xDesc.GetLengths().size()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Item: is greater than 0 and less than or equal tensor dimension length.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Item: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor dyDesc;
    TensorDescriptor xDesc;
    TensorDescriptor* indexDescs;
    int32_t indexCount;
    TensorDescriptor yDesc;
    TensorDescriptor dxDesc;

    int32_t dimCount;
    int32_t* dims;
    int32_t sliceCount;
    int32_t* slices;
    int32_t offset;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace item

} // namespace miopen
