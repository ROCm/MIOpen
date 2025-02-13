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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace getitem {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& dyDesc_,
                       uint32_t indexCount_,
                       const TensorDescriptor* const* indexDescs_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& errorDesc_,
                       uint32_t dimCount_,
                       const int32_t* dims_,
                       uint32_t sliceCount_,
                       const int32_t* slices_,
                       uint32_t offset_)
        : dyDesc(dyDesc_),
          indexCount(indexCount_),
          indexDescs(indexDescs_),
          dxDesc(dxDesc_),
          errorDesc(errorDesc_),
          dimCount(dimCount_),
          dims(dims_),
          sliceCount(sliceCount_),
          slices(slices_),
          offset(offset_)
    {
        IsValidIndexsLength();
        IsValidIndexs();
        IsValidDims();
        IsValidSlices();
    }

    ProblemDescription(const int32_t indexCount_, const TensorDescriptor* const* indexDescs_)
        : indexCount(indexCount_), indexDescs(indexDescs_)
    {
        IsValidIndexsLength();
        IsValidIndexs();
    }

    const TensorDescriptor& GetDYDesc() const { return dyDesc; }
    int32_t GetIndexCount() const { return indexCount; }
    const TensorDescriptor& GetIndexDesc(int i) const
    {
        if(i >= indexCount)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Item: Invalid tensor index.");
        }
        return (*indexDescs)[i];
    }
    const TensorDescriptor& GetDXDesc() const { return dxDesc; }
    const TensorDescriptor& GetErrorDesc() const { return errorDesc; }
    int32_t GetDimCount() const { return dimCount; }
    int32_t GetDim(int i) const
    {
        if(i >= indexCount)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Item: Invalid dim index.");
        }
        return dims[i];
    }
    int32_t GetSliceCount() const { return sliceCount; }
    int32_t GetSlice(int i) const
    {
        if(i >= sliceCount)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Item: Invalid slice index.");
        }
        return slices[i];
    }
    int32_t GetOffset() const { return offset; }

    bool IsValidIndexsLength() const
    {
        if(indexCount > 0)
        {
            auto firstlength = (*indexDescs)[0];
            for(int32_t i = 1; i < indexCount; ++i)
            {
                if(firstlength != (*indexDescs)[i])
                    MIOPEN_THROW(miopenStatusBadParm,
                                 "Getitem: Indexs dimension lengths do not match.");
            }
        }
        return true;
    }

    bool IsValidIndexs() const
    {
        if(indexCount > 0)
        {
            if(indexDescs == nullptr)
                MIOPEN_THROW(miopenStatusBadParm, "Getitem: indexDesc is nullptr.");
        }
        return true;
    }

    bool IsValidDims() const
    {
        if(dimCount > 0)

            if(dims == nullptr)
                MIOPEN_THROW(miopenStatusBadParm, "Getitem: dims is nullptr.");
        return true;
    }

    bool IsValidSlices() const
    {
        if(sliceCount > 0)
        {
            if(slices == nullptr)
                MIOPEN_THROW(miopenStatusBadParm, "Getitem: slices is nullptr.");
        }
        return true;
    }

    bool IsSameType() const
    {
        if(dyDesc.GetType() != dxDesc.GetType())
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor dyDesc{};
    uint32_t indexCount                       = 0;
    const TensorDescriptor* const* indexDescs = nullptr;
    TensorDescriptor dxDesc{};
    TensorDescriptor errorDesc{};

    uint32_t dimCount     = 0;
    const int32_t* dims   = nullptr;
    uint32_t sliceCount   = 0;
    const int32_t* slices = nullptr;
    uint32_t offset       = 0;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace getitem

} // namespace miopen
