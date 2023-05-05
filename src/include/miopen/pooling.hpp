/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_POOLING_HPP_
#define MIOPEN_POOLING_HPP_

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/object.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace miopen {

// get the previous (less or equal to v) power of 2
inline int prePow2(int v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return (v + 1) >> 1;
}

inline std::string get_pooling_index_type_name(miopenIndexType_t index_type)
{
    switch(index_type)
    {
    case miopenIndexUint8: {
        return "uchar";
    }
    case miopenIndexUint16: {
        return "ushort";
    }
    case miopenIndexUint32: {
        return "uint";
    }
    case miopenIndexUint64: {
        return "ulong";
    }
    }

    MIOPEN_THROW("not belong to any case");
}

inline std::string get_pooling_index_type_max_name(miopenIndexType_t index_type)
{
    switch(index_type)
    {
    case miopenIndexUint8: {
        return "UCHAR_MAX";
    }
    case miopenIndexUint16: {
        return "USHRT_MAX";
    }
    case miopenIndexUint32: {
        return "UINT_MAX";
    }
    case miopenIndexUint64: {
        return "ULONG_MAX";
    }
    }

    MIOPEN_THROW("not belong to any case");
}

struct Handle;
struct TensorDescriptor;

struct PoolingDescriptor : miopenPoolingDescriptor
{
    PoolingDescriptor();

    PoolingDescriptor(miopenPoolingMode_t m,
                      miopenPaddingMode_t pm,
                      const std::vector<int>& plens,
                      const std::vector<int>& pstrides,
                      const std::vector<int>& ppads);

    PoolingDescriptor(miopenPoolingMode_t m,
                      miopenPaddingMode_t pm,
                      const int* plens,
                      const int* ppads,
                      const int* pstrides,
                      int size);

    void SetIndexType(miopenIndexType_t index_type);

    void SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t workspace_index);

    miopenPoolingMode_t GetMode() const;

    miopenPaddingMode_t GetPaddingMode() const;

    miopenIndexType_t GetIndexType() const;

    miopenPoolingWorkspaceIndexMode_t GetWorkspaceIndexMode() const;

    const std::vector<int>& GetLengths() const;

    const std::vector<int>& GetStrides() const;

    const std::vector<int>& GetPads() const;

    int GetSize() const;

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
    GetForwardOutputDim(const TensorDescriptor& xDesc) const;

    void GetForwardOutputDimNd(const TensorDescriptor& xDesc, int dims, int* tensorDimArr) const;

    TensorDescriptor GetForwardOutputTensor(const TensorDescriptor& xDesc) const;

    std::size_t GetWorkSpaceSize(const TensorDescriptor& yDesc) const;

    miopenStatus_t Forward(Handle& handle,
                           const void* alpha,
                           const TensorDescriptor& xDesc,
                           ConstData_t x,
                           const void* beta,
                           const TensorDescriptor& yDesc,
                           Data_t y,
                           bool save_index,
                           Data_t workSpace,
                           size_t workSpaceSize) const;

    miopenStatus_t Backward(Handle& handle,
                            const void* alpha,
                            const TensorDescriptor& yDesc,
                            ConstData_t y,
                            const TensorDescriptor& dyDesc,
                            ConstData_t dy,
                            const TensorDescriptor& xDesc,
                            ConstData_t x,
                            const void* beta,
                            const TensorDescriptor& dxDesc,
                            Data_t dx,
                            Data_t workSpace) const;

    friend std::ostream& operator<<(std::ostream& stream, const PoolingDescriptor& x);

    std::vector<int> lens;
    std::vector<int> strides;
    std::vector<int> pads;

    miopenPoolingMode_t mode  = miopenPoolingMax;
    miopenPaddingMode_t pmode = miopenPaddingDefault;

    miopenIndexType_t indexType                          = miopenIndexUint8;
    miopenPoolingWorkspaceIndexMode_t workspaceIndexMode = miopenPoolingWorkspaceIndexMask;
};
} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenPoolingDescriptor, miopen::PoolingDescriptor);
#endif // _MIOPEN_POOLING_HPP_
