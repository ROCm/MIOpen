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
#ifndef GUARD_MIOPEN_TENSOR_DRIVER_HPP
#define GUARD_MIOPEN_TENSOR_DRIVER_HPP

#include "driver.hpp"

#include <algorithm>
#include <iterator>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_extra.hpp>
#include <miopen/tensor_layout.hpp>
#include <numeric>
#include <vector>

inline std::vector<int> GetTensorLengths(miopenTensorDescriptor_t& tensor)
{
    int n;
    int c;
    int h;
    int w;
    int d;

    int size = 0;
    miopenGetTensorDescriptorSize(tensor, &size);

    if(size == 5)
    {
        miopenGet5dTensorDescriptorLengths(tensor, &n, &c, &d, &h, &w);
        return std::vector<int>({n, c, d, h, w});
    }
    else if(size == 4)
    {
        miopenGet4dTensorDescriptorLengths(tensor, &n, &c, &h, &w);
        return std::vector<int>({n, c, h, w});
    }

    std::vector<int> tensor_len;
    tensor_len.resize(miopen::deref(tensor).GetSize());
    miopenGetTensorDescriptor(tensor, nullptr, tensor_len.data(), nullptr);

    return tensor_len;
}

inline std::vector<int> GetTensorStrides(miopenTensorDescriptor_t& tensor)
{
    int nstride;
    int cstride;
    int dstride;
    int hstride;
    int wstride;

    int size = 0;
    miopenGetTensorDescriptorSize(tensor, &size);

    if(size == 5)
    {
        miopenGet5dTensorDescriptorStrides(
            tensor, &nstride, &cstride, &dstride, &hstride, &wstride);
        return std::vector<int>({nstride, cstride, dstride, hstride, wstride});
    }
    else if(size == 4)
    {
        miopenGet4dTensorDescriptorStrides(tensor, &nstride, &cstride, &hstride, &wstride);
        return std::vector<int>({nstride, cstride, hstride, wstride});
    }

    std::vector<int> tensor_strides;
    tensor_strides.resize(miopen::deref(tensor).GetSize());

    miopenGetTensorDescriptor(tensor, nullptr, nullptr, tensor_strides.data());

    return tensor_strides;
}

inline int SetTensor4d(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSet4dTensorDescriptor(t, data_type, UNPACK_VEC4(len));
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), nullptr);
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       std::vector<int>& strides,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), strides.data());
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       const std::string& layout,
                       miopenDataType_t data_type = miopenFloat)
{
    if(layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    if(layout.size() != len.size())
    {
        MIOPEN_THROW("unmatched layout and dimension size");
    }

    // Dimension lengths vector 'len' comes with a default layout.
    std::string len_layout = miopen::tensor_layout_get_default(layout.size());
    if(len_layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    std::vector<int> strides;
    miopen::tensor_layout_to_strides(len, len_layout, layout, strides);

    return SetTensorNd(t, len, strides, data_type);
}

// This function ignores tensor strides completely and its result should not be interpreted as
// memory required for an "unpacked" tensor. In such cases GetTensorSpace should be used instead.
// For "packed" tensors result may be interpreted as an amount of memory required for the tensor.
// The implementation is a copy-paste from miopen::TensorDescriptor.
inline size_t GetTensorSize(miopenTensorDescriptor_t& tensor)
{
    assert(miopen::deref(tensor).IsPacked() &&
           "GetTensorSize should not be used on an unpacked tensor.");
    const auto len = GetTensorLengths(tensor);
    size_t sz      = std::accumulate(len.begin(), len.end(), size_t{1}, std::multiplies<size_t>());

    return sz;
}

// The result of this function may be interpreted as a correct amount of memory required for both
// "packed" and "unpacked" tensors. In general it should be used for such purposes rather than
// GetTensorSize. Unless, of course, there is absolutely zero chance to receive an unpacked tensor.
inline size_t GetTensorSpace(miopenTensorDescriptor_t& tensor)
{
    return miopen::deref(tensor).GetElementSpace();
}
#endif // GUARD_MIOPEN_TENSOR_DRIVER_HPP
