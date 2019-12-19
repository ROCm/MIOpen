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
#include <miopen/pooling.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>

#include <cassert>
#include <cmath>
#include <cmath>

namespace miopen {

template <class T, class U>
T iciel_div(T x, U y)
{
    auto rem = x % y;
    if(rem > 0)
        rem = 1;
    return ((x - rem) * y);
}

PoolingDescriptor::PoolingDescriptor() {}

PoolingDescriptor::PoolingDescriptor(miopenPoolingMode_t m,
                                     miopenPaddingMode_t pm,
                                     const int* plens,
                                     const int* ppads,
                                     const int* pstrides,
                                     int size)
    : lens(plens, plens + size),
      strides(pstrides, pstrides + size),
      pads(ppads, ppads + size),
      mode(m),
      pmode(pm),
      indexType(miopenIndexUint8)
{
}

PoolingDescriptor::PoolingDescriptor(miopenPoolingMode_t m,
                                     miopenPaddingMode_t pm,
                                     const std::vector<int>& plens,
                                     const std::vector<int>& pstrides,
                                     const std::vector<int>& ppads)
    : lens(plens), strides(pstrides), pads(ppads), mode(m), pmode(pm), indexType(miopenIndexUint8)
{
}

void PoolingDescriptor::SetIndexType(miopenIndexType_t index_type) { indexType = index_type; }

miopenIndexType_t PoolingDescriptor::GetIndexType() const { return indexType; }

miopenPoolingMode_t PoolingDescriptor::GetMode() const { return (mode); }

miopenPaddingMode_t PoolingDescriptor::GetPaddingMode() const { return (pmode); }

const std::vector<int>& PoolingDescriptor::GetLengths() const { return lens; }

const std::vector<int>& PoolingDescriptor::GetStrides() const { return strides; }

const std::vector<int>& PoolingDescriptor::GetPads() const { return pads; }

miopenPoolingMode_t PoolingDescriptor::GetMode() { return mode; }

int PoolingDescriptor::GetSize() const
{
    assert(lens.size() == strides.size() && lens.size() == pads.size());
    return lens.size();
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
PoolingDescriptor::GetForwardOutputDim(const TensorDescriptor& xDesc) const
{

    assert(xDesc.GetLengths().size() == 4);

    std::size_t input_n;
    std::size_t input_c;
    std::size_t input_h;
    std::size_t input_w;

    std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(xDesc.GetLengths());

    int stride_h, stride_w, pad_h, pad_w, window_h, window_w;
    std::tie(stride_h, stride_w) = miopen::tien<2>(GetStrides());
    std::tie(pad_h, pad_w)       = miopen::tien<2>(GetPads());
    std::tie(window_h, window_w) = miopen::tien<2>(GetLengths());
    miopenPaddingMode_t _pMode = GetPaddingMode();

    assert(stride_h > 0);
    assert(stride_w > 0);
    assert(window_h < (input_h + 2 * pad_h));
    assert(window_w < (input_w + 2 * pad_w));

    auto output_h = std::max<std::ptrdiff_t>(
        1, std::ptrdiff_t((input_h + 2 * pad_h - window_h) / stride_h + 1));
    auto output_w = std::max<std::ptrdiff_t>(
        1, std::ptrdiff_t((input_w + 2 * pad_w - window_w) / stride_w + 1));

    if(_pMode == miopenPaddingSame)
    {
        output_h = std::ceil(static_cast<double>(input_h) / static_cast<double>(stride_h));
        output_w = std::ceil(static_cast<double>(input_w) / static_cast<double>(stride_w));
    }
    else if(_pMode == miopenPaddingValid)
    {
        output_h =
            std::ceil(static_cast<double>(input_h - window_h + 1) / static_cast<double>(stride_h));
        output_w =
            std::ceil(static_cast<double>(input_w - window_w + 1) / static_cast<double>(stride_w));
    }

    return std::make_tuple(input_n, input_c, output_h, output_w);
}

TensorDescriptor PoolingDescriptor::GetForwardOutputTensor(const TensorDescriptor& xDesc) const
{
    auto dims = this->GetForwardOutputDim(xDesc);
    return TensorDescriptor(
        xDesc.GetType(),
        {std::get<0>(dims), std::get<1>(dims), std::get<2>(dims), std::get<3>(dims)});
}

std::size_t PoolingDescriptor::GetWorkSpaceSize(const TensorDescriptor& yDesc) const
{
    return GetMode() == miopenPoolingMax ? yDesc.GetElementSize() * get_data_size(GetIndexType())
                                         : 0;
}

std::ostream& operator<<(std::ostream& stream, const PoolingDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream, x.mode, miopenPoolingMax, miopenPoolingAverage) << ", ";
    LogRange(stream, x.lens, ", ") << ", ";
    LogRange(stream, x.pads, ", ") << ", ";
    LogRange(stream, x.strides, ", ") << ", ";
    return stream;
}

} // namespace miopen
