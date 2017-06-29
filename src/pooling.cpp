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
#include <cassert>
#include <cmath>
#include <cmath>
#include <miopen/logger.hpp>
#include <miopen/pooling.hpp>

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

PoolingDescriptor::PoolingDescriptor(
    miopenPoolingMode_t m, const int* plens, const int* ppads, const int* pstrides, int size)
    : lens(plens, plens + size),
      strides(pstrides, pstrides + size),
      pads(ppads, ppads + size),
      mode(m)
{
}

PoolingDescriptor::PoolingDescriptor(miopenPoolingMode_t m,
                                     std::vector<int> plens,
                                     std::vector<int> pstrides,
                                     std::vector<int> ppads)
    : lens(plens), strides(pstrides), pads(ppads), mode(m)
{
}

miopenPoolingMode_t PoolingDescriptor::GetMode() const { return (mode); }

const std::vector<int>& PoolingDescriptor::GetLengths() const { return lens; }
const std::vector<int>& PoolingDescriptor::GetStrides() const { return strides; }

const std::vector<int>& PoolingDescriptor::GetPads() const { return pads; }

miopenPoolingMode_t PoolingDescriptor::GetMode() { return mode; }

int PoolingDescriptor::GetSize() const
{
    assert(lens.size() == strides.size() && lens.size() == pads.size());
    return lens.size();
}

std::tuple<int, int, int, int>
PoolingDescriptor::GetForwardOutputDim(const TensorDescriptor& tensorDesc) const
{

    assert(tensorDesc.GetLengths().size() == 4);

    int input_n;
    int input_c;
    int input_h;
    int input_w;

    std::tie(input_n, input_c, input_h, input_w) = miopen::tie4(tensorDesc.GetLengths());

    int u, v, pad_h, pad_w, window_h, window_w;
    std::tie(u, v)               = miopen::tie2(GetStrides());
    std::tie(pad_h, pad_w)       = miopen::tie2(GetPads());
    std::tie(window_h, window_w) = miopen::tie2(GetLengths());

    return std::make_tuple(
        input_n,
        input_c,
        std::max(1,
                 static_cast<int>(
                     std::ceil((input_h - window_h + 2 * pad_h) / static_cast<float>(u)) + 1)),
        std::max(1,
                 static_cast<int>(
                     std::ceil((input_w - window_w + 2 * pad_w) / static_cast<float>(v)) + 1)));
}

TensorDescriptor PoolingDescriptor::GetForwardOutputTensor(const TensorDescriptor& tensorDesc) const
{
    auto dims = this->GetForwardOutputDim(tensorDesc);
    return TensorDescriptor(
        tensorDesc.GetType(),
        {std::get<0>(dims), std::get<1>(dims), std::get<2>(dims), std::get<3>(dims)});
}

std::ostream& operator<<(std::ostream& stream, const PoolingDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream, x.mode, miopenPoolingMax, miopenPoolingAverage) << ", ";
    LogRange(stream, x.lens, ", ") << ", ";
    LogRange(stream, x.strides, ", ") << ", ";
    LogRange(stream, x.pads, ", ") << ", ";
    return stream;
}

} // namespace miopen
