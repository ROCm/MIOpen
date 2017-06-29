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
#include <algorithm>
#include <cassert>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <numeric>
#include <string>

namespace miopen {

TensorDescriptor::TensorDescriptor() {}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, std::initializer_list<int> plens)
    : lens(plens), type(t)
{
    this->CalculateStrides();
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   std::initializer_list<int> plens,
                                   std::initializer_list<int> pstrides)
    : lens(plens), strides(pstrides), type(t)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, const int* plens, int size)
    : lens(plens, plens + size), type(t)
{
    this->CalculateStrides();
}
TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const int* plens,
                                   const int* pstrides,
                                   int size)
    : lens(plens, plens + size), strides(pstrides, pstrides + size), type(t)
{
}

void TensorDescriptor::CalculateStrides()
{
    strides.clear();
    strides.resize(lens.size(), 0);
    strides.back() = 1;
    std::partial_sum(lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<int>());
}

const std::vector<int>& TensorDescriptor::GetLengths() const { return lens; }
const std::vector<int>& TensorDescriptor::GetStrides() const { return strides; }
int TensorDescriptor::GetSize() const
{
    assert(lens.size() == strides.size());
    return lens.size();
}
int TensorDescriptor::GetElementSize() const
{
    assert(lens.size() == strides.size());
    return std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<int>());
}
miopenDataType_t TensorDescriptor::GetType() const { return this->type; }

int TensorDescriptor::GetIndex(std::initializer_list<int> l) const
{
    assert(l.size() <= this->GetSize());
    return std::inner_product(l.begin(), l.end(), strides.begin(), 0);
}

bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
{
    assert(this->lens.size() == rhs.strides.size());
    return this->type == rhs.type && this->lens == rhs.lens && this->strides == rhs.strides;
}

bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const { return !(*this == rhs); }

std::string TensorDescriptor::ToString() const
{
    std::string result;
    for(auto i : this->lens)
    {
        result += std::to_string(i) + ", ";
    }
    return result.substr(0, result.length() - 2);
}

std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
{
    return LogRange(stream, t.lens, ", ");
}

} // namespace miopen

// TODO(paul): Remove
MIOPEN_EXPORT
int miopenGetTensorIndex(miopenTensorDescriptor_t tensorDesc, std::initializer_list<int> indices)
{
    return miopen::deref(tensorDesc).GetIndex(indices);
}
