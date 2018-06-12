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
#include <string>

namespace miopen {

TensorDescriptor::TensorDescriptor() : packed(true) {}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, std::initializer_list<std::size_t> plens)
    : lens(plens), packed(true), type(t)
{
    this->CalculateStrides();
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   std::initializer_list<std::size_t> plens,
                                   std::initializer_list<std::size_t> pstrides)
    : lens(plens), strides(pstrides), type(t)
{
    packed = (this->GetElementSize() == this->GetElementSpace());
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, const int* plens, int size)
    : lens(plens, plens + size), packed(true), type(t)
{
    if(!std::all_of(plens, plens + size, [](int x) { return x >= 0; }))
        MIOPEN_THROW("Invalid length. Length must be greater than 0.");
    this->CalculateStrides();
}
TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const int* plens,
                                   const int* pstrides,
                                   int size)
    : lens(plens, plens + size), strides(pstrides, pstrides + size), type(t)
{
    if(!std::all_of(plens, plens + size, [](int x) { return x >= 0; }))
        MIOPEN_THROW("Invalid length. Length must be greater than 0.");
    if(!std::all_of(pstrides, pstrides + size, [](int x) { return x >= 0; }))
        MIOPEN_THROW("Invalid strides. Strides must be greater than 0.");
    packed = (this->GetElementSize() == this->GetElementSpace());
}

void TensorDescriptor::CalculateStrides()
{
    strides.clear();
    strides.resize(lens.size(), 0);
    if(strides.empty())
        return;
    strides.back() = 1;
    std::partial_sum(
        lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());
}

const std::vector<std::size_t>& TensorDescriptor::GetLengths() const { return lens; }
const std::vector<std::size_t>& TensorDescriptor::GetStrides() const { return strides; }
int TensorDescriptor::GetSize() const
{
    assert(lens.size() == strides.size());
    return lens.size();
}
std::size_t TensorDescriptor::GetElementSize() const
{
    assert(lens.size() == strides.size());
    return std::accumulate(
        lens.begin(), lens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}
miopenDataType_t TensorDescriptor::GetType() const { return this->type; }

std::size_t TensorDescriptor::GetIndex(std::initializer_list<int> l) const
{
    assert(l.size() <= this->GetSize());
    return std::inner_product(l.begin(), l.end(), strides.begin(), std::size_t{0});
}

std::size_t TensorDescriptor::GetElementSpace() const
{
    std::vector<std::size_t> maxIndices(lens.size());
    std::transform(lens.begin(),
                   lens.end(),
                   std::vector<std::size_t>(lens.size(), 1).begin(),
                   maxIndices.begin(),
                   std::minus<std::size_t>());
    return std::inner_product(
               maxIndices.begin(), maxIndices.end(), strides.begin(), std::size_t{0}) +
           1;
}

std::size_t TensorDescriptor::GetNumBytes() const
{
    std::size_t typesize = 0;
    switch(this->type)
    {
    case miopenHalf: typesize  = 2; break;
    case miopenFloat: typesize = 4; break;
    }
    return typesize * this->GetElementSpace();
}

bool TensorDescriptor::IsPacked() const { return this->packed; }

TensorDescriptor TensorDescriptor::GetFlattenedTensorDescriptor() const
{
    // is packed
    if(IsPacked())
        return {GetType(), {GetElementSize()}, {1}};

    // ignore dimensions, where length is 1
    std::size_t non1_ndim = 0;
    std::vector<std::size_t> non1_lengths;
    std::vector<std::size_t> non1_strides;

    for(std::size_t i = 0; i < GetSize(); ++i)
    {
        std::size_t len = GetLengths()[i];
        if(len > 1)
        {
            ++non1_ndim;
            non1_lengths.push_back(len);
            non1_strides.push_back(GetStrides()[i]);
        }
    }

    // is a scalar
    if(non1_ndim == 0)
        return {GetType(), {1}, {1}};

    // start flattening tensor
    std::vector<std::size_t> flat_lengths;
    std::vector<std::size_t> flat_strides;

#if 0
    std::size_t flat_len = non1_lengths[0];
    for(std::size_t i = 1; i < non1_ndim; ++i)
    {
        std::size_t len      = non1_lengths[i];
        std::size_t full_len = non1_strides[i - 1] / non1_strides[i];

        if(len == full_len)
        {
            flat_len *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);
            flat_strides.push_back(non1_strides[i - 1]);
            flat_len = non1_lengths[i];
        }
    }
    flat_lengths.push_back(flat_len);
    flat_strides.push_back(non1_strides[non1_ndim - 1]);
#elif 0
    auto i_len = non1_lengths.begin();
    auto i_stride = non1_strides.begin();

    std::size_t flat_len = *i_len;

    auto i_previous_len = i_len++;
    auto i_previous_stride = i_stride++;

    while(i_len != non1_lengths.end())
    {
        std::size_t len = *i_len;
        std::size_t full_len = (*i_previous_stride) / (*i_stride);

        if(len == full_len)
        {
            flat_len *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);
            flat_strides.push_back(*i_previous_stride);
            flat_len = len;
        }
        i_previous_len = i_len++;
        i_previous_stride = i_stride++;
    };
    flat_lengths.push_back(flat_len);
    flat_strides.push_back(*i_previous_stride);
#else
    auto i_len = non1_lengths.begin();
    auto i_stride = non1_strides.begin();

    std::size_t flat_len = *i_len;

    auto i_previous_len = i_len++;
    auto i_previous_stride = i_stride++;

    while(i_len != non1_lengths.end())
    {
        std::size_t len = *i_len;
        std::size_t full_len = (*i_previous_stride) / (*i_stride);

        if(len == full_len)
        {
            flat_len *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);
            flat_strides.push_back(*i_previous_stride);
            flat_len = len;
        }
        i_previous_len = i_len++;
        i_previous_stride = i_stride++;
    };
    flat_lengths.push_back(flat_len);
    flat_strides.push_back(*i_previous_stride);
#endif

    return {GetType(), std::move(flat_lengths), std::move(flat_strides)};
}

bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
{
    assert(this->lens.size() == rhs.strides.size());
    return this->type == rhs.type && this->lens == rhs.lens && this->strides == rhs.strides;
}

bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const { return !(*this == rhs); }

bool TensorDescriptor::operator<(const TensorDescriptor& rhs) const
{
    return (std::tie(this->GetLengths(), this->GetStrides()) <
            std::tie(rhs.GetLengths(), rhs.GetStrides()));
}

bool TensorDescriptor::operator>(const TensorDescriptor& rhs) const
{
    return (std::tie(this->GetLengths(), this->GetStrides()) >
            std::tie(rhs.GetLengths(), rhs.GetStrides()));
}

std::string TensorDescriptor::ToString() const
{
    std::string result;
    if(this->lens.empty())
        return result;
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
