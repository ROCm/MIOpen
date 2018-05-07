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
#include <miopen/functional.hpp>
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
    std::vector<std::size_t> full_lengths(non1_ndim);

    full_lengths[0] = 0; // the 0-th dimension full-length doesn't matter
    for(std::size_t i   = 1; i < non1_ndim; ++i)
        full_lengths[i] = non1_strides[i - 1] / non1_strides[i];

    std::vector<std::size_t> flat_lengths;
    std::vector<std::size_t> flat_strides;

    std::size_t flat_len = non1_lengths[0];
    for(std::size_t i = 1; i < non1_ndim; ++i)
    {
        std::size_t len      = non1_lengths[i];
        std::size_t full_len = full_lengths[i];

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

    return {GetType(), flat_lengths, flat_strides};
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

template <typename... TDescriptors>
void get_consistent_flattened_tensor_descriptors(
    std::tuple<const TDescriptors&...> real_descriptors,
    std::tuple<TDescriptors&...> flat_descriptors)
{
    constexpr std::size_t NTensor = std::tuple_size<std::tuple<TDescriptors&...>>::value;

    using TN = std::integral_constant<std::size_t, NTensor>;

    if(NTensor == 0)
        MIOPEN_THROW(miopenStatusBadParm, "NTensor == 0.");

    miopenDataType_t data_type = std::get<0>(real_descriptors).GetType();

#if 0
    call_n_time(
        [&](const auto i) {
            constexpr int itensor = decltype(i)::value;
            std::cout << __func__ << ": tensor " << itensor << ", real_lengths "
                      << std::get<itensor>(real_descriptors).GetLengths() << ", real_strides "
                      << std::get<itensor>(real_descriptors).GetStrides() << std::endl;
        },
        TN{});
#endif

#if 1
    // check input tensor descriptors
    auto& real_desc_0_lens = std::get<0>(real_descriptors).GetLengths();

    call_n_time(
        [&](const auto i) {
            constexpr std::size_t itensor = decltype(i)::value;

            if(i > 0)
            {
                auto& real_desc_lens = std::get<itensor>(real_descriptors).GetLengths();

                if(real_desc_0_lens != real_desc_lens)
                    MIOPEN_THROW(miopenStatusBadParm, "Lengths of Tensors are different.");
            }
        },
        TN{});
#endif

    // check if is all packed
    bool is_all_packed = true;

    call_n_time(
        [&](const auto i) {
            constexpr int itensor = decltype(i)::value;

            if(is_all_packed)
            {
                if(!std::get<itensor>(real_descriptors).IsPacked())
                    is_all_packed = false;
            }
        },
        TN{});
    // std::cout << __func__ << ": is_all_packed: " << is_all_packed << std::endl;

    // all packed
    if(is_all_packed)
    {
        std::size_t element_size = std::get<0>(real_descriptors).GetElementSize();

        call_n_time(
            [&](const auto i) {
                constexpr int itensor = decltype(i)::value;
                std::get<itensor>(flat_descriptors) =
                    TensorDescriptor{data_type, {element_size}, {1}};
            },
            TN{});

        return; // early return for all-packed tensors
    }

#if 1
    // ignore dimensions, where non1_lengths of all tensors are 1
    const std::size_t real_ndim                  = std::get<0>(real_descriptors).GetSize();
    const std::vector<std::size_t>& real_lengths = std::get<0>(real_descriptors).GetLengths();

    std::size_t non1_ndim = 0;
    std::vector<std::size_t> non1_lengths;
    std::array<std::vector<std::size_t>, NTensor> array_of_non1_strides;

    for(std::size_t idim = 0; idim < real_ndim; ++idim)
    {
        std::size_t len = real_lengths[idim];
        if(len > 1)
        {
            ++non1_ndim;
            non1_lengths.push_back(len);

            call_n_time(
                [&](const auto i) {
                    constexpr int itensor = decltype(i)::value;
                    array_of_non1_strides[itensor].push_back(
                        std::get<itensor>(real_descriptors).GetStrides()[idim]);
                    return true;
                },
                TN{});
        }
    } // now, non1_ndim, non1_lengths, array_of_non1_strides contains non-1-length dimensions

// std::cout << __func__ << ": non1_ndim: " << non1_ndim << std::endl;
// std::cout << __func__ << ": non1_lengths: " << non1_lengths << std::endl;
#endif

    // is scalar
    if(non1_ndim == 0)
    {
        call_n_time(
            [&](const auto i) {
                constexpr int itensor               = decltype(i)::value;
                std::get<itensor>(flat_descriptors) = TensorDescriptor{data_type, {1}, {1}};
            },
            TN{});

        return; // early return for all-scalar tensors
    }

    // start flattening tensors
    std::array<std::vector<std::size_t>, NTensor> array_of_full_lengths;

    call_n_time(
        [&](const auto i) {
            constexpr int itensor = decltype(i)::value;
            array_of_full_lengths[itensor].reserve(non1_ndim);
        },
        TN{});

    for(std::size_t idim = 1; idim < non1_ndim; ++idim)
    // the 0-th dimension full-length doesn't matter
    {
        for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
            array_of_full_lengths[itensor][idim] =
                array_of_non1_strides[itensor][idim - 1] / array_of_non1_strides[itensor][idim];
    }

    std::vector<std::size_t> flat_lengths;
    std::array<std::vector<std::size_t>, NTensor> array_of_flat_strides;

    std::size_t flat_len = non1_lengths[0];
    for(std::size_t idim = 1; idim < non1_ndim; ++idim)
    {
        std::size_t len = non1_lengths[idim];

        bool is_all_full_length = true;
        for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
        {
            if(len != array_of_full_lengths[itensor][idim])
            {
                is_all_full_length = false;
                break;
            }
        }

        if(is_all_full_length)
        {
            flat_len *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);

            for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
                array_of_flat_strides[itensor].push_back(array_of_non1_strides[itensor][idim - 1]);

            flat_len = non1_lengths[idim];
        }
    }

    flat_lengths.push_back(flat_len);
    for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
        array_of_flat_strides[itensor].push_back(array_of_non1_strides[itensor][non1_ndim - 1]);

    call_n_time(
        [&](const auto i) {
            constexpr int itensor = decltype(i)::value;
            std::get<itensor>(flat_descriptors) =
                TensorDescriptor{data_type, flat_lengths, array_of_flat_strides[itensor]};
        },
        TN{});

#if 0
    // print
    call_n_time(
        [&](const auto i) {
            constexpr int itensor = decltype(i)::value;
            std::cout << __func__ << ": tensor " << itensor << ", flat_lengths "
                      << std::get<itensor>(flat_descriptors).GetLengths() << ", flat_strides "
                      << std::get<itensor>(flat_descriptors).GetStrides() << std::endl;
        },
        TN{});
#endif
}

template void get_consistent_flattened_tensor_descriptors(
    std::tuple<const TensorDescriptor&, const TensorDescriptor&> real_descriptors,
    std::tuple<TensorDescriptor&, TensorDescriptor&> flat_descriptors);

template void get_consistent_flattened_tensor_descriptors(
    std::tuple<const TensorDescriptor&, const TensorDescriptor&, const TensorDescriptor&>
        real_descriptors,
    std::tuple<TensorDescriptor&, TensorDescriptor&, TensorDescriptor&> flat_descriptors);

} // namespace miopen

// TODO(paul): Remove
MIOPEN_EXPORT
int miopenGetTensorIndex(miopenTensorDescriptor_t tensorDesc, std::initializer_list<int> indices)
{
    return miopen::deref(tensorDesc).GetIndex(indices);
}
