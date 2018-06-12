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
#ifndef GUARD_MIOPEN_TENSOR_HPP_
#define GUARD_MIOPEN_TENSOR_HPP_

#include <cassert>
#include <iostream>
#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/each_args.hpp>
#include <miopen/functional.hpp>
#include <miopen/returns.hpp>
#include <miopen/errors.hpp>
#include <vector>
#include <algorithm>
// TODO(paul): remove this include later
#include <cstdio>

namespace miopen {

template <class T, std::size_t... Ns>
auto tie_impl(T&& x, detail::seq<Ns...>) -> decltype(std::tie(x[Ns]...))
{
    assert(x.size() == sizeof...(Ns));
    return std::tie(x[Ns]...);
}

template <class T, class U, std::size_t... Ns>
auto tie_impl(T&& x, U y, detail::seq<Ns...>) -> decltype(std::make_tuple(x[Ns]...))
{
    return std::make_tuple((Ns < x.size() ? x[Ns] : y)...);
}

template <std::size_t N, class T>
auto tien(T&& x) MIOPEN_RETURNS(tie_impl(std::forward<T>(x), typename detail::gens<N>::type{}));

template <std::size_t N, class T, class U>
auto tien(T&& x, U y)
    MIOPEN_RETURNS(tie_impl(std::forward<T>(x), y, typename detail::gens<N>::type{}));

template <typename T, std::size_t... Ns>
auto to_tuple_impl(T&& x, detail::seq<Ns...>)
{
    return std::make_tuple(std::move(x[Ns])...);
}

template <typename T>
auto to_tuple(T&& x)
    MIOPEN_RETURNS(to_tuple_impl(std::forward<T>(x),
                                 typename detail::gens<std::tuple_size<T>::value>::type{}));

inline std::size_t GetTypeSize(miopenDataType_t d)
{
    switch(d)
    {
    case miopenFloat: return 4;
    case miopenHalf: return 2;
    }
    MIOPEN_THROW("Unknown data type");
}

struct TensorDescriptor : miopenTensorDescriptor
{
    TensorDescriptor();
    TensorDescriptor(miopenDataType_t t, std::initializer_list<std::size_t> plens);
    TensorDescriptor(miopenDataType_t t,
                     std::initializer_list<std::size_t> plens,
                     std::initializer_list<std::size_t> pstrides);
    TensorDescriptor(miopenDataType_t t, const int* plens, int size);
    TensorDescriptor(miopenDataType_t t, const int* plens, const int* pstrides, int size);
    template <class Range>
    TensorDescriptor(miopenDataType_t t, const Range& plens)
        : lens(plens.begin(), plens.end()), packed(true), type(t)
    {
        this->CalculateStrides();
    }

    template <class Range1, class Range2, class = decltype(std::declval<Range1>().begin())>
    TensorDescriptor(miopenDataType_t t, const Range1& plens, const Range2& pstrides)
        : lens(plens.begin(), plens.end()), strides(pstrides.begin(), pstrides.end()), type(t)
    {
        packed = (this->GetElementSize() == this->GetElementSpace());
    }

    void CalculateStrides();

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;
    int GetSize() const;

    miopenDataType_t GetType() const;

    std::size_t GetElementSize() const;

    std::size_t GetElementSpace() const;

    std::size_t GetNumBytes() const;

    std::size_t GetIndex(std::initializer_list<int> l) const;

    template <class... Ts>
    std::size_t GetIndex(Ts... is) const
    {
        return this->GetIndex({static_cast<int>(is)...});
    }

    bool IsPacked() const;

    TensorDescriptor GetFlattenedTensorDescriptor() const;

    bool operator==(const TensorDescriptor& rhs) const;
    bool operator!=(const TensorDescriptor& rhs) const;
    bool operator<(const TensorDescriptor& rhs) const;
    bool operator>(const TensorDescriptor& rhs) const;

    std::string ToString() const;

    friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    private:
    std::vector<std::size_t> lens;
    std::vector<std::size_t> strides;

    bool packed;

    miopenDataType_t type = miopenFloat;
};

template <typename... TDescriptors>
std::tuple<TDescriptors...>
get_consistent_flattened_tensor_descriptors(const TDescriptors&... real_descriptor_pack)
{
    constexpr std::size_t NTensor = sizeof...(TDescriptors);

    std::array<const TensorDescriptor*, NTensor> real_descriptors{{(&real_descriptor_pack)...}};
    std::array<TensorDescriptor, NTensor> flat_descriptors;

#if 1
    if(NTensor == 0)
        MIOPEN_THROW(miopenStatusBadParm, "NTensor == 0.");

    // check input tensor descriptors consistency
    const auto& real_desc_0_lens = real_descriptors[0]->GetLengths();

    for(std::size_t itensor = 1; itensor < NTensor; ++itensor)
    {
        auto& real_desc_lens = real_descriptors[itensor]->GetLengths();

        if(real_desc_0_lens != real_desc_lens)
            MIOPEN_THROW(miopenStatusBadParm, "Lengths of Tensors are different.");
    }
#endif

    // is all packed
    if(std::all_of(real_descriptors.begin(), real_descriptors.end(), [](const TensorDescriptor* desc) { return desc->IsPacked(); }))
    {
        std::size_t element_size = real_descriptors[0]->GetElementSize();

        for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
            flat_descriptors[itensor] =
                TensorDescriptor{real_descriptors[itensor]->GetType(), {element_size}, {1}};

        return to_tuple(std::move(flat_descriptors)); // early return for all-packed tensors
    }

    //
    const std::vector<std::size_t>& real_lengths = real_descriptors[0]->GetLengths();

    // is scalar
    if(std::all_of(
           real_lengths.begin(), real_lengths.end(), [](const std::size_t v) { return v <= 1; }))
    {
        for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
            flat_descriptors[itensor] =
                TensorDescriptor{real_descriptors[itensor]->GetType(), {1}, {1}};

        return to_tuple(std::move(flat_descriptors)); // early return for all-scalar tensors
    }

#if 0
    for(std::size_t idim = 1; idim < non1_ndim; ++idim)
    {
        for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
            array_of_full_lengths[itensor][idim] =
                array_of_non1_strides[itensor][idim - 1] / array_of_non1_strides[itensor][idim];
    }
#endif

    // start flattening tensors
    std::vector<std::size_t> flat_lengths;
    std::array<std::vector<std::size_t>, NTensor> array_of_flat_strides;

#if 0
    for(; i_non1 != i_non1_end; i_non1++)
    // the 0-th dimension full-length doesn't matter
    {
        std::size_t len = non1_lengths[idim];

        auto f_is_full_length = [&](const std::vector<std::size_t>& full_lengths) -> bool {
            return len == full_lengths[idim];
        };

        if(std::all_of(
               array_of_full_lengths.begin(), array_of_full_lengths.end(), f_is_full_length))
        {
            flat_len *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);

            for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
                array_of_flat_strides[itensor].push_back(array_of_non1_strides[itensor][idim - 1]);

            flat_len = len;
        }
    }

    flat_lengths.push_back(flat_len);
    for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
        array_of_flat_strides[itensor].push_back(array_of_non1_strides[itensor][non1_ndim - 1]);
#else
    using ZipIterator = decltype(
        boost::make_zip_iterator(boost::make_tuple(real_lengths.begin(), (real_descriptor_pack.GetStrides().begin())...)));
    struct is_non1_length_t
    {
        bool operator()(ZipIterator::reference v) { return v.get<0>() > 1; }
    };
    using FilterZipIterator = boost::filter_iterator<is_non1_length_t, ZipIterator>;

    ZipIterator i_begin =
        boost::make_zip_iterator(boost::make_tuple(real_lengths.begin(), (real_descriptor_pack.GetStrides().begin())...));
    ZipIterator i_end =
        boost::make_zip_iterator(boost::make_tuple(real_lengths.end(), (real_descriptor_pack.GetStrides().end())...));

    FilterZipIterator i_non1     = boost::make_filter_iterator(is_non1_length_t{}, i_begin, i_end);
    FilterZipIterator i_non1_end = boost::make_filter_iterator(is_non1_length_t{}, i_end, i_end);

    std::size_t flat_len = i_non1->get<0>();

    FilterZipIterator i_non1_previous = i_non1++;

    for(; i_non1 != i_non1_end; i_non1++)
    // the 0-th dimension full-length doesn't matter
    {
        std::size_t len = i_non1->get<0>();

        bool is_all_full_length = true;
        call_n_time([&](std::size_t itensor) {  
                std::size_t full_len == i_non1_previous->get<itensor+1>() / i_non1->get<itensor+1>();
                if(len == full_len)
                    is_all_full_length = false;
                    },
                NTensor);

        if(is_all_full_length)
        {
            flat_lent *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);
            call_n_time([&](std::size_t itensor) {
                    array_of_flat_strides[itensor].push_back(i_non1_previous->get<itensor+1>());
                    },
                    );
            flat_len *= len;
        }
        flat_lenghts.push_back(flat_len);
        call_n_time([&](std::size_t itensor) {
                array_of_flat_strides[itensor].push_back(i_non1_previous->get<itensor+1>());
                },
                );
    }

#endif

    for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
    {
        flat_descriptors[itensor] = TensorDescriptor{
            real_descriptors[itensor]->GetType(), flat_lengths, array_of_flat_strides[itensor]};
    }

    return to_tuple(std::move(flat_descriptors));
}

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenTensorDescriptor, miopen::TensorDescriptor)

#endif // GUARD_MIOPEN_TENSOR_HPP_
