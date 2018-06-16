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
#include <boost/range/combine.hpp>
#include <boost/range/adaptor/filtered.hpp>
// TODO(paul): remove this include later
#include <cstdio>

namespace miopen {
extern bool debug_tensor_descriptor;
} // namespace miopen

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
    TensorDescriptor(TensorDescriptor&& other)
        : lens(std::move(other.lens)), strides(std::move(other.strides)), type(other.type)
    {
        if(debug_tensor_descriptor)
            std::cout << __func__ << ": TD move constructor" << std::endl;

        packed = (this->GetElementSize() == this->GetElementSpace());
    }
    TensorDescriptor(const TensorDescriptor& other)
        : lens(other.lens), strides(other.strides), type(other.type)
    {
        if(debug_tensor_descriptor)
            std::cout << __func__ << ": TD copy constructor" << std::endl;
        packed = (this->GetElementSize() == this->GetElementSpace());
    }
    TensorDescriptor(miopenDataType_t t, std::initializer_list<std::size_t> plens);
    TensorDescriptor(miopenDataType_t t,
                     std::initializer_list<std::size_t> plens,
                     std::initializer_list<std::size_t> pstrides);
    TensorDescriptor(miopenDataType_t t, const int* plens, int size);
    TensorDescriptor(miopenDataType_t t, const int* plens, const int* pstrides, int size);

    TensorDescriptor(miopenDataType_t t, std::vector<std::size_t>&& lens_in, std::vector<std::size_t>&& strides_in);

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
        if(debug_tensor_descriptor)
            std::cout << __func__ << ": TD custom constructor (range)" << std::endl;
        packed = (this->GetElementSize() == this->GetElementSpace());
    }

    TensorDescriptor& operator= (const TensorDescriptor& other)
    {
        if(debug_tensor_descriptor)
            std::cout << __func__ << ": TD copy reference assignment" << std::endl;
        lens = other.lens;
        strides = other.strides;
        type = other.type;
        packed = (this->GetElementSize() == this->GetElementSpace());
        return *this;
    }
    TensorDescriptor& operator= (TensorDescriptor&& other)
    {
        if(debug_tensor_descriptor)
            std::cout << __func__ << ": TD move assignment" << std::endl;

        lens = std::move(other.lens);
        strides = std::move(other.strides);
        type = other.type;
        packed = (this->GetElementSize() == this->GetElementSpace());
        return *this;
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

struct f_length_is_not_1_t
{
    template <typename T>
    bool operator()(T&& v)
    {
        return boost::get<0>(v) > 1;
    }
};

template <typename... TDescriptors>
std::tuple<TDescriptors...>
get_consistent_flattened_tensor_descriptors(const TDescriptors&... real_descriptor_pack)
{
    std::tuple<TDescriptors...> flat_descriptors;

    constexpr std::size_t NTensor = sizeof...(TDescriptors);
    std::integral_constant<std::size_t, NTensor> NTensorConstant;

    std::array<const TensorDescriptor*, NTensor> real_descriptors{{(&real_descriptor_pack)...}};

#if 0
    // check input tensor descriptors consistency
    const auto& real_desc_0_lens = real_descriptors[0]->GetLengths();

    for(std::size_t itensor = 1; itensor < NTensor; ++itensor)
    {
        auto& real_desc_lens = real_descriptors[itensor]->GetLengths();

        if(real_desc_0_lens != real_desc_lens)
            MIOPEN_THROW(miopenStatusBadParm, "Lengths of Tensors are different.");
    }
#endif

    // start flattening tensors
    std::array<std::vector<std::size_t>, NTensor> array_of_flat_lengths;
    std::array<std::vector<std::size_t>, NTensor> array_of_flat_strides;

    auto non1_length_strides =
        boost::combine(real_descriptors[0]->GetLengths(), real_descriptor_pack.GetStrides()...) |
        boost::adaptors::filtered(f_length_is_not_1_t());

    auto i               = non1_length_strides.begin();
    std::size_t flat_len = boost::get<0>(*i);
    auto i_previous      = i++;

    for(; i != non1_length_strides.end(); ++i)
    // the 0-th dimension full-length doesn't matter
    {
        std::size_t len = boost::get<0>(*i);

        bool is_all_full_length = true;
        call_n_time(
            [&](auto itensor) {
                std::size_t stride          = boost::get<itensor + 1>(*i);
                std::size_t previous_stride = boost::get<itensor + 1>(*i_previous);
                std::size_t full_len        = previous_stride / stride;
                if(len != full_len)
                    is_all_full_length = false;
            },
            NTensorConstant);

        if(is_all_full_length)
        {
            flat_len *= len;
        }
        else
        {
            array_of_flat_lengths[0].push_back(flat_len);

            call_n_time(
                [&](auto itensor) {
                    std::size_t previous_stride = boost::get<itensor + 1>(*i_previous);
                    array_of_flat_strides[itensor].push_back(previous_stride);
                },
                NTensorConstant);
            flat_len = len;
        }
        i_previous = i;
    }
    array_of_flat_lengths[0].push_back(flat_len);

    call_n_time(
        [&](auto itensor) {
            std::size_t previous_stride = boost::get<itensor + 1>(*i_previous);
            array_of_flat_strides[itensor].push_back(previous_stride);
        },
        NTensorConstant);

    for(std::size_t itensor            = 1; itensor < NTensor; ++itensor)
        array_of_flat_lengths[itensor] = array_of_flat_lengths[0];

    call_n_time(
        [&](auto itensor) {
            std::get<itensor>(flat_descriptors) = TensorDescriptor{real_descriptors[itensor]->GetType(),
                                                  std::move(array_of_flat_lengths[itensor]),
                                                  std::move(array_of_flat_strides[itensor])};
        },
        NTensorConstant);

    return flat_descriptors;
}

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenTensorDescriptor, miopen::TensorDescriptor)

#endif // GUARD_MIOPEN_TENSOR_HPP_
