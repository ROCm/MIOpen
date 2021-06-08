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

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/each_args.hpp>
#include <miopen/returns.hpp>
#include <miopen/errors.hpp>
#include <miopen/functional.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

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

template <class T, std::size_t... Ns>
auto tie_pick_impl(T&& x, detail::seq<Ns...>)
{
#ifndef NDEBUG
    each_args([&](auto i) { assert(i < x.size()); }, Ns...);
#endif
    return std::tie(x[Ns]...);
}

template <std::size_t... Ns>
struct tie_pick
{
    template <class T>
    auto operator()(T&& x) MIOPEN_RETURNS(tie_pick_impl(std::forward<T>(x), detail::seq<Ns...>{}))
};

template <typename F, std::size_t... Ns>
auto create_tuple_impl(F f, detail::seq<Ns...>)
{
    return std::make_tuple(std::forward<decltype(f(Ns))>(f(Ns))...);
}

template <std::size_t N, typename F>
auto create_tuple(F f)
{
    return create_tuple_impl(f, typename detail::gens<N>::type{});
}

inline std::size_t GetTypeSize(miopenDataType_t d)
{
    switch(d)
    {
    case miopenInt32:
    case miopenFloat: return 4;
    case miopenHalf:
    case miopenBFloat16: return 2;
    case miopenInt8x4:
    case miopenInt8: return 1;
    case miopenDouble: return 8;
    }
    MIOPEN_THROW("Unknown data type");
}

template <class X, class Y>
std::ptrdiff_t integer_division_ceil(X x, Y y)
{
    std::ptrdiff_t tx = static_cast<std::ptrdiff_t>(x);
    std::ptrdiff_t ty = static_cast<std::ptrdiff_t>(y);

    if(ty < 1)
    {
        MIOPEN_THROW("integer_division_ceil: y < 1");
    }

    return (tx + ty - 1) / ty;
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

    TensorDescriptor(miopenDataType_t t,
                     std::vector<std::size_t> lens_in,
                     std::vector<std::size_t> strides_in);

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

    bool operator==(const TensorDescriptor& rhs) const;
    bool operator!=(const TensorDescriptor& rhs) const;
    bool operator<(const TensorDescriptor& rhs) const;
    bool operator>(const TensorDescriptor& rhs) const;

    std::string ToString() const;

    bool IsPossibleLayout(const std::string& labels, const std::string& layout) const;

    static inline std::vector<int64_t> find_permutation(const std::vector<std::size_t>& lens,
                                                        const std::vector<std::size_t>& strides)
    {
        std::vector<std::int64_t> result(lens.size());
        std::iota(result.begin(), result.end(), 0);
        std::stable_sort(
            result.begin(),
            result.end(),
            by(std::greater<>{}, [&](auto x) { return std::make_tuple(strides[x], lens[x]); }));
        return result;
    }

    std::string GetLayout(std::string labels) const
    {
        if(labels.size() != strides.size())
        {
            MIOPEN_THROW(
                "Invalid labels size. Layout labels size must be equavalent to stride size");
        }

        // Copy construct the result string from labels. This allocates the space at one go
        // and is faster than calling push_back in transform.
        auto result = labels;
        auto p      = find_permutation(lens, strides);
        std::transform(p.begin(), p.end(), result.begin(), [&](auto i) { return labels[i]; });
        return result;
    }

    friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    private:
    std::vector<std::size_t> lens;
    std::vector<std::size_t> strides;

    bool packed;

    miopenDataType_t type = miopenFloat;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenTensorDescriptor, miopen::TensorDescriptor)

#endif // GUARD_MIOPEN_TENSOR_HPP_
