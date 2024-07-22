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

#include <miopen/miopen.h>

#include <miopen/common.hpp>
#include <miopen/each_args.hpp>
#include <miopen/errors.hpp>
#include <miopen/functional.hpp>
#include <miopen/object.hpp>
#include <miopen/returns.hpp>

#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <optional>

namespace miopen {

template <class T, std::size_t... Ns>
auto tie_impl(T&& x, detail::seq<Ns...>) -> decltype(std::tie(x[Ns]...))
{
    assert(x.size() >= sizeof...(Ns));
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
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8: return 1;
    case miopenDouble:
    case miopenInt64: return 8;
    }
    MIOPEN_THROW("Unknown or unsupported data type");
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

struct MIOPEN_INTERNALS_EXPORT TensorDescriptor : miopenTensorDescriptor
{
    TensorDescriptor();

    // This constructor is only used in test/tensor_holder.hpp
    // clang-format off
    [[deprecated("Use constructor with lengths instead")]]
    TensorDescriptor(miopenDataType_t t);
    // clang-format on

    // It is preferable to use constructors with lengths and strides with the std::size_t
    // data type, because in this format the data is stored inside the class

    // The delegation constructor should be placed above the target constructor in the
    // code for better dependency tracking

    TensorDescriptor(miopenDataType_t t, const std::initializer_list<int>& lens_in);
    TensorDescriptor(miopenDataType_t t, const std::vector<int>& lens_in);
    TensorDescriptor(miopenDataType_t t, const std::initializer_list<std::size_t>& lens_in);
    TensorDescriptor(miopenDataType_t t, const std::vector<std::size_t>& lens_in);
    TensorDescriptor(miopenDataType_t t, std::vector<std::size_t>&& lens_in);

    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     const std::vector<int>& lens_in);
    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     const std::initializer_list<std::size_t>& lens_in);
    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     const std::vector<std::size_t>& lens_in);
    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     std::vector<std::size_t>&& lens_in);

    TensorDescriptor(miopenDataType_t t,
                     const std::vector<int>& lens_in,
                     const std::vector<int>& strides_in);
    TensorDescriptor(miopenDataType_t t,
                     const std::initializer_list<std::size_t>& lens_in,
                     const std::initializer_list<std::size_t>& strides_in);
    TensorDescriptor(miopenDataType_t t,
                     const std::vector<std::size_t>& lens_in,
                     const std::vector<std::size_t>& strides_in);
    TensorDescriptor(miopenDataType_t t,
                     std::vector<std::size_t>&& lens_in,
                     std::vector<std::size_t>&& strides_in);

    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     const std::vector<std::size_t>& lens_in,
                     const std::vector<std::size_t>& strides_in);
    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     std::vector<std::size_t>&& lens_in,
                     std::vector<std::size_t>&& strides_in);

    // Use only for external API
    static TensorDescriptor MakeDescriptor(miopenDataType_t t, const int* plens, int size);
    static TensorDescriptor MakeDescriptor(miopenDataType_t t, const std::size_t* plens, int size);
    static TensorDescriptor
    MakeDescriptor(miopenDataType_t t, miopenTensorLayout_t layout, const int* plens, int size);
    static TensorDescriptor MakeDescriptor(miopenDataType_t t,
                                           miopenTensorLayout_t layout,
                                           const std::size_t* plens,
                                           int size);
    static TensorDescriptor
    MakeDescriptor(miopenDataType_t t, const int* plens, const int* pstrides, int size);
    static TensorDescriptor MakeDescriptor(miopenDataType_t t,
                                           const std::size_t* plens,
                                           const std::size_t* pstrides,
                                           int size);

    bool IsVectorized() const;

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;
    unsigned GetNumDims() const;

    miopenDataType_t GetType() const;
    miopenTensorLayout_t GetLayout_t() const;
    static std::string GetLayoutStr(miopenTensorLayout_t layout);
    std::string GetLayout_str() const;

    std::size_t GetVectorLength() const;
    std::optional<miopenDataType_t> GetCastType() const;
    void SetCastType(miopenDataType_t cast_type_);

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
    bool IsContiguous() const;
    /// Checks all lengths and strides.
    bool AllDimsFitIntoInt() const;
    /// Checks only lengths.
    bool AllLengthsFitIntoInt() const;

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
        std::stable_sort(result.begin(), result.end(), by(std::greater<>{}, [&](auto x) {
                             return std::make_tuple(strides[x], lens[x]);
                         }));
        return result;
    }

    std::string GetLayout(std::string labels) const
    {
        if(*(labels.end() - 1) != 'c')
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
        else
        {
            const std::string base_label = labels.substr(0, labels.size() - 1);
            if(base_label.size() != strides.size())
            {
                MIOPEN_THROW(
                    "Invalid labels size. Layout labels size must be equavalent to stride size");
            }
            auto result = base_label;
            auto p      = find_permutation(lens, strides);
            std::transform(p.begin(), p.end(), result.begin(), [&](auto i) { return labels[i]; });
            return result + 'c';
        }
    }

    friend MIOPEN_INTERNALS_EXPORT std::ostream& operator<<(std::ostream& stream,
                                                            const TensorDescriptor& t);

    friend void to_json(nlohmann::json& j, const TensorDescriptor& descriptor);
    friend void from_json(const nlohmann::json& j, TensorDescriptor& descriptor);

protected:
    static miopenTensorLayout_t GetDefaultLayout() { return miopenTensorNCHW; };

private:
    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     const std::vector<std::size_t>& lens_in,
                     const std::vector<std::size_t>& strides_in,
                     bool use_strides);

    TensorDescriptor(miopenDataType_t t,
                     miopenTensorLayout_t layout_in,
                     std::vector<std::size_t>&& lens_in,
                     std::vector<std::size_t>&& strides_in,
                     bool use_strides);

    void CheckArgsAndInit(bool use_strides);

    void SetStrideNd(const std::string& layout);
    void LensReorder(const std::string& layout);

    void CalculateStrides();
    void CalculateVectorLength();

    std::vector<std::size_t> lens;
    std::vector<std::size_t> strides;

    bool packed;
    std::size_t vector_length = 1;

    miopenDataType_t type = miopenFloat;
    std::optional<miopenDataType_t> cast_type;
    miopenTensorLayout_t tensorLayout = GetDefaultLayout();
};

template <class TElement>
constexpr auto GetNCDHW(unsigned spatial_dims, const std::vector<TElement>& data)
{
    if(spatial_dims == 3)
        return miopen::tien<5>(data, 1);
    else
        return std::make_tuple(data[0], data[1], static_cast<TElement>(1), data[2], data[3]);
}

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenTensorDescriptor, miopen::TensorDescriptor)

#endif // GUARD_MIOPEN_TENSOR_HPP_
