#ifndef CK_ARRAY_HPP
#define CK_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"

#ifdef __HIPCC_RTC__
#ifdef WORKAROUND_ISSUE_HIPRTC_TRUE_TYPE
/// We need <utility> for std::forward. In some cases, it includes <type_traits>
/// (this is against the Standard, but it doesn't matter in this case).
/// But <type_traits> also defines std::true_type, per Standard.
/// However the latter definition conflicts with
/// /opt/rocm/include/hip/amd_detail/amd_hip_vector_types.h,
/// which defines std::true_type as well (which is wrong).

namespace std {

template <typename T>
constexpr T&& forward(typename remove_reference<T>::type& t_) noexcept
{
    return static_cast<T&&>(t_);
}

template <typename T>
constexpr T&& forward(typename remove_reference<T>::type&& t_) noexcept
{
    return static_cast<T&&>(t_);
}

} // namespace std
#else
#include <utility> // std::forward
#endif
#endif // __HIPCC_RTC__

namespace ck {

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array;
    using data_type = TData;

    TData mData[NSize];

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    __host__ __device__ constexpr const TData& At(index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& At(index_t i) { return mData[i]; }

    __host__ __device__ constexpr const TData& operator[](index_t i) const { return At(i); }

    __host__ __device__ constexpr TData& operator()(index_t i) { return At(i); }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }
};

// empty Array
template <typename TData>
struct Array<TData, 0>
{
    using type      = Array;
    using data_type = TData;

    __host__ __device__ static constexpr index_t Size() { return 0; }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_array(X&& x, Xs&&... xs)
{
    using data_type = remove_cv_t<remove_reference_t<X>>;
    return Array<data_type, sizeof...(Xs) + 1>{{std::forward<X>(x), std::forward<Xs>(xs)...}};
}

// make empty array
template <typename X>
__host__ __device__ constexpr auto make_array()
{
    return Array<X, 0>{};
}

} // namespace ck
#endif
