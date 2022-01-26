#ifndef CK_UTILS_TYPE_HPP
#define CK_UTILS_TYPE_HPP

#include "static_kernel_integral_constant.hpp"

#ifdef __HIPCC_RTC__
#ifdef WORKAROUND_ISSUE_HIPRTC_TRUE_TYPE
/// We need <type_traits> for std::remove_reference and std::remove_cv.
/// But <type_traits> also defines std::true_type, per Standard.
/// However the latter definition conflicts with
/// /opt/rocm/include/hip/amd_detail/amd_hip_vector_types.h,
/// which defines std::true_type as well (which is wrong).

namespace std {

template <class T>
struct remove_reference
{
    typedef T type;
};
template <class T>
struct remove_reference<T&>
{
    typedef T type;
};
template <class T>
struct remove_reference<T&&>
{
    typedef T type;
};

template <class T>
struct remove_const
{
    typedef T type;
};
template <class T>
struct remove_const<const T>
{
    typedef T type;
};

template <class T>
struct remove_volatile
{
    typedef T type;
};
template <class T>
struct remove_volatile<volatile T>
{
    typedef T type;
};

template <class T>
struct remove_cv
{
    typedef typename remove_volatile<typename remove_const<T>::type>::type type;
};

} // namespace std
#else
#include <type_traits> // std::remove_reference, std::remove_cv
#endif
#endif // __HIPCC_RTC__

namespace ck {

template <index_t... Is>
struct Sequence;

template <typename X, typename Y>
struct is_same : public integral_constant<bool, false>
{
};

template <typename X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <typename>
struct is_static : integral_constant<bool, false>
{
};

template <typename T, T X>
struct is_static<integral_constant<T, X>> : integral_constant<bool, true>
{
};

template <index_t... Is>
struct is_static<Sequence<Is...>> : integral_constant<bool, true>
{
};

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

} // namespace ck
#endif
