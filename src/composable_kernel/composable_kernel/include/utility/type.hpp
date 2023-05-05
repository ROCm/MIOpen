#ifndef CK_TYPE_HPP
#define CK_TYPE_HPP

#include "integral_constant.hpp"
#include "enable_if.hpp"

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
using remove_reference_t = typename remove_reference<T>::type;

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

template <class T>
struct is_pointer_helper : std::false_type
{
};

template <class T>
struct is_pointer_helper<T*> : std::true_type
{
};

template <class T>
struct is_pointer : is_pointer_helper<typename std::remove_cv<T>::type>
{
};

} // namespace std
#else
#include <type_traits> // std::remove_reference, std::remove_cv, is_pointer
#endif
#endif // __HIPCC_RTC__

namespace ck {

template <typename X, typename Y>
struct is_same : public integral_constant<bool, false>
{
};

template <typename X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer<T>::value;

template <typename T>
struct is_known_at_compile_time;

template <>
struct is_known_at_compile_time<index_t>
{
    static constexpr bool value = false;
};

template <typename T, T X>
struct is_known_at_compile_time<integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <typename Y, typename X, typename enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y as_type(X x)
{
    union AsType
    {
        X x;
        Y y;
    };

    return AsType{x}.y;
}

} // namespace ck
#endif
