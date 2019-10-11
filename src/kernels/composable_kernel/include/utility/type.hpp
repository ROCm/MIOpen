#ifndef CK_TYPE_HPP
#define CK_TYPE_HPP

#include "integral_constant.hpp"

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
