#ifndef CK_TYPE_HELPER_HPP
#define CK_TYPE_HELPER_HPP

#include "float_type.hpp"

namespace ck {

template <char tid>
struct get_type_from_type_id
{
    using type = float;
};

template <>
struct get_type_from_type_id<'H'>
{
    using type = half_t;
};

template <>
struct get_type_from_type_id<'F'>
{
    using type = float;
};

template <>
struct get_type_from_type_id<'D'>
{
    using type = double;
};

} // namespace ck

#endif
