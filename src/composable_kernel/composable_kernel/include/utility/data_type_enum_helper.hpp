#ifndef CK_DATA_TYPE_ENUM_HELPER_HPP
#define CK_DATA_TYPE_ENUM_HELPER_HPP

#include "data_type.hpp"
#include "data_type_enum.hpp"

namespace ck {

template <DataTypeEnum_t DataTypeEnum>
struct get_datatype_from_enum;

template <>
struct get_datatype_from_enum<DataTypeEnum_t::Int8>
{
    using type = int8_t;
};

template <>
struct get_datatype_from_enum<DataTypeEnum_t::Int32>
{
    using type = int32_t;
};

template <>
struct get_datatype_from_enum<DataTypeEnum_t::Half>
{
    using type = half_t;
};

template <>
struct get_datatype_from_enum<DataTypeEnum_t::Float>
{
    using type = float;
};

template <>
struct get_datatype_from_enum<DataTypeEnum_t::Double>
{
    using type = double;
};

template <typename T>
struct get_datatype_enum_from_type;

template <>
struct get_datatype_enum_from_type<int8_t>
{
    static constexpr DataTypeEnum_t value = DataTypeEnum_t::Int8;
};

template <>
struct get_datatype_enum_from_type<int32_t>
{
    static constexpr DataTypeEnum_t value = DataTypeEnum_t::Int32;
};

template <>
struct get_datatype_enum_from_type<half_t>
{
    static constexpr DataTypeEnum_t value = DataTypeEnum_t::Half;
};

template <>
struct get_datatype_enum_from_type<float>
{
    static constexpr DataTypeEnum_t value = DataTypeEnum_t::Float;
};

template <>
struct get_datatype_enum_from_type<double>
{
    static constexpr DataTypeEnum_t value = DataTypeEnum_t::Double;
};

} // namespace ck
#endif
