/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
