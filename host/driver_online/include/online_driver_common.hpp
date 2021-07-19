#ifndef OLC_DRIVER_COMMON_HPP
#define OLC_DRIVER_COMMON_HPP

#include <half.hpp>
#include <vector>
#include <cassert>

// this enumerate should be synchronized with include/miopen.h
typedef enum {
    appHalf     = 0,
    appFloat    = 1,
    appInt32    = 2,
    appInt8     = 3,
    appInt8x4   = 4,
    appBFloat16 = 5,
    appDouble   = 6,
} appDataType_t;

namespace Driver {

template <appDataType_t typeNum>
struct get_type_from_type_enum
{
    using type = float;
};

template <>
struct get_type_from_type_enum<appHalf>
{
    using type = half_float::half;
};

template <>
struct get_type_from_type_enum<appFloat>
{
    using type = float;
};

template <>
struct get_type_from_type_enum<appDouble>
{
    using type = double;
};

template <>
struct get_type_from_type_enum<appInt32>
{
    using type = int;
};

static inline int get_typeid_from_type_enum(appDataType_t t)
{
    switch(t)
    {
    case appHalf: return (static_cast<int>('H'));
    case appFloat: return (static_cast<int>('F'));
    case appBFloat16: return (static_cast<int>('B'));
    case appDouble: return (static_cast<int>('D'));
    case appInt8:
    case appInt8x4:
    case appInt32: return (static_cast<int>('O'));
    default: throw std::runtime_error("Only float, half, bfloat16 data type is supported."); break;
    };
};

template <typename T>
static inline int get_typeid_from_type()
{
    throw std::runtime_error("Unsupported typeid conversion for this type!");
};

template <>
inline int get_typeid_from_type<float>()
{
    return (static_cast<int>('F'));
};

template <>
inline int get_typeid_from_type<half_float::half>()
{
    return (static_cast<int>('H'));
};

template <>
inline int get_typeid_from_type<double>()
{
    return (static_cast<int>('D'));
};

static inline float get_effective_average(std::vector<float>& values)
{
    assert(!values.empty());

    if(values.size() == 1)
        return (values[0]);
    else
    {
        float sum    = 0.0f;
        float maxVal = 0.0f;

        for(const auto val : values)
        {
            if(maxVal < val)
                maxVal = val;
            sum += val;
        };

        return ((sum - maxVal) / (values.size() - 1));
    };
};

} // namespace Driver

#endif
