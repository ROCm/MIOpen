#ifndef GUARD_MLOPEN_LOGGER_HPP
#define GUARD_MLOPEN_LOGGER_HPP

#include <iostream>
#include <array>
#include <type_traits>
#include <mlopen/each_args.hpp>


// See https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
#define MLOPEN_PP_CAT(x, y) MLOPEN_PP_PRIMITIVE_CAT(x, y)
#define MLOPEN_PP_PRIMITIVE_CAT(x, y) x ## y

#define MLOPEN_PP_IIF(c) MLOPEN_PP_PRIMITIVE_CAT(MLOPEN_PP_IIF_, c)
#define MLOPEN_PP_IIF_0(t, ...) __VA_ARGS__
#define MLOPEN_PP_IIF_1(t, ...) t

#define MLOPEN_PP_IS_PAREN(x) MLOPEN_PP_IS_PAREN_CHECK(MLOPEN_PP_IS_PAREN_PROBE x)
#define MLOPEN_PP_IS_PAREN_CHECK(...) MLOPEN_PP_IS_PAREN_CHECK_N(__VA_ARGS__,0)
#define MLOPEN_PP_IS_PAREN_PROBE(...) ~, 1,
#define MLOPEN_PP_IS_PAREN_CHECK_N(x, n, ...) n

#define MLOPEN_PP_EAT(...)
#define MLOPEN_PP_EXPAND(...) __VA_ARGS__
#define MLOPEN_PP_COMMA(...) ,

#define MLOPEN_PP_TRANSFORM_ARGS(m, ...) \
MLOPEN_PP_EXPAND(MLOPEN_PP_PRIMITIVE_TRANSFORM_ARGS(m, MLOPEN_PP_COMMA, __VA_ARGS__,(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()))

#define MLOPEN_PP_EACH_ARGS(m, ...) \
MLOPEN_PP_EXPAND(MLOPEN_PP_PRIMITIVE_TRANSFORM_ARGS(m, MLOPEN_PP_EAT, __VA_ARGS__,(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()))

#define MLOPEN_PP_PRIMITIVE_TRANSFORM_ARGS(m, delim, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, ...) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x0)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x1) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x1)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x2) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x2)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x3) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x3)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x4) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x4)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x5) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x5)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x6) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x6)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x7) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x7)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x8) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x8)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x9) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x9)  MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x10) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x10) MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x11) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x11) MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x12) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x12) MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x13) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x13) MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x14) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x14) MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x15) \
MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x15)

#define MLOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x) \
MLOPEN_PP_IIF(MLOPEN_PP_IS_PAREN(x))(MLOPEN_PP_EAT, m)(x)

namespace mlopen {

template<class Range>
std::ostream& LogRange(std::ostream& os, Range&& r, std::string delim)
{
    bool first = true;
    for(auto&& x:r)
    {
        if (first) first = false;
        else os << delim;
        os << x;
    }
    return os;
}

template<class T, class... Ts>
std::array<T, sizeof...(Ts)+1> make_array(T x, Ts... xs)
{
    return {{x, xs...}};
}

#define MLOPEN_LOG_ENUM_EACH(x) std::pair<std::string, decltype(x)>(#x, x)
#define MLOPEN_LOG_ENUM(os, x, ...) mlopen::LogEnum(os, x, make_array(MLOPEN_PP_TRANSFORM_ARGS(MLOPEN_LOG_ENUM_EACH, __VA_ARGS__)))

template<class T, class Range>
std::ostream& LogEnum(std::ostream& os, T x, Range&& values)
{
    for(auto&& p:values) 
    {
        if (p.second == x)
        {
            os << p.first;
            return os;
        }
    }
    os << "Unknown: " << x;
    return os;
}

bool IsLogging();

template<class T>
auto LogObjImpl(T* x) -> decltype(get_object(*x))
{
    return get_object(*x);
}

inline void* LogObjImpl(void* x)
{
    return x;
}

inline const void* LogObjImpl(const void* x)
{
    return x;
}

template<class T, typename std::enable_if<(std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x)
{
    os << name << " = ";
    if(x == nullptr) os << "nullptr";
    else os << LogObjImpl(x);
    return os;
}

template<class T, typename std::enable_if<(not std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x)
{
    os << name << " = " << get_object(x);
    return os;
}

#define MLOPEN_LOG_FUNCTION_EACH(param) mlopen::LogParam(std::cerr, #param, param) << std::endl;

#define MLOPEN_LOG_FUNCTION(...) \
if (mlopen::IsLogging()) { \
    std::cerr << __PRETTY_FUNCTION__ << "{" << std::endl; \
    MLOPEN_PP_EACH_ARGS(MLOPEN_LOG_FUNCTION_EACH, __VA_ARGS__) \
    std::cerr << "}" << std::endl; \
}


} // namespace mlopen

#endif
