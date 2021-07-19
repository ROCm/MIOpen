#ifndef CK_PRINT_HPP
#define CK_PRINT_HPP

#include "array.hpp"
#include "statically_indexed_array.hpp"
#include "container_helper.hpp"
#include "sequence.hpp"

namespace ck {

template <typename T>
__host__ __device__ void print_array(const char* s, T a)
{
    using data_type         = decltype(a.At(Number<0>{}));
    constexpr index_t nsize = a.Size();

#if 0
    if constexpr(is_same<data_type, uint32_t>{})
    {
        printf("%s size %u, {", s, nsize);
        static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("%u, ", uint32_t{a[i]}); });
        printf("}\n");
    }
    else if constexpr(is_same<data_type, int32_t>{})
    {
        printf("%s size %d, {", s, nsize);
        static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("%d, ", int32_t{a[i]}); });
        printf("}\n");
    }
    else if constexpr(is_same<data_type, bool>{})
    {
        printf("%s size %d, {", s, nsize);
        static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("%d, ", bool{a[i]}); });
        printf("}\n");
    }
#else
    printf("%s size %d, {", s, nsize);
    static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("%d, ", int32_t{a[i]}); });
    printf("}\n");
#endif
}

template <typename T>
__host__ __device__ void print_array_v2(const char* s, T a)
{
    using data_type         = decltype(a.At(Number<0>{}));
    constexpr index_t nsize = a.Size();

#if 0
    if constexpr(is_same<data_type, uint32_t>{})
    {
        printf("%s size %u, {", s, nsize);
        static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("[%u] %u, ", i.value, a[i]); });
        printf("}\n");
    }
    else if constexpr(is_same<data_type, int32_t>{})
    {
        printf("%s size %d, {", s, nsize);
        static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("[%d] %d, ", i.value, a[i]); });
        printf("}\n");
    }
#else
    printf("%s size %d, {", s, nsize);
    static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("[%d] %d, ", i.value, a[i]); });
    printf("}\n");
#endif
}

} // namespace ck
#endif
