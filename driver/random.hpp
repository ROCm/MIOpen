#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <random>
#include <miopen/bfloat16.hpp>
#include <half.hpp>

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline T FRAND()
{
    static const int seed = 44619;
    static std::mt19937 rng(seed);
    std::uniform_int_distribution<T> uniform_dist;

    return uniform_dist(rng);
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline T FRAND()
{
    static const int seed = 44619;
    static std::mt19937 rng(seed);
    std::uniform_real_distribution<T> uniform_dist;

    return uniform_dist(rng);
}

inline int GET_RAND() { return FRAND<int>(); }

template <typename T>
inline T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}

template <>
inline half_float::half RAN_GEN(half_float::half A, half_float::half B)
{
    return static_cast<half_float::half>(
        RAN_GEN<float>(static_cast<float>(A), static_cast<float>(B)));
}

template <>
inline bfloat16 RAN_GEN(bfloat16 A, bfloat16 B)
{
    return static_cast<bfloat16>(RAN_GEN<float>(static_cast<float>(A), static_cast<float>(B)));
}
#endif // GUARD_RANDOM_GEN_
