#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <random>
#include <cstdlib>
#include <cassert>
#include <cstdint>
#if HIP_PACKAGE_VERSION_FLAT >= 5006000000ULL
#include <half/half.hpp>
#else
#include <half.hpp>
#endif
using float16 = half_float::half;

std::minstd_rand& get_minstd_gen()
{
    static thread_local std::minstd_rand minstd_gen(std::random_device{}());
    return minstd_gen;
}

template <typename T>
inline T FRAND()
{
    double d = std::generate_canonical<double, 1>(get_minstd_gen());
    return static_cast<T>(d);
}

inline int GET_RAND()
{
    decltype(auto) minstd_gen = get_minstd_gen();
    return minstd_gen();
}

template <typename T>
inline T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}

template <typename T>
inline T RAN_SUBNORM(int fraction_msb_bit = 0, int fraction_lsb_bit = 0)
{
    (void)fraction_msb_bit;
    (void)fraction_lsb_bit;
    return static_cast<T>(0);
}

#define SUBNORM_INITIALIZE_FRACTION(msb, lsb, msb_default, lsb_default) \
    do                                                                  \
    {                                                                   \
        if((msb) == 0 && (lsb) == 0)                                    \
        {                                                               \
            (msb) = (msb_default);                                      \
            (lsb) = (lsb_default);                                      \
        }                                                               \
        assert((msb) <= (msb_default) && (msb) >= (lsb_default));       \
        assert((lsb) <= (msb_default) && (lsb) >= (lsb_default));       \
        assert((lsb) <= (msb));                                         \
    } while(0)

#define SUBNORM_RANDOM_FRACTION(result, msb, lsb)        \
    do                                                   \
    {                                                    \
        int _bits = (msb) - (lsb);                       \
        if(_bits == 0)                                   \
            (result) = 0;                                \
        else                                             \
            (result) = (rand() % (1 << _bits)) << (lsb); \
    } while(0)

template <>
inline float RAN_SUBNORM(int fraction_msb_bit, int fraction_lsb_bit)
{
    uint32_t float32_value;
    SUBNORM_INITIALIZE_FRACTION(fraction_msb_bit, fraction_lsb_bit, 22, 0);
    SUBNORM_RANDOM_FRACTION(float32_value, fraction_msb_bit, fraction_lsb_bit);
    return *(reinterpret_cast<float*>(&float32_value));
}

template <>
inline float16 RAN_SUBNORM(int fraction_msb_bit, int fraction_lsb_bit)
{
    uint16_t float16_value;
    SUBNORM_INITIALIZE_FRACTION(fraction_msb_bit, fraction_lsb_bit, 9, 0);
    SUBNORM_RANDOM_FRACTION(float16_value, fraction_msb_bit, fraction_lsb_bit);
    return *(reinterpret_cast<float16*>(&float16_value));
}

#endif // GUARD_RANDOM_GEN_
