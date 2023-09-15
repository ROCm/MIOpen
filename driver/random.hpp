#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <random>
#if HIP_PACKAGE_VERSION_FLAT >= 5006000000ULL
#include <half/half.hpp>
#else
#include <half.hpp>
#endif

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

template <typename T, bool Signed = false>
inline T RAN_SUBNORM()
{
    T denorm_val = static_cast<T>(0);
    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, half_float::half>)
    {
        using BitType = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
        static_assert(sizeof(T) == sizeof(BitType));

        // -1 because ::digits counts the first implicit digit
        static constexpr auto mantissa_bits = std::numeric_limits<T>::digits - 1;
        static constexpr auto mantissa_mask = (1 << mantissa_bits) - 1;

        BitType denorm_bits = GET_RAND() & mantissa_mask;
        denorm_bits |= Signed ? ((GET_RAND() % 2) << (sizeof(T) * 8 - 1)) : 0;

        // the proper way to do a type punning
        std::memcpy(&denorm_val, &denorm_bits, sizeof(T));
    }
    return denorm_val;
}
#endif // GUARD_RANDOM_GEN_
