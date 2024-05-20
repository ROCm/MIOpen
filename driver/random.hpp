#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <miopen/env.hpp>

#include <cassert>
#include <iostream>
#include <random>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_DRIVER_PRNG_SEED, uint64_t, 12345678)
namespace prng {
namespace details {
using glibc_gen = std::linear_congruential_engine<std::uint32_t, 1103515245, 12345, 2147483648>;

inline std::random_device::result_type get_default_seed()
{
    static std::random_device::result_type seed{[] {
        auto external_seed = miopen::Value(ENV(MIOPEN_DEBUG_DRIVER_PRNG_SEED));

        auto seed = external_seed == 0
                        ? std::random_device{}()
                        : static_cast<std::random_device::result_type>(external_seed);
        std::cout << "PRNG seed: " << seed << "\n";
        return seed;
    }()};
    return seed;
}

inline glibc_gen& get_prng()
{
    static thread_local glibc_gen gen{get_default_seed()};
    return gen;
}

template <class, class = void>
struct has_digits : std::false_type
{
};

template <class T>
struct has_digits<T, std::void_t<decltype(std::numeric_limits<T>::digits)>> : std::true_type
{
};

} // namespace details

inline void reset_seed(std::random_device::result_type seed = 0)
{
    details::get_prng().seed(seed + details::get_default_seed());
}

// similar to std::generate_canonical, but simpler and faster
template <typename T>
inline T gen_canonical()
{
    if constexpr(std::is_floating_point_v<T>) // native fp
    {
        static constexpr T range =
            static_cast<T>(1) /
            static_cast<T>(details::glibc_gen::max() - details::glibc_gen::min() + 1);
        return range * static_cast<T>(details::get_prng()() - details::glibc_gen::min());
    }
    else if constexpr(std::is_integral_v<T>)
    {
        auto val = details::get_prng()();
        return static_cast<T>(((val >> 4) + (val >> 16)) & 0x1);
    }
    else
    {
        return static_cast<T>(gen_canonical<float>());
    }
}

template <typename T>
inline T gen_0_to_B(T B)
{
    if constexpr(std::is_floating_point_v<T>) // native fp
    {
        return gen_canonical<T>() * B;
    }
    else if constexpr(std::is_integral_v<T>)
    {
        // can only generate 27bit range, so it may not be suitable
        // for huge 64 bit ranges, but we do not expect such ranges
        return static_cast<T>((details::get_prng()() >> 4) % B);
    }
    else // half/bfloat/etc
    {
        return static_cast<T>(gen_0_to_B(static_cast<float>(B)));
    }
}

template <typename T>
inline T gen_A_to_B(T A, T B)
{
    assert(B > A);
    return gen_0_to_B(B - A) + A;
}

template <typename T>
inline T gen_off_range(T offset, T range)
{
    static_assert(std::is_integral_v<T>);
    return prng::gen_0_to_B(range) + offset;
}

template <typename T, bool Signed = false>
inline T gen_subnorm()
{
    T denorm_val = static_cast<T>(0);
    if constexpr(!std::is_integral_v<T> && !std::is_same_v<T, double> &&
                 details::has_digits<T>::value)
    {
        using BitType = std::conditional_t<sizeof(T) == 1,
                                           uint8_t,
                                           std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;
        static_assert(sizeof(T) == sizeof(BitType));

        // -1 because ::digits counts the first implicit digit
        static constexpr auto mantissa_bits = std::numeric_limits<T>::digits - 1;

        BitType denorm_bits = static_cast<BitType>(gen_0_to_B(1 << mantissa_bits));
        denorm_bits |= Signed ? (gen_canonical<BitType>() << (sizeof(T) * 8 - 1)) : 0;

        // the proper way to do a type punning
        std::memcpy(&denorm_val, &denorm_bits, sizeof(T));
    }
    return denorm_val;
}
} // namespace prng
#endif // GUARD_RANDOM_GEN_
