#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <miopen/env.hpp>
#include <iostream>
#include <random>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_DRIVER_PRNG_SEED)
namespace prng {
namespace details {
using glibc_gen = std::linear_congruential_engine<std::uint32_t, 1103515245, 12345, 2147483648>;

inline glibc_gen& get_prng()
{
    static thread_local glibc_gen gen{[] {
        auto external_seed = miopen::Value(MIOPEN_DEBUG_DRIVER_PRNG_SEED{}, 100500);

        auto seed = external_seed == 0
                        ? std::random_device{}()
                        : static_cast<std::random_device::result_type>(external_seed);
        std::cout << "PRNG seed: " << seed << "\n";
        return seed;
    }()};
    return gen;
}
} // namespace details

// similar to std::generate_canonical, but simpler and faster in cost of quality
// and generate full [0;1] range instead of [0;1)
template <typename T>
inline T gen_canonical()
{
    if constexpr(std::is_floating_point_v<T>) // native fp
    {
        static constexpr T range = static_cast<T>(1) / static_cast<T>(details::glibc_gen::max() -
                                                                      details::glibc_gen::min());
        return range * static_cast<T>(details::get_prng()() - details::glibc_gen::min());
    }
    else if constexpr(std::is_integral_v<T>)
    {
        return static_cast<T>((details::get_prng()() >> 4) & 0x1);
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
    return gen_0_to_B(B - A) + A;
}

template <typename T>
inline T gen_off_range(T offset, T range)
{
    static_assert(std::is_integral_v<T>);
    return prng::gen_0_to_B(range) + offset;
}
} // namespace prng
#endif // GUARD_RANDOM_GEN_
