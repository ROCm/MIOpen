#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <miopen/env.hpp>
#include <iostream>
#include <cstdlib>
#include <random>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DRIVER_PRNG_SEED)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DRIVER_PRNG_LEGACY)

inline std::minstd_rand& get_minstd_gen()
{
    static thread_local std::minstd_rand minstd_gen{[]() {
        auto external_seed = miopen::Value(MIOPEN_DRIVER_PRNG_SEED{}, 100500);

        auto seed = external_seed == 0
                        ? std::random_device{}()
                        : static_cast<std::random_device::result_type>(external_seed);
        std::cout << "PRNG seed: " << seed << "\n";
        return seed;
    }()};

    return minstd_gen;
}

template <typename T>
inline T FRAND()
{
    return miopen::IsEnabled(MIOPEN_DRIVER_PRNG_LEGACY{})
               ? static_cast<T>(rand() / (static_cast<double>(RAND_MAX)))
               : std::generate_canonical<T, 1>(get_minstd_gen());
}

inline int GET_RAND()
{
    using prng_type = std::decay_t<decltype(get_minstd_gen())>::result_type;

    return miopen::IsEnabled(MIOPEN_DRIVER_PRNG_LEGACY{}) ? static_cast<prng_type>(rand())
                                                          : get_minstd_gen()();
}

template <typename T>
inline T RAN_GEN(T A, T B)
{
    return static_cast<T>(FRAND<double>() * (B - A)) + A;
}

#endif // GUARD_RANDOM_GEN_
