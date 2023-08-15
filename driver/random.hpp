#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <miopen/env.hpp>
#include <iostream>
#include <cstdlib>
#include <random>

std::minstd_rand& get_minstd_gen()
{
    static thread_local std::minstd_rand minstd_gen{[]() {
        auto external_seed = miopen::EnvvarValue("MIOPEN_DRIVER_PRNG_SEED", 100500);

        auto seed = external_seed == 0
                        ? std::random_device{}()
                        : static_cast<std::random_device::result_type>(external_seed);
        std::cout << "Random seed: " << seed << "\n";
        return seed;
    }()};

    return minstd_gen;
}

bool use_legacy_prng()
{
    static bool legacy_prng = miopen::IsEnvvarValueEnabled("MIOPEN_USE_LEGACY_PRNG");
    return legacy_prng;
}

template <typename T>
inline T FRAND()
{

    return use_legacy_prng() ? static_cast<T>(rand() / (static_cast<double>(RAND_MAX)))
                             : std::generate_canonical<T, 1>(get_minstd_gen());
}

inline int GET_RAND() { return use_legacy_prng() ? rand() : get_minstd_gen()(); }

template <typename T>
inline T RAN_GEN(T A, T B)
{
    return static_cast<T>(FRAND<double>() * (B - A)) + A;
}

#endif // GUARD_RANDOM_GEN_
