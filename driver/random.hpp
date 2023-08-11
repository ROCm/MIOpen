#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <miopen/env.hpp>
#include <miopen/logger.hpp>

#include <random>

std::minstd_rand& get_minstd_gen()
{
    static thread_local std::minstd_rand minstd_gen{[]() {
        auto external_seed = miopen::EnvvarValue("MIOPEN_DRIVER_PRNG_SEED", 100500);

        auto seed = external_seed == 0
                        ? std::random_device{}()
                        : static_cast<std::random_device::result_type>(external_seed);
        MIOPEN_LOG_I("Random seed: " << seed);
        return seed;
    }()};

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

#endif // GUARD_RANDOM_GEN_
