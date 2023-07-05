#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <chrono>

template <typename T>
inline T FRAND()
{
    auto const clk    = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned int seed = static_cast<unsigned int>(clk);
    double d          = static_cast<double>(rand_r(&seed) / (static_cast<double>(RAND_MAX)));
    return static_cast<T>(d);
}

inline int GET_RAND()
{
    auto const clk    = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned int seed = static_cast<unsigned int>(clk);
    return rand_r(&seed);
}

template <typename T>
inline T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}

#endif // GUARD_RANDOM_GEN_
