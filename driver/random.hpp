#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <random>

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

#endif // GUARD_RANDOM_GEN_
