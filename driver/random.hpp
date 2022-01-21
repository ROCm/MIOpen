#ifndef GUARD_RANDOM_GEN_
#define GUARD_RANDOM_GEN_

#include <cstdlib>

template <typename T>
inline T FRAND(void)
{
    double d = static_cast<double>(rand() / (static_cast<double>(RAND_MAX)));
    return static_cast<T>(d);
}

inline int GET_RAND(void) { return rand(); }

template <typename T>
inline T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}

#endif // GUARD_RANDOM_GEN_
