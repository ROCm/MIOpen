#ifndef GUARD_FIN_RANDOM_GEN_
#define GUARD_FIN_RANDOM_GEN_

template <typename T>
static T FRAND(void)
{
    double d = static_cast<double>(rand() / (static_cast<double>(RAND_MAX)));
    return static_cast<T>(d);
}

template <typename T>
static T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}

#endif // GUARD_FIN_RANDOM_GEN_
