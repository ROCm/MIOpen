#ifndef HOST_TENSOR_GENERATOR_HPP
#define HOST_TENSOR_GENERATOR_HPP

#include <cmath>
#include "config.hpp"

struct GeneratorTensor_1
{
    int value = 1;

    template <typename... Is>
    float operator()(Is... is)
    {
        return value;
    }
};

struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

template <typename T>
struct GeneratorTensor_3
{
    T min_value = 0;
    T max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        return min_value + tmp * (max_value - min_value);
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, ck::index_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

#endif
