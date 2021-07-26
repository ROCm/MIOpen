#ifndef OLC_DRIVER_COMMON_HPP
#define OLC_DRIVER_COMMON_HPP

#include <half.hpp>
#include <vector>
#include <cassert>

namespace ck_driver {

static inline float get_effective_average(std::vector<float>& values)
{
    assert(!values.empty());

    if(values.size() == 1)
        return (values[0]);
    else
    {
        float sum    = 0.0f;
        float maxVal = 0.0f;

        for(const auto val : values)
        {
            if(maxVal < val)
                maxVal = val;
            sum += val;
        };

        return ((sum - maxVal) / (values.size() - 1));
    };
};

} // namespace ck_driver

#endif
