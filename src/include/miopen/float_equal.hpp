#ifndef GUARD_MLOPEN_FLOAT_EQUAL_HPP
#define GUARD_MLOPEN_FLOAT_EQUAL_HPP

#include <numeric>
#include <algorithm>
#include <cmath>

namespace miopen {

template<class... Ts>
using common_type = typename std::common_type<Ts...>::type;

struct float_equal_fn
{
    template<class T>
    static bool apply(T x, T y)
    {
        return 
            std::isfinite(x) and 
            std::isfinite(y) and
            std::nextafter(x, std::numeric_limits<T>::lowest()) <= y and 
            std::nextafter(x, std::numeric_limits<T>::max()) >= y;
    }

    template<class T, class U>
    bool operator()(T x, U y) const
    {
        return float_equal_fn::apply<common_type<T, U>>(x, y);
    }

};

static constexpr float_equal_fn float_equal{};

} // namespace miopen

#endif
