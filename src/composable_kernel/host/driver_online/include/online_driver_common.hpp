#ifndef ONLINE_DRIVER_COMMON_HPP
#define ONLINE_DRIVER_COMMON_HPP

namespace ck_driver {

// greatest common divisor, aka highest common factor
inline int gcd(int x, int y)
{
    if(x < 0)
    {
        return gcd(-x, y);
    }
    else if(y < 0)
    {
        return gcd(x, -y);
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y);
    }
    else
    {
        return gcd(x, y % x);
    }
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
auto gcd(X x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

} // namespace ck_driver
#endif
