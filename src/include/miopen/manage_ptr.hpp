#ifndef GUARD_MIOPEN_MANAGE_PTR_HPP
#define GUARD_MIOPEN_MANAGE_PTR_HPP

#include <memory>
#include <type_traits>

namespace miopen {

template<class F, F f>
struct manage_deleter
{
    template<class T>
    void operator()(T* x) const
    {
        if (x != nullptr) { f(x);
}
    }
};

struct null_deleter
{
    template<class T>
    void operator()(T*  /*x*/) const
    {}
};

template<class T, class F, F f>
using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

} // namespace miopen

#define MIOPEN_MANAGE_PTR(T, F) miopen::manage_ptr<typename std::remove_pointer<T>::type, decltype(&F), &F>

#endif
