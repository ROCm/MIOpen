#include <memory>
#include <type_traits>

#ifndef GUARD_MLOPEN_MANAGE_PTR_HPP
#define GUARD_MLOPEN_MANAGE_PTR_HPP

template<class F, F f>
struct manage_deleter
{
    template<class T>
    void operator()(T* x) const
    {
        if (x != nullptr) f(x);
    }
};

template<class T, class F, F f>
using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

#define MLOPEN_MANAGE_PTR(T, F) manage_ptr<std::remove_pointer_t<T>, decltype(&F), &F>

#endif
