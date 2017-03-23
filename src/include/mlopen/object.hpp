#ifndef GUARD_MLOPEN_HANDLE_HPP
#define GUARD_MLOPEN_HANDLE_HPP

#if defined(MLOPEN_USE_CLANG_TIDY)
#define MLOPEN_OBJECT_CAST reinterpret_cast
#else
#define MLOPEN_OBJECT_CAST static_cast
#endif

#define MLOPEN_DEFINE_OBJECT(object, ...) \
inline __VA_ARGS__& mlopen_get_object(object& obj)  \
{ \
    return MLOPEN_OBJECT_CAST<__VA_ARGS__&>(obj); \
} \
inline const __VA_ARGS__& mlopen_get_object(const object& obj) \
{ \
    return MLOPEN_OBJECT_CAST<const __VA_ARGS__&>(obj); \
} \
inline void mlopen_destroy_object(object* p) \
{ \
    delete MLOPEN_OBJECT_CAST<__VA_ARGS__*>(p); \
}

namespace mlopen {

namespace detail {
template<int N>
struct rank : rank<N-1> {};

template<>
struct rank<0> {};    


template<class T>
T& get_object_impl(rank<0>, T& x)
{
    return x;
}

template<class T>
auto get_object_impl(rank<1>, T& x) -> decltype(mlopen_get_object(x))
{
    return mlopen_get_object(x);
}

}  // namespace detail

template<class T>
auto get_object(T& x) -> decltype(detail::get_object_impl(detail::rank<1>{}, x))
{
    return detail::get_object_impl(detail::rank<1>{}, x);
}

} // namespace mlopen

#endif
