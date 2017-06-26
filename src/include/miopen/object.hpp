#ifndef GUARD_MIOPEN_HANDLE_HPP
#define GUARD_MIOPEN_HANDLE_HPP

#if defined(MIOPEN_USE_CLANG_TIDY)
#define MIOPEN_OBJECT_CAST reinterpret_cast
#else
#define MIOPEN_OBJECT_CAST static_cast
#endif

#define MIOPEN_DEFINE_OBJECT(object, ...) \
inline __VA_ARGS__& miopen_get_object(object& obj)  \
{ \
    return MIOPEN_OBJECT_CAST<__VA_ARGS__&>(obj); \
} \
inline const __VA_ARGS__& miopen_get_object(const object& obj) \
{ \
    return MIOPEN_OBJECT_CAST<const __VA_ARGS__&>(obj); \
} \
inline void miopen_destroy_object(object* p) \
{ \
    delete MIOPEN_OBJECT_CAST<__VA_ARGS__*>(p); \
}

namespace miopen {

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
auto get_object_impl(rank<1>, T& x) -> decltype(miopen_get_object(x))
{
    return miopen_get_object(x);
}

}  // namespace detail

template<class T>
auto get_object(T& x) -> decltype(detail::get_object_impl(detail::rank<1>{}, x))
{
    return detail::get_object_impl(detail::rank<1>{}, x);
}

} // namespace miopen

#endif
