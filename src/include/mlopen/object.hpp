#ifndef GUARD_MLOPEN_HANDLE_HPP
#define GUARD_MLOPEN_HANDLE_HPP

#define MLOPEN_DEFINE_OBJECT(object, ...) \
inline __VA_ARGS__& mlopen_get_object(object& obj)  \
{ \
    return static_cast<__VA_ARGS__&>(obj); \
} \
inline const __VA_ARGS__& mlopen_get_object(const object& obj) \
{ \
    return static_cast<const __VA_ARGS__&>(obj); \
} \
inline void mlopen_destroy_object(object* p) \
{ \
    delete static_cast<__VA_ARGS__*>(p); \
}

#endif
