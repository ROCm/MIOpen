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

#endif
