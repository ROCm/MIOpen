#ifndef GUARD_MLOPEN_HANDLE_HPP
#define GUARD_MLOPEN_HANDLE_HPP

#define MLOPEN_DEFINE_HANDLE(handle, ...) \
inline __VA_ARGS__& mlopen_deref_handle(handle& h)  \
{ \
    return static_cast<__VA_ARGS__&>(h); \
} \
inline const __VA_ARGS__& mlopen_deref_handle(const handle& h) \
{ \
    return static_cast<const __VA_ARGS__&>(h); \
} \
inline void mlopen_destroy_handle(handle* p) \
{ \
    delete static_cast<__VA_ARGS__*>(p); \
}

#endif