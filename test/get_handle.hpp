
#ifndef GUARD_GET_HANDLE_HPP
#define GUARD_GET_HANDLE_HPP

#include <mlopen/handle.hpp>

#ifndef MLOPEN_TEST_USE_GLOBAL_HANDLE
#define MLOPEN_TEST_USE_GLOBAL_HANDLE 1
#endif


#if MLOPEN_TEST_USE_GLOBAL_HANDLE

static mlopen::Handle& get_handle()
{
    static mlopen::Handle h{};
    return h;
}

#else

static mlopen::Handle get_handle()
{
    return mlopen::Handle{};
}

#endif

#endif
