
#ifndef GUARD_GET_HANDLE_HPP
#define GUARD_GET_HANDLE_HPP

#include <mlopen/handle.hpp>

#if 0

static mlopen::Handle get_handle()
{
    return mlopen::Handle{};
}

#else

static mlopen::Handle& get_handle()
{
    static mlopen::Handle h{};
    return h;
}

#endif

#endif
