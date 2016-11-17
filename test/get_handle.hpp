
#ifndef GUARD_GET_HANDLE_HPP
#define GUARD_GET_HANDLE_HPP

#include <mlopen/handle.hpp>

static mlopen::Handle get_handle()
{
    // static mlopen::Handle h{};
    // return h;
    return mlopen::Handle{};
}

#endif
