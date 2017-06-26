
#ifndef GUARD_GET_HANDLE_HPP
#define GUARD_GET_HANDLE_HPP

#include <miopen/handle.hpp>

#ifndef MIOPEN_TEST_USE_GLOBAL_HANDLE
#define MIOPEN_TEST_USE_GLOBAL_HANDLE 1
#endif


#if MIOPEN_TEST_USE_GLOBAL_HANDLE

static miopen::Handle& get_handle()
{
    static miopen::Handle h{};
    return h;
}

#else

static miopen::Handle get_handle()
{
    return miopen::Handle{};
}

#endif

#endif
