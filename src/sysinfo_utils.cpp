// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/sysinfo_utils.hpp>

#ifdef __linux__
#include <unistd.h>
#endif

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace miopen {
namespace sysinfo {

/// Returns hostname for identifying logs from different machines
std::string GetSystemHostname()
{
    char name[256] = "";
#ifdef __linux__
    if(gethostname(name, sizeof(name)) == 0)
    {
        return {name};
    }
#elif defined(_WIN32)
    WSADATA wsaData;
    int wsaResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if(wsaResult == 0)
    {
        int hostnameResult = gethostname(name, sizeof(name));
        WSACleanup();
        if(hostnameResult == 0)
        {
            return {name};
        }
    }
#endif
    return {"unknown host"};
}

} // namespace sysinfo
} // namespace miopen
