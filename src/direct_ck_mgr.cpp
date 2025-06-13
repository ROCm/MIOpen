#include "miopen/direct_ck_mgr.hpp"

#include <cstdlib>
#include <functional>

static bool IsEnvEnabled(const char* pEnv)
{
    const char* env_val = std::getenv(pEnv);
    if (env_val == nullptr)
    {
        return true;
    }

    std::string value(env_val);
    
    // Trim leading whitespace
    size_t start = value.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return false;
    }
    value.erase(0, start);

    // Trim trailing whitespace
    size_t end = value.find_last_not_of(" \t\n\r\f\v");
    if (end != std::string::npos) {
        value.erase(end + 1);
    }

    return value == "1";
}

DirectCkMgr::DirectCkMgr()
{
    Init();
}

void DirectCkMgr::Init()
{
    enableConvCache = IsEnvEnabled("DCK_CONV_FILE_CACHE");
}

size_t DirectCkMgr::GetStringHash(std::string str)
{
    std::hash<std::string> hasher;
    size_t hashValue = hasher(str);

    return hashValue;
}
