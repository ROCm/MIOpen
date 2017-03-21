#ifndef GUARD_MLOPEN_ENV_HPP
#define GUARD_MLOPEN_ENV_HPP

#include <cstdlib>
#include <cstring>

namespace mlopen {

/*
 * Returns false if a feature-controlling environment variable is defined
 * and set to something which disables a feature.
 */
inline bool IsEnvvarValueDisabled(const char* name)
{
    const auto value_env_p = std::getenv(name);
    return value_env_p != nullptr && 
         ( std::strcmp(value_env_p, "disable") == 0
        || std::strcmp(value_env_p, "disabled") == 0
        || std::strcmp(value_env_p, "0") == 0
        || std::strcmp(value_env_p, "no") == 0
        || std::strcmp(value_env_p, "false") == 0 );
}
} // namespace mlopen

#endif
