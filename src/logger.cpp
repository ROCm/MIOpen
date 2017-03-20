#include <mlopen/logger.hpp>
#include <mlopen/env.hpp>
#include <cstdlib>

namespace mlopen {

bool IsLogging()
{
    return !mlopen::IsEnvvarValueDisabled("MIOPEN_ENABLE_LOGGING");
}

} // namespace mlopen
