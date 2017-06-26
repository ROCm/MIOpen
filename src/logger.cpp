#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <cstdlib>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING)

bool IsLogging()
{
    return miopen::IsEnabled(MIOPEN_ENABLE_LOGGING{});
}

} // namespace miopen
