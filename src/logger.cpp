#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <cstdlib>

namespace miopen {

bool IsLogging()
{
    return miopen::IsEnvvarValueEnabled("MIOPEN_ENABLE_LOGGING");
}

} // namespace miopen
