#include <mlopen/logger.hpp>
#include <cstdlib>

namespace mlopen {

bool IsLogging()
{
    char* cs = std::getenv("MIOPEN_ENABLE_LOGGING");
    return cs != nullptr && cs != std::string("0");
}

} // namespace mlopen
