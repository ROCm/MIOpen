#ifndef _OLC_LOGGER_HPP_
#define _OLC_LOGGER_HPP_

#include <fstream>

namespace olCompile {

enum class LogLevel
{
    Quiet   = 1,
    Error   = 2,
    Warning = 3,
    Info    = 4,
    Info2   = 5
};

std::ostream& fdt_log(LogLevel level, const char* header, const char* content);
std::ostream& fdt_log();
void fdt_log_flush();

}; // namespace olCompile

#endif
