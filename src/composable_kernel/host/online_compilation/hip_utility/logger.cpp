#include <config.h>
#include <logger.hpp>
#include <iostream>
#include <string>

using namespace std;

namespace olCompile {

#if OLC_DEBUG
static LogLevel defLevel = LogLevel::Info2;
#else
static LogLevel defLevel = LogLevel::Error;
#endif

string LogLevelString(LogLevel level)
{
    switch(level)
    {
    case LogLevel::Error: return ("Error");
    case LogLevel::Warning: return ("Warning");
    case LogLevel::Info: return ("Info");
    case LogLevel::Info2: return ("Info2");
    default: return ("Unknown");
    };
};

ostream& fdt_log(LogLevel level, const char* header, const char* content)
{
    if(level > olCompile::defLevel)
    {
        return (cerr);
    };

    cerr << endl << LogLevelString(level) << ":" << header << ", " << content;

    return (cerr);
}

ostream& fdt_log() { return (cerr); };

void fdt_log_flush() { cerr << endl; }
};
