#ifndef SOURCE_FILE_DESC_HPP

#define SOURCE_FILE_DESC_HPP
#include <string>

class SourceFileDesc
{
public:
    const std::string path;
    int included_line;
    SourceFileDesc* included_from;

    SourceFileDesc(const std::string& path, SourceFileDesc* from, int line);
};

#endif // SOURCE_FILE_DESC_HPP
