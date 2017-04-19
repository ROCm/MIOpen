#ifndef SOURCE_FILE_DESC_HPP

#define SOURCE_FILE_DESC_HPP
#include <string>

class SourceFileDesc
{
public:
    const std::string path;
    int included_line;
    SourceFileDesc* included_from;

    SourceFileDesc(const std::string& path_, SourceFileDesc* from, int line)
        : path(path_)
        , included_from(from)
        , included_line(line)
    {

    }
};

#endif // SOURCE_FILE_DESC_HPP
