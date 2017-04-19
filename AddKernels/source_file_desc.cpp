#include "source_file_desc.hpp"

SourceFileDesc::SourceFileDesc(const std::string& path_, SourceFileDesc* from, int line)
    : path(path_)
    , included_from(from)
    , included_line(line)
{

}
