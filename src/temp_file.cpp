#include <miopen/temp_file.hpp>
#include <miopen/errors.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

namespace miopen {
TempFile::TempFile(const std::string& path_infix_) : path_infix(path_infix_), dir(path_infix)
{
    if(!std::ofstream{this->Path(), std::ios_base::out | std::ios_base::in | std::ios_base::trunc}
            .good())
    {
        MIOPEN_THROW("Failed to create temp file: " + this->Path());
    }
}

} // namespace miopen
