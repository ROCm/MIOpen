#include <miopen/temp_file.hpp>
#include <miopen/errors.hpp>
#include <boost/filesystem/fstream.hpp>

#include <unistd.h>

namespace miopen {
TempFile::TempFile(const std::string& path_template) : name(path_template), dir("tmp")
{
    boost::filesystem::ofstream{dir.path / path_template};
}

} // namespace miopen
