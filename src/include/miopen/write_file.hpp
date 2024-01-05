#ifndef GUARD_MLOPEN_WRITE_FILE_HPP
#define GUARD_MLOPEN_WRITE_FILE_HPP

#include <miopen/filesystem.hpp>
#include <fstream>

namespace miopen {

inline void WriteFile(const std::string& content, const fs::path& name)
{
    // std::cerr << "Write file: " << name << std::endl;
    std::ofstream f{name};
    if(f.write(content.data(), content.size()).fail())
        MIOPEN_THROW("Failed to write to file");
}

inline void WriteFile(const std::vector<char>& content, const fs::path& name)
{
    // std::cerr << "Write file: " << name << std::endl;
    std::ofstream f{name, std::ios::binary};
    if(f.write(content.data(), content.size()).fail())
        MIOPEN_THROW("Failed to write to file");
}

} // namespace miopen

#endif
