#ifndef GUARD_OLC_WRITE_FILE_HPP
#define GUARD_OLC_WRITE_FILE_HPP

#include <boost/filesystem.hpp>
#include <manage_ptr.hpp>
#include <fstream>

namespace online_compile {

using FilePtr = OLC_MANAGE_PTR(FILE*, std::fclose);

inline void WriteFile(const std::string& content, const boost::filesystem::path& name)
{
    // std::cerr << "Write file: " << name << std::endl;
    FilePtr f{std::fopen(name.string().c_str(), "w")};
    if(std::fwrite(content.c_str(), 1, content.size(), f.get()) != content.size())
        throw std::runtime_error("Failed to write to file");
}

inline void WriteFile(const std::vector<char>& content, const boost::filesystem::path& name)
{
    // std::cerr << "Write file: " << name << std::endl;
    FilePtr f{std::fopen(name.string().c_str(), "w")};
    if(std::fwrite(&content[0], 1, content.size(), f.get()) != content.size())
        throw std::runtime_error("Failed to write to file");
}

} // namespace online_compile

#endif
