#include <miopen/binary_cache.hpp>
#include <miopen/md5.hpp>
#include <miopen/errors.hpp>
#include <fstream>
#include <iostream>

namespace miopen {

inline bool exists(const std::string& name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

std::string GetCacheFile(const std::string& name, const std::string& args, bool is_kernel_str)
{
    const std::string path = "~/.cache/miopen/";
    return path + miopen::md5(args) + "/" + (is_kernel_str ? miopen::md5(name) : name) + ".o";
}

std::string LoadBinary(const std::string& name, const std::string& args, bool is_kernel_str)
{
    std::string f = GetCacheFile(name, args, is_kernel_str);
    if(exists(f))
    {
        return f;
    }
    else
    {
        return "";
    }
}
void SaveBinary(const std::string& binary_path, const std::string& name, const std::string& args, bool is_kernel_str)
{
    auto err = std::rename(binary_path.c_str(), GetCacheFile(name, args, is_kernel_str).c_str());
    if(err != 0)
        MIOPEN_THROW("Can't write cache file");
}

} // namespace miopen
