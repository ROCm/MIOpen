#include <miopen/load_file.hpp>

#include <fstream>
#include <sstream>

namespace miopen {

std::string LoadFile(const std::string& path)
{
    const std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

std::vector<char> LoadFileAsVector(const std::string& path)
{
    auto file = std::ifstream{path, std::ios::binary | std::ios::ate};
    const auto file_size = file.tellg();
    file.seekg(std::ios::beg);
    auto ret = std::vector<char>(file_size);
    file.read(&ret[0], file_size);
    return ret;
}

} // namespace miopen
