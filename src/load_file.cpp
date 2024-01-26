#include <miopen/load_file.hpp>
#include <sstream>
#include <fstream>

namespace miopen {

std::string LoadFile(const fs::path& p) { return LoadFile(p.string()); }

std::string LoadFile(const std::string& s)
{
    const std::ifstream t(s);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

} // namespace miopen
