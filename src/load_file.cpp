#include <miopen/load_file.hpp>
#include <sstream>
#include <fstream>

namespace miopen {

std::string LoadFile(const std::string& s)
{
    std::ifstream t(s);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

} // namespace miopen
