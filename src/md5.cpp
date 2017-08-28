#include <miopen/md5.hpp>
#include <openssl/md5.h>
#include <array>
#include <sstream>
#include <iomanip>

namespace miopen {

std::string md5(std::string s)
{
    std::array<unsigned char, MD5_DIGEST_LENGTH> result{};
    MD5(reinterpret_cast<const unsigned char*>(s.data()), s.length(), result.data());

    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for(auto c : result)
        sout << std::setw(2) << int{c};

    return sout.str();
}
} // namespace miopen
