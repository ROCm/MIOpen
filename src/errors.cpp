#include <miopen/errors.hpp>

namespace miopen {

const char* Exception::what() const noexcept
{
    return message.c_str();
}

} // namespace miopen
