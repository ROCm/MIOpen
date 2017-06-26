#ifndef GUARD_MIOPEN_DATATYPE_HPP
#define GUARD_MIOPEN_DATATYPE_HPP

#include <string>

namespace miopen {

std::string inline GetDataType(miopenDataType_t type)
{
    std::string type_str;
    switch(type)
    {
        case miopenFloat:
        {
            type_str = "float";
        }
        break;
        case miopenHalf:
        {
            type_str = "half";
        }
        break;
    }
    return type_str;
}

} // namespace miopen

#endif // GUARD_MIOPEN_DATATYPE_HPP
