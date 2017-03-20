#include <mlopen/errors.hpp>
#include <hip/hip_runtime_api.h>

namespace mlopen {

std::string HIPErrorMessage(int error, const std::string& msg)
{
    return msg + " " + hipGetErrorString(static_cast<hipError_t>(error));
}

}
