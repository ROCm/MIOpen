#include <miopen/errors.hpp>
#include <hip/hip_runtime_api.h>

namespace miopen {

std::string HIPErrorMessage(int error, const std::string& msg)
{
    return msg + " " + hipGetErrorString(static_cast<hipError_t>(error));
}

} // namespace miopen
