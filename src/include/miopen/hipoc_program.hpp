#ifndef GUARD_MIOPEN_HIPOC_PROGRAM_HPP
#define GUARD_MIOPEN_HIPOC_PROGRAM_HPP

#include <string>
#include <miopen/manage_ptr.hpp>
#include <hip/hip_runtime_api.h>

namespace miopen {

using hipModulePtr = MIOPEN_MANAGE_PTR(hipModule_t, hipModuleUnload);
struct HIPOCProgram
{
    using SharedModulePtr = std::shared_ptr<typename std::remove_pointer<hipModule_t>::type>;
    using FilePtr = MIOPEN_MANAGE_PTR(FILE*, std::fclose);
    HIPOCProgram();
    HIPOCProgram(const std::string &program_name, std::string params, bool is_kernel_str);
    SharedModulePtr module;
    std::string name;
};
}

#endif
