#ifndef GUARD_MLOPEN_HIPOC_PROGRAM_HPP
#define GUARD_MLOPEN_HIPOC_PROGRAM_HPP

#include <string>
#include <mlopen/manage_ptr.hpp>
#include <hip/hip_runtime.h>

namespace mlopen {

using hipModulePtr = MLOPEN_MANAGE_PTR(hipModule_t, hipModuleUnload);
struct HIPOCProgram
{
    using SharedModulePtr = std::shared_ptr<typename std::remove_pointer<hipModule_t>::type>;
    using FilePtr = MLOPEN_MANAGE_PTR(FILE*, std::fclose);
    HIPOCProgram();
    HIPOCProgram(const std::string &program_name, std::string params);
    SharedModulePtr modulue;
};
}

#endif
