#ifndef GUARD_MIOPEN_KERNEL_HPP
#define GUARD_MIOPEN_KERNEL_HPP

#include <string>

#include <miopen/config.h>

namespace miopen {
std::string GetKernelSrc(std::string name);
} // namespace miopen

#if MIOPEN_BACKEND_OPENCL
#include <miopen/oclkernel.hpp>
#include <miopen/clhelper.hpp>

namespace miopen {
using Kernel = OCLKernel;
using KernelInvoke = OCLKernelInvoke;
using Program = SharedProgramPtr;

} // namespace miopen

#elif MIOPEN_BACKEND_HIP
#include <miopen/hipoc_kernel.hpp>

namespace miopen {
using Kernel = HIPOCKernel;
using KernelInvoke = HIPOCKernelInvoke;
using Program = HIPOCProgram;

} // namespace miopen
#endif


#endif
