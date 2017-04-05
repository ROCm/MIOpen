#ifndef GUARD_MIOPEN_KERNEL_HPP
#define GUARD_MIOPEN_KERNEL_HPP

#include <string>

#if MIOPEN_BACKEND_OPENCL || MIOPEN_BACKEND_HIPOC
namespace miopen {
std::string GetKernelSrc(std::string name);
} // namespace miopen
#endif

#if MIOPEN_BACKEND_OPENCL
#include <miopen/oclkernel.hpp>
#include <miopen/clhelper.hpp>

namespace miopen {
using Kernel = OCLKernel;
using KernelInvoke = OCLKernelInvoke;
using Program = SharedProgramPtr;

} // namespace miopen

#elif MIOPEN_BACKEND_HIPOC
#include <miopen/hipoc_kernel.hpp>

namespace miopen {
using Kernel = HIPOCKernel;
using KernelInvoke = HIPOCKernelInvoke;
using Program = HIPOCProgram;

} // namespace miopen
#endif


#endif
