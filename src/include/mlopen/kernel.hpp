#ifndef GUARD_MLOPEN_KERNEL_HPP
#define GUARD_MLOPEN_KERNEL_HPP

#include <string>

#if MLOPEN_BACKEND_OPENCL || MLOPEN_BACKEND_HIPCO
namespace mlopen {
std::string GetKernelSrc(const std::string& name);
} // namespace mlopen
#endif

#if MLOPEN_BACKEND_OPENCL
#include <mlopen/oclkernel.hpp>
#include <mlopen/clhelper.hpp>

namespace mlopen {
using Kernel = OCLKernel;
using KernelInvoke = OCLKernelInvoke;
using Program = SharedProgramPtr;

} // namespace mlopen

#elif MLOPEN_BACKEND_HIP

#endif


#endif
