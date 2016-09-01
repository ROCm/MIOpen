#ifndef GUARD_MLOPEN_KERNEL_HPP
#define GUARD_MLOPEN_KERNEL_HPP

#if MLOPEN_BACKEND_OPENCL
#include <mlopen/oclkernel.hpp>

namespace mlopen {
using KernelInvoke = OCLKernelInvoke;

std::string GetKernelSrc(const std::string& name);
} // namespace mlopen

#elif MLOPEN_BACKEND_HIP

#endif


#endif
