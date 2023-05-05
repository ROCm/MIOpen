#ifndef CK_AMD_LLVM_INTRINSIC_HPP
#define CK_AMD_LLVM_INTRINSIC_HPP

#include "data_type.hpp"

namespace ck {

__device__ int32_t llvm_amdgcn_readfirstlane_i32(int32_t i) __asm("llvm.amdgcn.readfirstlane");

} // namespace ck
#endif
