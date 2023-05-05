#ifndef CK_AMD_ADDRESS_SPACE_HPP
#define CK_AMD_ADDRESS_SPACE_HPP

#include "config.hpp"
#include "c_style_pointer_cast.hpp"

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

namespace ck {

enum AddressSpaceEnum_t
{
    Generic,
    Global,
    Lds,
    Sgpr,
    Vgpr,
};

template <typename T>
__device__ T* cast_pointer_to_generic_address_space(T CONSTANT* p)
{
    // cast a pointer in "Constant" address space (4) to "Generic" address space (0)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T*)p; // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

template <typename T>
__host__ __device__ T CONSTANT* cast_pointer_to_constant_address_space(T* p)
{
    // cast a pointer in "Generic" address space (0) to "Constant" address space (4)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T CONSTANT*)p; // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

} // namespace ck
#endif
