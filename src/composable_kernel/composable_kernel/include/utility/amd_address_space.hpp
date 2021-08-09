#ifndef CK_AMD_ADDRESS_SPACE_HPP
#define CK_AMD_ADDRESS_SPACE_HPP

#include "config.hpp"

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
    return (T*)p;
}

} // namespace ck

#endif
