#ifndef CK_COMMON_HEADER_HPP
#define CK_COMMON_HEADER_HPP

#include "static_kernel_config.hpp"
#include "static_kernel_utility.hpp"
#include "static_kernel_integral_constant.hpp"
#include "static_kernel_number.hpp"
#include "static_kernel_float_type.hpp"
#include "static_kernel_ck_utils_type.hpp"
#include "static_kernel_tuple.hpp"
#include "static_kernel_math.hpp"
#include "static_kernel_sequence.hpp"
#include "static_kernel_array.hpp"
#include "static_kernel_functional.hpp"
#include "static_kernel_functional2.hpp"
#include "static_kernel_functional3.hpp"
#include "static_kernel_functional4.hpp"
#include "static_kernel_in_memory_operation.hpp"
#include "static_kernel_synchronization.hpp"

#if CK_USE_AMD_INLINE_ASM
#include "static_kernel_amd_inline_asm.hpp"
#endif

#if CK_USE_AMD_XDLOPS
#include "static_kernel_amd_xdlops.hpp"
#include "static_kernel_amd_xdlops_inline_asm.hpp"
#endif

#endif
