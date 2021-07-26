#ifndef CK_COMMON_HEADER_HPP
#define CK_COMMON_HEADER_HPP

#include "config.hpp"
#include "array.hpp"
#include "container_helper.hpp"
#include "statically_indexed_array.hpp"
#include "container_element_picker.hpp"
#include "multi_index.hpp"
#include "data_type_enum.hpp"
#include "data_type.hpp"
#include "data_type_helper.hpp"
#include "functional.hpp"
#include "functional2.hpp"
#include "functional3.hpp"
#include "functional4.hpp"
#include "integral_constant.hpp"
#include "math.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "sequence_helper.hpp"
#include "synchronization.hpp"
#include "tuple.hpp"
#include "tuple_helper.hpp"
#include "type.hpp"
#include "utility.hpp"
#include "magic_division.hpp"
#include "amd_buffer_addressing_v2.hpp"
#include "static_buffer.hpp"
#include "dynamic_buffer.hpp"

// TODO: remove this
#if CK_USE_AMD_INLINE_ASM
#include "amd_inline_asm.hpp"
#endif

#if CK_USE_AMD_DLOP
#include "amd_dlop.hpp"
#endif

#if CK_USE_AMD_XDLOPS
#include "amd_xdlops.hpp"
#endif

#endif
