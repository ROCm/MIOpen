/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_COMMON_HEADER_HPP
#define CK_COMMON_HEADER_HPP

#include "config.hpp"
#include "array.hpp"
#include "container_helper.hpp"
#include "statically_indexed_array.hpp"
#include "container_element_picker.hpp"
#include "multi_index.hpp"
#include "data_type.hpp"
#include "data_type_enum.hpp"
#include "data_type_enum_helper.hpp"
#include "functional.hpp"
#include "functional2.hpp"
#include "functional3.hpp"
#include "functional4.hpp"
#include "enable_if.hpp"
#include "integral_constant.hpp"
#include "math.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "sequence_helper.hpp"
#include "synchronization.hpp"
#include "tuple.hpp"
#include "tuple_helper.hpp"
#include "type.hpp"
#include "magic_division.hpp"
#include "utility.hpp"
#include "c_style_pointer_cast.hpp"
#include "amd_address_space.hpp"
#include "amd_buffer_addressing.hpp"
#include "static_buffer.hpp"
#include "dynamic_buffer.hpp"

#include "inner_product.hpp"

// TODO: remove this
#if CK_USE_AMD_INLINE_ASM
#include "amd_inline_asm.hpp"
#endif

#if CK_USE_AMD_XDLOPS
#include "amd_xdlops.hpp"
#endif

#endif
