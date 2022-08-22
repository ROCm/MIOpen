/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
