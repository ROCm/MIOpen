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
