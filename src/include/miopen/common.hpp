/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_COMMON_HPP_
#define GUARD_MIOPEN_COMMON_HPP_

#include <miopen/manage_ptr.hpp>
#include <miopen/miopen.h>

#if MIOPEN_BACKEND_OPENCL

using Data_t = cl_mem;
// Const doesnt apply to cl_mem
using ConstData_t   = Data_t;
using ManageDataPtr = MIOPEN_MANAGE_PTR(cl_mem, clReleaseMemObject);

inline Data_t DataCast(void* p) { return reinterpret_cast<Data_t>(p); }

inline ConstData_t DataCast(const void* p)
{
    // Casting away const is undefined behaviour, but we do it anyways
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    return reinterpret_cast<ConstData_t>(const_cast<void*>(p));
}

#elif MIOPEN_BACKEND_HIP

using Data_t        = void*;
using ConstData_t   = const void*;
using ManageDataPtr = MIOPEN_MANAGE_PTR(void, hipFree);

inline Data_t DataCast(void* p) { return p; }

inline ConstData_t DataCast(const void* p) { return p; }
#endif // OpenCL vs hip
#endif // GUARD_MIOPEN_COMMON_HPP_
