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
#ifndef GUARD_MIOPEN_GEMM_ROCBLAS_HPP_
#define GUARD_MIOPEN_GEMM_ROCBLAS_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>

namespace miopen {

struct Handle;
struct TensorDescriptor;

enum GemmCallBackend_t
{
    default       = 0,
    rocblas       = 1,
    miopengemm    = 2,
};

miopenStatus_t CallGemmRocblas(const Handle& handle,
                                GemmDescriptor gemm_desc,
                                ConstData_t A,
                                int a_offset,
                                ConstData_t B,
                                int b_offset,
                                Data_t C,
                                int c_offset,
                                GemmCallBackend_t gemm_backend = GemmCallBackend_t::rocblas);


} // namespace miopen

#endif // MIOPEN_GEMM_ROCBLAS_HPP_