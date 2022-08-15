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

typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));

#pragma clang diagnostic push
#pragma clang diagnostic ignored \
    "-Wunknown-warning-option" // clang in ROCm 4.3 does not support "reserved-identifier".
#pragma clang diagnostic ignored "-Wreserved-identifier"

__device__ float
__llvm_amdgcn_buffer_atomic_add_f32(float vdata,
                                    int32x4_t rsrc,
                                    int32_t vindex,
                                    int32_t offset,
                                    bool slc) __asm("llvm.amdgcn.buffer.atomic.fadd.f32");

extern "C" __global__ void test_llvm_amdgcn_buffer_atomic_fadd_f32_float(float* p_global)
{
    int32x4_t buffer_resource{0};
    (void)__llvm_amdgcn_buffer_atomic_add_f32(*p_global, buffer_resource, 0, 0, false);
}

#pragma clang diagnostic pop // "-Wreserved-identifier"
