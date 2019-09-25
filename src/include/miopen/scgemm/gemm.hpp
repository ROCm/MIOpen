/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_SCGEMM_GEMM_HPP_
#define GUARD_MIOPEN_SCGEMM_GEMM_HPP_

#include <miopen/scgemm/scgemm_op.hpp>
#include <miopen/scgemm/tensorshape.hpp>
#include <miopen/scgemm/scgemm.hpp>
#include <string>

namespace miopen {
namespace scgemm {

bool is_fgemm(const std::string& kernel_name);

scgemm_gemm_routine_t get_fgemm_routine(const std::string& kernel_name);

int generate_fgemm_aux(void* auxbuf, const group_prop_t* gprop, uint32_t ntidx);

scgemm_fgemm_params create_fgemm_params(std::string&,
                                        std::vector<uint32_t>&,
                                        std::vector<uint32_t>&,
                                        scgemm_gemm_routine_t,
                                        const group_prop_t*,
                                        uint32_t,
                                        uint32_t);

/*
/// This fucintion is used to present how to fill in the kernel arguments for a
/// Static Compiled GEMM - GEMM Kernel.
/// Left the code here for reference.
scgemm_kernel_args_t compiled_fgemm_args(
    size_t&, const void*, const void*, const void*, void*, void*, scgemm_fgemm_params*, float);
*/
} // namespace scgemm
} // namespace miopen
#endif
