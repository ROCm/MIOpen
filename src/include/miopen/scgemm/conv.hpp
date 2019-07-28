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
#ifndef GUARD_MIOPEN_SCGEMM_CONV_HPP_
#define GUARD_MIOPEN_SCGEMM_CONV_HPP_

#include <miopen/scgemm/scgemm_op.hpp>
#include <miopen/scgemm/tensorshape.hpp>
#include <miopen/scgemm/scgemm.hpp>
#include <string>

namespace miopen {
namespace scgemm {
bool is_fconv(const std::string& kernel_name);

scgemm_conv_routine_t get_fconv_routine(const std::string& kernel_name);

int generate_fconv_aux(void* auxbuf,
                       const group_prop_t* gprop,
                       const std::vector<uint32_t>& strides,
                       const std::vector<uint32_t>& dilations,
                       uint32_t ntidx,
                       uint32_t nb_amap,
                       uint32_t aozero);

scgemm_fconv_params create_fconv_params(std::string& kernel_name,
                                        std::vector<uint32_t>& grids,
                                        std::vector<uint32_t>& blocks,
                                        scgemm_conv_routine_t routine,
                                        const group_prop_t* gprop,
                                        uint32_t mask,
                                        uint32_t ng);

/*
/// This fucintion is used to present how to fill in the kernel arguments for a
/// Static Compiled GEMM - Forwared Convolution Kernel.
/// Left the code here for reference.
scgemm_kernel_args_t compiled_fconv_args(
    size_t&, const void*, const void*, const void*, void*, void*, scgemm_fconv_params*, float);
*/
} // namespace scgemm
} // namespace miopen
#endif
