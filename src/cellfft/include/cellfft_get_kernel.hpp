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

// clang-format off
#ifndef MIOPEN_CELLFFT_GET_KERNEL_H
#define MIOPEN_CELLFFT_GET_KERNEL_H

#include <miopen/kernel_info.hpp>
#include "cellfft_param.hpp"

namespace miopen {
namespace cellfft {
solver::KernelInfo get_kernel_cgemm( const cellfft_param_t&, const std::string& );
solver::KernelInfo get_kernel_r2c_a( const cellfft_param_t&, const std::string&, uint32_t );
solver::KernelInfo get_kernel_r2c_b( const cellfft_param_t&, const std::string& );
solver::KernelInfo get_kernel_r2c_grid( const cellfft_param_t&, const std::string& );
solver::KernelInfo get_kernel_r2c_xgrad_a( const cellfft_param_t&, const std::string& );
solver::KernelInfo get_kernel_r2c_xgrad_b( const cellfft_param_t&, const std::string& );
solver::KernelInfo get_kernel_c2r( const cellfft_param_t&, const std::string&, uint32_t );
solver::KernelInfo get_kernel_c2r_grid( const cellfft_param_t&, const std::string&, uint32_t );
solver::KernelInfo get_kernel_c2r_grad( const cellfft_param_t&, const std::string& );
} // namespace cellfft
} // namespace miopen
#endif
// clang-format on
