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

#ifndef MIOPEN_CELLFFT_OP_H
#define MIOPEN_CELLFFT_OP_H

#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include "cellfft_param.hpp"

namespace miopen {
namespace cellfft {
void lk_cgemm( const Handle&, const Kernel&, const cellfft_param_t&, void*, void*, void*, float );
void lk_fft2d_r2c_perm_a( const Handle&, const Kernel&, const cellfft_param_t&, void*, const void* );
void lk_fft2d_r2c_perm_b( const Handle&, const Kernel&, const cellfft_param_t&, void*, const void* );
void lk_fft2d_r2c_perm_pad( const Handle&, const Kernel&, const cellfft_param_t&, void*, const void* );
void lk_fft2d_r2c_perm_s( const Handle&, const Kernel&, const cellfft_param_t&, void*, const void* );
void lk_fft2d_r2c_grid_perm( const Handle&, const Kernel&, const cellfft_param_t&, void*, const void* );
void lk_fft2d_r2c_grid_perm_pad( const Handle&, const Kernel&, const cellfft_param_t&, void*, const void* );
void lk_fft2d_r2c_grid_perm_nov( const Handle&, const Kernel&, const cellfft_param_t&, void*, const void* );
void lk_fft2d_c2r_perm( const Handle&, const Kernel&, const cellfft_param_t&, void*, void* );
void lk_fft2d_c2r_grid_perm( const Handle&, const Kernel&, const cellfft_param_t&, void*, void* );
void lk_fft2d_c2r_grad_perm( const Handle&, const Kernel&, const cellfft_param_t&, void*, void* );
void lk_fft2d_c2r_grad_perm_s( const Handle&, const Kernel&, const cellfft_param_t&, void*, void* );
} //namespace cellfft
} //namespace miopen
// clang-format on

#endif
