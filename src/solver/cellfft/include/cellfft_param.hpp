/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_CELLFFT_PARAM_HPP
#define GUARD_MIOPEN_CELLFFT_PARAM_HPP

#include <miopen/conv/context.hpp>
#include <miopen/idiv.hpp>

namespace miopen {
namespace solver {
struct cellfft_param_t
{
    magic_t xmag;
    magic_t ymag;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t tile_x;
    uint32_t tile_y;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t abks;
    uint32_t bbks;
    uint32_t cbks;
    uint32_t aldx;
    uint32_t aldy;
    uint32_t bldx;
    uint32_t bldy;
    uint32_t cldx;
    uint32_t cldy;
    uint32_t anx;
    uint32_t any;
    uint32_t bnx;
    uint32_t bny;
    uint32_t cnx;
    uint32_t cny;
    uint32_t pad_l;
    uint32_t pad_r;
    uint32_t pad_t;
    uint32_t pad_b;
    uint32_t nbanks;
    uint32_t id;
    uint32_t dir;
};
size_t get_auxbuf_size(const ConvolutionContext&);
size_t get_auxbuf_size_grad(const ConvolutionContext&);
size_t get_auxbuf_size(const cellfft_param_t&);
void build_cellfft_params(cellfft_param_t&, const ConvolutionContext&);
void build_cellfft_params_grad(cellfft_param_t&, const ConvolutionContext&);
} // namespace solver
} // namespace miopen
#endif
