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
#ifndef GUARD_MIOPEN_FLEXGEMM_PARAM_HPP
#define GUARD_MIOPEN_FLEXGEMM_PARAM_HPP

#include <miopen/conv/context.hpp>
#include <miopen/idiv.hpp>

namespace miopen {
namespace solver {
struct param_ufconv_t
{
    magic_t amag;
    magic_t cmag;
    uint32_t id;
    uint32_t ng;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t dimx;
    uint32_t ntidx;
    uint32_t dir;
};
struct param_conv_t
{
    size_t spad;
    size_t sidx;
    size_t sperm;
    uint32_t id;
    uint32_t ng;
    uint32_t bs;
    uint32_t sd;
    uint32_t pad;
    uint32_t pnx;
    uint32_t pny;
    uint32_t anx;
    uint32_t any;
    uint32_t bnx;
    uint32_t bny;
    uint32_t cnx;
    uint32_t cny;
    uint32_t lda;
    uint32_t ldc;
    uint32_t inc;
    uint32_t ags;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t ntidx;
    uint32_t dir;
};
size_t get_auxbuf_size(const ConvolutionContext&);
size_t get_auxbuf_size(const param_conv_t&);
void build_params_ufconv(param_ufconv_t&, const ConvolutionContext&);
void build_params_conv(param_conv_t&, const ConvolutionContext&);
} // namespace solver
} // namespace miopen
#endif
