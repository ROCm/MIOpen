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
#include "../include/flexgemm_op.hpp"

namespace miopen {
namespace flexgemm {
void lk_ufconv(const Handle& handle, const Kernel& kern, const param_ufconv_t& p, void* c, const void* a, const void* b, float alpha)
{
    handle.Run(kern)(a, b, p.ng, p.m, p.n, p.k, p.amag, p.cmag, c, alpha, p.dimx);
}
void lk_padding2d(const Handle& handle, const Kernel& kern, const param_conv_t& p, void* dst, const void* src)
{
    handle.Run(kern)(dst, src, p.anx, p.any, p.pad, p.ng*p.inc, p.lda);
}
void lk_perm2d(const Handle& handle, const Kernel& kern, const param_conv_t& p, void* dst, const void* src)
{
    uint32_t bnn=p.bnx*p.bny;
    uint32_t lda=p.ng*p.n*bnn;
    uint32_t align=p.id!=3?7:15;
    handle.Run(kern)(dst, src, lda, (p.n+3)&~3, p.k, p.n, bnn, (p.k+align)&~align);
}
void lk_genidx2d(const Handle& handle, const Kernel& kern, const param_conv_t& p, void* dst)
{
    uint32_t npx=p.pnx*p.pny;
    uint32_t inc=p.ng*p.inc;
    uint32_t onc=p.ng*p.n;
    uint32_t ldx=npx*(p.pad==0?inc:1);
    handle.Run(kern)(dst, p.ntidx, p.pnx, p.sd, ldx, onc, p.m, p.cnx, p.cny, p.bnx, p.bny, p.lda, p.k);
}
void lk_conv(const Handle& handle, const Kernel& kern, const param_conv_t& p, void* c, const void* a, const void* b, const void* idx, float alpha)
{
    const uint8_t* relo=static_cast<const uint8_t*>(idx)+(p.ntidx<<3);
    uint32_t align=p.id!=3?7:15;
    uint32_t ldb=p.dir==0?(p.k*p.ng):((p.n+3)&~3);
    uint32_t n=(p.pad!=0?0x80000000:0)|p.n;
    uint32_t k=p.dir==0?p.k:((p.k+align)&~align);
    handle.Run(kern)(idx, relo, ldb, n, k, p.ags, a, b, c, alpha, p.m, p.ldc);
}
} //namespace flexgemm
} //namespace miopen
// clang-format on
