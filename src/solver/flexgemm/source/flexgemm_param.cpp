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
#include "../include/flexgemm_param.hpp"

static inline uint32_t choose_routine_ufconv(uint32_t m, uint32_t n, uint32_t k, int dir)
{
    uint32_t r    = (n + 31) >> 5;
    uint32_t s    = (n + 15) >> 4;
    uint32_t mode = ((m & 1) ^ 1) + ((m & 3) == 0 ? 1 : 0);
    uint32_t id   = 1 + ((r & 3) == 0 ? ((k & 15) == 0 ? 2 : 1) : ((r & 1) ^ 1));
    if((s & 1) != 0 && n <= 112)
        id = 0;
    if((dir != 0) && (id != 0))
    {
        id = (n & 3) != 0 ? ((n & 1) != 0 ? 1 : 2) : id;
    }
    return ((mode << 16) | id);
}
static inline uint32_t choose_routine_fconv(uint32_t n, uint32_t k)
{
    uint32_t r  = (n + 31) >> 5;
    uint32_t s  = (n + 15) >> 4;
    uint32_t id = 1 + ((r & 3) == 0 ? ((k & 15) == 0 ? 2 : 1) : ((r & 1) ^ 1));
    return ((k & 7) != 0 ? 4 : (((s & 1) != 0) && (n <= 112) ? 0 : id));
}
static inline uint32_t choose_routine_bconv(uint32_t n)
{
    uint32_t r  = (n + 31) >> 5;
    uint32_t s  = (n + 15) >> 4;
    uint32_t id = 1 + ((r & 3) == 0 ? 2 : ((r & 1) ^ 1));
    return (((s & 1) != 0) && (n <= 112) ? 0 : id);
}

namespace miopen {
namespace solver {
size_t get_auxbuf_size(const ConvolutionContext& ctx)
{
    uint32_t pu  = ctx.pad_w;
    uint32_t pv  = ctx.pad_h;
    uint32_t ng  = ctx.group_counts;
    uint32_t bs  = ctx.batch_sz;
    uint32_t inc = ctx.n_inputs / ng;
    uint32_t anx = ctx.in_width;
    uint32_t any = ctx.in_height;
    uint32_t bnx = ctx.kernel_size_w;
    uint32_t bny = ctx.kernel_size_h;
    uint32_t cnx = ctx.out_width;
    uint32_t cny = ctx.out_height;
    if(!ctx.direction.IsForward())
    {
        pu = bnx - pu - 1;
        pv = bny - pv - 1;
    }
    uint32_t pnx   = anx + (pu << 1);
    uint32_t pny   = any + (pv << 1);
    uint32_t k     = bnx * bny * inc;
    uint32_t n     = ctx.n_outputs / ng;
    uint32_t ldc   = cnx * cny;
    uint32_t m     = ldc * bs;
    uint32_t fid   = choose_routine_fconv(n, k);
    uint32_t bid   = choose_routine_bconv(n);
    uint32_t id    = ctx.direction.IsForward() ? fid : bid;
    uint32_t temp  = id == 0 ? 127 : 255;
    uint32_t ntidx = (m + temp) & ~temp;
    uint32_t lda   = pnx * pny;
    if((pu | pv) != 0)
    {
        lda *= bs;
        if(lda > 1024)
        {
            temp = (lda + 63) >> 6;
            lda  = (temp + (1 ^ (temp & 1))) << 6;
        }
    }
    temp         = id != 3 ? 7 : 15;
    size_t ags   = lda * inc;
    size_t spad  = static_cast<size_t>(ng << 2) * ags;
    size_t sperm = static_cast<size_t>(ng << 2) * ((k + temp) & ~temp) * ((n + 3) & ~3);
    size_t sidx  = (ntidx << 3) + (((k + 15) & ~15) << 2) + 128;
    spad         = (pu | pv) == 0 ? 0 : spad;
    sperm        = ctx.direction.IsForward() ? 0 : sperm;
    return (spad + sperm + sidx);
}
size_t get_auxbuf_size(const param_conv_t& p) { return (p.spad + p.sperm + p.sidx); }
void build_params_ufconv(param_ufconv_t& p, const ConvolutionContext& ctx)
{
    p.ng               = ctx.group_counts;
    p.m                = ctx.out_width * ctx.out_height;
    p.n                = ctx.n_outputs / p.ng;
    p.k                = ctx.n_inputs / p.ng;
    p.dir              = ctx.direction.IsForward() ? 0 : 1;
    p.id               = choose_routine_ufconv(p.m, p.n, p.k, p.dir);
    uint32_t tile      = p.id & 0xffff;
    uint32_t mode      = p.id >> 16;
    uint32_t sx        = (0x024 >> (mode << 1)) & 0x3;
    uint32_t sy        = (0xc00 >> (tile * 3 + mode)) & 0x1;
    uint32_t alignment = (tile > 0) && (tile < 3) ? 255 : 127;
    p.dimx             = p.m * ctx.batch_sz;
    p.ntidx            = (p.dimx + alignment) & (~alignment);
    p.amag             = idiv_magic(p.ntidx >> sx, p.m >> sx);
    if(sx != sy)
    {
        p.cmag = idiv_magic(p.ntidx >> sy, p.m >> sy);
    }
}
void build_params_conv(param_conv_t& p, const ConvolutionContext& ctx)
{
    uint32_t pu = ctx.pad_w;
    uint32_t pv = ctx.pad_h;
    uint32_t su = ctx.kernel_stride_w;
    uint32_t sv = ctx.kernel_stride_h;
    uint32_t du = ctx.kernel_dilation_w;
    uint32_t dv = ctx.kernel_dilation_h;
    p.dir       = ctx.direction.IsForward() ? 0 : 1;
    p.ng        = ctx.group_counts;
    p.bs        = ctx.batch_sz;
    p.inc       = ctx.n_inputs / p.ng;
    p.anx       = ctx.in_width;
    p.any       = ctx.in_height;
    p.bnx       = ctx.kernel_size_w;
    p.bny       = ctx.kernel_size_h;
    p.cnx       = ctx.out_width;
    p.cny       = ctx.out_height;
    if(p.dir != 0)
    {
        pu = p.bnx - pu - 1;
        pv = p.bny - pv - 1;
    }
    p.pnx = p.anx + (pu << 1);
    p.pny = p.any + (pv << 1);
    p.pad = (pv << 24) | (pv << 16) | (pu << 8) | pu;
    p.sd  = (dv << 18) | (du << 12) | (sv << 6) | su;
    p.ldc = p.cnx * p.cny;
    p.m   = p.ldc * p.bs;
    p.n   = ctx.n_outputs / p.ng;
    p.k   = p.bnx * p.bny * p.inc;
    if(p.dir == 0)
    {
        p.id = choose_routine_fconv(p.n, p.k);
    }
    else
    {
        p.id = choose_routine_bconv(p.n);
    }
    uint32_t temp = p.id == 0 ? 127 : 255;
    p.ntidx       = (p.m + temp) & ~temp;
    p.lda         = p.pnx * p.pny;
    if(p.pad != 0)
    {
        p.lda *= p.bs;
        if(p.lda > 1024)
        {
            temp  = (p.lda + 63) >> 6;
            p.lda = (temp + (1 ^ (temp & 1))) << 6;
        }
    }
    temp    = p.id != 3 ? 7 : 15;
    p.ags   = p.lda * p.inc;
    p.spad  = static_cast<size_t>(p.ng << 2) * p.ags;
    p.sperm = static_cast<size_t>(p.ng << 2) * ((p.k + temp) & ~temp) * ((p.n + 3) & ~3);
    p.sidx  = (p.ntidx << 3) + (((p.k + 15) & ~15) << 2) + 128;
    p.spad  = p.pad == 0 ? 0 : p.spad;
    p.sperm = p.dir == 0 ? 0 : p.sperm;
}
} // namespace solver
} // namespace miopen
