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
#include "../include/cellfft_op.hpp"

namespace miopen {
namespace cellfft {
void lk_cgemm( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* c, void* a, void* b, float alpha )
{
    float coef=alpha*(p.id==0?0.00390625f:0.0009765625f);
    handle.Run(kern)( c, p.lda, p.cbks, a, b, p.lda, p.ldb, p.m, p.n, p.k, p.abks, p.bbks, coef );
}
void lk_fft2d_r2c_perm_a( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src )
{
    handle.Run(kern)( dst, p.lda, p.abks, src, (p.dir!=2?0:0x80000000)|p.m, p.anx , p.aldx, p.aldy );
}
void lk_fft2d_r2c_perm_b( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src )
{
    handle.Run(kern)( dst, p.ldb, p.bbks, src, (p.dir==0?0:0x80000000)|p.n, p.bnx, p.bldx, p.bldy );
}
void lk_fft2d_r2c_perm_pad( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src )
{
    handle.Run(kern)( dst, p.lda, p.abks, src, p.m, (p.dir!=2?0:0x80000000)|p.pad_l, p.aldx, p.aldy, p.anx , p.any );
}
void lk_fft2d_r2c_perm_s( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t ldr=p.bnx*p.bny*(p.dir==0?p.k:p.n);
    handle.Run(kern)( dst, p.ldb, p.bbks, src, p.n, ldr );
}
void lk_fft2d_r2c_grid_perm( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    handle.Run(kern)( dst, p.lda, p.abks, src, p.m, p.anx, p.aldx, p.aldy, grid, tile, p.xmag, p.ymag, p.any );
}
void lk_fft2d_r2c_grid_perm_pad( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    uint32_t pad=(p.pad_t<<16)|p.pad_l;
    handle.Run(kern)( dst, p.lda, p.abks, src, p.m, p.anx, p.aldx, p.aldy, grid, tile, p.xmag, p.ymag, p.any, pad );
}
void lk_fft2d_r2c_grid_perm_nov( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    handle.Run(kern)( dst, p.ldb, p.bbks, src, p.n, p.bnx, p.bldx, p.bldy, grid, tile, p.xmag, p.ymag, p.bny );
}
void lk_fft2d_c2r_perm( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src )
{
    handle.Run(kern)( dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.m, p.cnx );
}
void lk_fft2d_c2r_grid_perm( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    handle.Run(kern)( dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.xmag, p.ymag, grid, tile, p.cnx, p.cny, p.m );
}
void lk_fft2d_c2r_grad_perm( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src )
{
    handle.Run(kern)( dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.m, p.cnx );
}
void lk_fft2d_c2r_grad_perm_s( const Handle& handle, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src )
{
    handle.Run(kern)( dst, p.cnx*p.cny*p.m, p.m, src, p.lda, p.cbks );
}
} //namespace cellfft
} //namespace miopen
// clang-format on
