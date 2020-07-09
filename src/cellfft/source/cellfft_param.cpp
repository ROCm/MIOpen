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

//clang-format off
#include "../include/cellfft_param.hpp"

#define PSIZE(n,m) (((n)+(m))&(~(m)))

static inline uint32_t bim_fls( uint32_t n )
{
    n=n|(n>>0x01);
    n=n|(n>>0x02);
    n=n|(n>>0x04);
    n=n|(n>>0x08);
    n=n|(n>>0x10);
    return __builtin_popcount(n);
}
static inline miopen::cellfft::magic_t idiv_magic( uint32_t nmax, uint32_t d )
{
    miopen::cellfft::magic_t magic;
    if(d==1){ magic.m=1; magic.s=0; return magic; }
    uint64_t nc=((nmax+1)/d)*d-1;
    uint32_t nbits=bim_fls(nmax);
    uint32_t r=(nbits<<1)+1;
    magic.m=-1; magic.s=-1;
    for( uint32_t s=0; s<r; s++ ){
        uint64_t exp=static_cast<uint64_t>(1u)<<s;
        uint64_t mod=d-1-(exp-1)%d;
        if(exp>(nc*mod)){
            magic.m=static_cast<uint32_t>((exp+mod)/d);
            magic.s=s;
            break;
        }
    }
    return magic;
}
static uint32_t choose_optimal_cell_id( uint32_t anx, uint32_t any, uint32_t bnx, uint32_t bny )
{
    uint32_t id=1, n=0x7fffffff;
    for( int i=1; i>=0; --i ){
        uint32_t cell_size=1<<(4+i);
        uint32_t tile_x=cell_size-bnx+1;
        uint32_t tile_y=cell_size-bny+1;
        uint32_t grid_x=(anx+tile_x-bnx)/tile_x;
        uint32_t grid_y=(any+tile_y-bny)/tile_y;
        uint32_t size=grid_x*grid_y*cell_size*((cell_size>>1)+1);
        if(size<n){ id=i; n=size; }
    }
    return id;
}

namespace miopen
{
namespace cellfft
{
size_t get_auxbuf_size( const ConvolutionContext& ctx )
{
    uint32_t bs   =ctx.batch_sz;
    uint32_t inc  =ctx.n_inputs;
    uint32_t onc  =ctx.n_outputs;
    uint32_t anx  =ctx.in_width;
    uint32_t any  =ctx.in_height;
    uint32_t fnx  =ctx.kernel_size_w;
    uint32_t fny  =ctx.kernel_size_h;
    uint32_t pad_u=ctx.pad_w;
    uint32_t pad_v=ctx.pad_h;
    if(!ctx.direction.IsForward()){
        pad_u=fnx-pad_u-1;
        pad_v=fny-pad_v-1;
    }
    uint32_t pnx=anx+(pad_u<<1);
    uint32_t pny=any+(pad_v<<1);
    uint32_t id=choose_optimal_cell_id( pnx, pny, fnx, fny );
    uint32_t cell=1<<(4+id);
    uint32_t nbanks=cell*((cell>>1)+1);
    uint32_t tile_x=cell-fnx+1;
    uint32_t tile_y=cell-fny+1;
    uint32_t grid_x=(pnx+tile_x-fnx)/tile_x;
    uint32_t grid_y=(pny+tile_y-fny)/tile_y;
    uint32_t m=bs*grid_x*grid_y;
    uint32_t n=onc;
    uint32_t k=inc;
    uint32_t ek=PSIZE(k,7);
    uint32_t lda=PSIZE(m,31)>>5;
    uint32_t ldb=PSIZE(n,31)>>5;
    lda=(lda+(1^(lda&1)))<<5;
    ldb=(ldb+(1^(ldb&1)))<<5;
    uint64_t abks=lda*ek+16;
    uint64_t bbks=ldb*ek+16;
    uint64_t cbks=lda* n+16;
    return ((abks+bbks+cbks)*(nbanks<<3));
}
size_t get_auxbuf_size_grad( const ConvolutionContext& ctx )
{
    uint32_t bs   =ctx.batch_sz;
    uint32_t pnc  =ctx.n_outputs;
    uint32_t qnc  =ctx.n_inputs;
    uint32_t cnx  =ctx.kernel_size_w;
    uint32_t cny  =ctx.kernel_size_h;
    uint32_t anx  =ctx.out_width;
    uint32_t any  =ctx.out_height;
    uint32_t pad_u=ctx.pad_w;
    uint32_t pad_v=ctx.pad_h;
    uint32_t pnx=anx;
    uint32_t pny=any;
    if((pad_u|pad_v)!=0){
        pnx+=pad_u<<1;
        pny+=pad_v<<1;
    }
    uint32_t id=choose_optimal_cell_id( pnx, pny, cnx, cny );
    uint32_t cell=1<<(4+id);
    uint32_t nbanks=cell*((cell>>1)+1);
    uint32_t tile_x=cell-cnx+1;
    uint32_t tile_y=cell-cny+1;
    uint32_t grid_x=(pnx+tile_x-cnx)/tile_x;
    uint32_t grid_y=(pny+tile_y-cny)/tile_y;
    uint32_t k=bs*grid_x*grid_y;
    uint32_t ek=PSIZE(k,7);
    uint32_t lda=PSIZE(pnc,31)>>5;
    uint32_t ldb=PSIZE(qnc,31)>>5;
    lda=(lda+(1^(lda&1)))<<5;
    ldb=(ldb+(1^(ldb&1)))<<5;
    uint32_t abks=lda*ek +16;
    uint32_t bbks=ldb*ek +16;
    uint32_t cbks=lda*qnc+16;
    return ((abks+bbks+cbks)*(nbanks<<3));
}
size_t get_auxbuf_size( const cellfft_param_t& p )
{
    return ((p.abks+p.bbks+p.cbks)*(p.nbanks<<3));
}
void build_cellfft_params( cellfft_param_t& p, const ConvolutionContext& ctx )
{
    uint32_t bs   =ctx.batch_sz;
    uint32_t inc  =ctx.n_inputs;
    uint32_t onc  =ctx.n_outputs;
    uint32_t anx  =ctx.in_width;
    uint32_t any  =ctx.in_height;
    uint32_t bnx  =ctx.kernel_size_w;
    uint32_t bny  =ctx.kernel_size_h;
    uint32_t cnx  =ctx.out_width;
    uint32_t cny  =ctx.out_height;
    uint32_t pad_u=ctx.pad_w;
    uint32_t pad_v=ctx.pad_h;
    p.dir=ctx.direction.IsForward()?0:1;
    if(!ctx.direction.IsForward()){
        pad_u=bnx-pad_u-1;
        pad_v=bny-pad_v-1;
    }
    p.pad_l=pad_u;
    p.pad_r=pad_u;
    p.pad_t=pad_v;
    p.pad_b=pad_v;
    uint32_t pnx=anx+(pad_u<<1);
    uint32_t pny=any+(pad_v<<1);
    p.id=choose_optimal_cell_id( pnx, pny, bnx, bny );
    uint32_t cell=1<<(4+p.id);
    p.nbanks=cell*((cell>>1)+1);
    p.tile_x=cell-bnx+1;
    p.tile_y=cell-bny+1;
    p.grid_x=(pnx+p.tile_x-bnx)/p.tile_x;
    p.grid_y=(pny+p.tile_y-bny)/p.tile_y;
    p.m=bs*p.grid_x*p.grid_y;
    p.n=onc;
    p.k=inc;
    uint32_t ek=PSIZE(p.k,7);
    p.lda=PSIZE(p.m,31)>>5;
    p.ldb=PSIZE(p.n,31)>>5;
    p.lda=(p.lda+(1^(p.lda&1)))<<5;
    p.ldb=(p.ldb+(1^(p.ldb&1)))<<5;
    p.abks=p.lda*ek+16;
    p.bbks=p.ldb*ek+16;
    p.cbks=p.lda*p.n+16;
    p.aldy=anx*any;
    p.cldy=cnx*cny;
    p.bldy=bnx*bny;
    p.aldx=inc*p.aldy;
    p.cldx=onc*p.cldy;
    uint32_t pnc=p.dir==0?inc:onc;
    p.bldx=(p.dir==0?pnc:1)*p.bldy;
    p.bldy=(p.dir==0?1:pnc)*p.bldy;
    p.anx=anx;
    p.any=any;
    p.cnx=cnx;
    p.cny=cny;
    p.bnx=bnx;
    p.bny=bny;
    if((p.grid_x|p.grid_y)!=1){
        uint32_t pm=PSIZE(p.m,15);
        uint32_t reso=p.grid_x*p.grid_y;
        p.xmag=idiv_magic(pm,reso);
        p.ymag=idiv_magic(reso,p.grid_x);
    }
}
void build_cellfft_params_grad( cellfft_param_t& p, const ConvolutionContext& ctx )
{
    uint32_t bs   =ctx.batch_sz;
    uint32_t pnc  =ctx.n_outputs;
    uint32_t qnc  =ctx.n_inputs;
    uint32_t anx  =ctx.out_width;
    uint32_t any  =ctx.out_height;
    uint32_t bnx  =ctx.in_width;
    uint32_t bny  =ctx.in_height;
    uint32_t cnx  =ctx.kernel_size_w;
    uint32_t cny  =ctx.kernel_size_h;
    uint32_t pad_u=ctx.pad_w;
    uint32_t pad_v=ctx.pad_h;
    uint32_t pnx=anx;
    uint32_t pny=any;
    if((pad_u|pad_v)!=0){
        pnx+=pad_u<<1;
        pny+=pad_v<<1;
    }
    p.dir=2;
    p.pad_l=pad_u;
    p.pad_r=pad_u;
    p.pad_t=pad_v;
    p.pad_b=pad_v;
    p.id=choose_optimal_cell_id( pnx, pny, cnx, cny );
    uint32_t cell=1<<(4+p.id);
    p.nbanks=cell*((cell>>1)+1);
    p.tile_x=cell-cnx+1;
    p.tile_y=cell-cny+1;
    p.grid_x=(pnx+p.tile_x-cnx)/p.tile_x;
    p.grid_y=(pny+p.tile_y-cny)/p.tile_y;
    p.m=pnc;
    p.n=qnc;
    p.k=bs*p.grid_x*p.grid_y;
    uint32_t ek=PSIZE(p.k,7);
    p.lda=PSIZE(p.m,31)>>5;
    p.ldb=PSIZE(p.n,31)>>5;
    p.lda=(p.lda+(1^(p.lda&1)))<<5;
    p.ldb=(p.ldb+(1^(p.ldb&1)))<<5;
    p.abks=p.lda* ek+16;
    p.bbks=p.ldb* ek+16;
    p.cbks=p.lda*p.n+16;
    p.aldx=anx*any;
    p.bldx=bnx*bny;
    p.cldx=cnx*cny;
    p.aldy=p.m*p.aldx;
    p.bldy=p.n*p.bldx;
    p.cldy=p.m*p.cldx;
    p.anx=anx;
    p.any=any;
    p.bnx=bnx;
    p.bny=bny;
    p.cnx=cnx;
    p.cny=cny;
    if((p.grid_x|p.grid_y)!=1){
        uint32_t xk=PSIZE(p.k,15);
        uint32_t reso=p.grid_x*p.grid_y;
        p.xmag=idiv_magic(xk,reso);
        p.ymag=idiv_magic(reso,p.grid_x);
    }
}
} //namespace cellfft
} //namespace miopen
//clang-format on
