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

#include <miopen/conv/invokers/cellfft.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <boost/any.hpp>

// clang-format off
namespace miopen {
static void lk_cgemm( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* c, void* a, void* b, float alpha )
{
    float coef=alpha*(p.id==0?0.00390625f:0.0009765625f);
    handle.Run(kern)( c, p.lda, p.cbks, a, b, p.lda, p.ldb, p.m, p.n, p.k, p.abks, p.bbks, coef );
}
static void lk_fft2d_r2c_perm_a( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    handle.Run(kern)( dst, p.lda, p.abks, src, (p.dir!=2?0:0x80000000)|p.m, p.anx , p.aldx, p.aldy );
}
static void lk_fft2d_r2c_perm_b( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    handle.Run(kern)( dst, p.ldb, p.bbks, src, (p.dir==0?0:0x80000000)|p.n, p.bnx, p.bldx, p.bldy );
}
static void lk_fft2d_r2c_perm_pad( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    handle.Run(kern)( dst, p.lda, p.abks, src, p.m, (p.dir!=2?0:0x80000000)|p.pad_l, p.aldx, p.aldy, p.anx , p.any );
}
static void lk_fft2d_r2c_perm_s( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t ldr=p.bnx*p.bny*(p.dir==0?p.k:p.n);
    handle.Run(kern)( dst, p.ldb, p.bbks, src, p.n, ldr );
}
static void lk_fft2d_r2c_grid_perm( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    handle.Run(kern)( dst, p.lda, p.abks, src, p.m, p.anx, p.aldx, p.aldy, grid, tile, p.xmag, p.ymag, p.any );
}
static void lk_fft2d_r2c_grid_perm_pad( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    uint32_t pad=(p.pad_t<<16)|p.pad_l;
    handle.Run(kern)( dst, p.lda, p.abks, src, p.m, p.anx, p.aldx, p.aldy, grid, tile, p.xmag, p.ymag, p.any, pad );
}
static void lk_fft2d_r2c_grid_perm_nov( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    handle.Run(kern)( dst, p.ldb, p.bbks, src, p.n, p.bnx, p.bldx, p.bldy, grid, tile, p.xmag, p.ymag, p.bny );
}
static void lk_fft2d_c2r_perm( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    handle.Run(kern)( dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.m, p.cnx );
}
static void lk_fft2d_c2r_grid_perm( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    uint32_t grid=(p.grid_y<<16)|p.grid_x;
    uint32_t tile=(p.tile_y<<16)|p.tile_x;
    handle.Run(kern)( dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.xmag, p.ymag, grid, tile, p.cnx, p.cny, p.m );
}
static void lk_fft2d_c2r_grad_perm( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    handle.Run(kern)( dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.m, p.cnx );
}
static void lk_fft2d_c2r_grad_perm_s( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    handle.Run(kern)( dst, p.cnx*p.cny*p.m, p.m, src, p.lda, p.cbks );
}

static void cgemm( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* c, void* a, void* b, float alpha )
{
    lk_cgemm( handle, kern, p, c, a, b, alpha );
}
static void fft2d_r2c_a( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.pad_l|p.pad_t)!=0){
        lk_fft2d_r2c_perm_pad( handle, kern, p, dst, src );
    } else {
        lk_fft2d_r2c_perm_a( handle, kern, p, dst, src );
    }
}
static void fft2d_r2c_b( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.bnx==p.bny)&&((p.bnx==3)||(p.bnx==5))){
        lk_fft2d_r2c_perm_s( handle, kern, p, dst, src );
    } else {
        lk_fft2d_r2c_perm_b( handle, kern, p, dst, src );
    }
}
static void fft2d_r2c_grid_a( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.pad_l|p.pad_t)!=0){
        lk_fft2d_r2c_grid_perm_pad( handle, kern, p, dst, src );
    } else {
        lk_fft2d_r2c_grid_perm( handle, kern, p, dst, src );
    }
}
static void fft2d_r2c_grid_b( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    lk_fft2d_r2c_grid_perm_nov( handle, kern, p, dst, src );
}
static void fft2d_c2r( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    lk_fft2d_c2r_perm( handle, kern, p, dst, src );
}
static void fft2d_c2r_grid( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    lk_fft2d_c2r_grid_perm( handle, kern, p, dst, src );
}
static void fft2d_c2r_grad( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    bool cc0=(p.cnx==p.cny)&&((p.cnx==3)||(p.cnx==5)||(p.cnx==7));
    bool cc1=(p.cnx==1)&&((p.cny&0x1)!=0&&(p.cny>1)&&(p.cny<=9));
    bool cc2=(p.cny==1)&&((p.cnx&0x1)!=0&&(p.cnx>1)&&(p.cnx<=9));
    if(cc0||cc1||cc2){
        lk_fft2d_c2r_grad_perm_s( handle, kern, p, dst, src );
    } else {
        lk_fft2d_c2r_grad_perm( handle, kern, p, dst, src );
    }
}
static void dtr( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.grid_x|p.grid_y)>1){
        fft2d_r2c_grid_a( handle, kern, p, dst, src );
    } else {
        fft2d_r2c_a( handle, kern, p, dst, src );
    }
}
static void ftr( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.dir==2)&&((p.grid_x|p.grid_y)>1)){
        fft2d_r2c_grid_b( handle, kern, p, dst, src );
    } else {
        fft2d_r2c_b( handle, kern, p, dst, src );
    }
}
static void otr( const Handle& handle, const Kernel& kern, const solver::cellfft_param_t& p, void* dst, void* src )
{
    if(p.dir!=2){
        if((p.grid_x|p.grid_y)>1){
            fft2d_c2r_grid( handle, kern, p, dst, src );
        } else {
            fft2d_c2r( handle, kern, p, dst, src );
        }
    } else {
        fft2d_c2r_grad( handle, kern, p, dst, src );
    }
}
namespace conv {
InvokerFactory MakeCellfftInvokerFactory(const solver::cellfft_param_t& conv_params, float alpha)
{
    return [=]( const std::vector<Kernel>& kernels )
    {
        return [=]( const Handle& handle, const boost::any& prim_params )
        {
            const size_t abks=static_cast<size_t>(conv_params.abks);
            const size_t bbks=static_cast<size_t>(conv_params.bbks);
            const size_t cbks=static_cast<size_t>(conv_params.cbks);
            const size_t nbks=static_cast<size_t>(conv_params.nbanks)<<3;
            const size_t auxsize=nbks*(abks+bbks+cbks);
            const auto& params=boost::any_cast<DataInvokeParams>(prim_params);
            if(params.workSpace==nullptr||params.workSpaceSize<auxsize)
                MIOPEN_THROW("Workspace is not enough for cellfft");
            const auto& tensors=params.tensors;
            auto& auxbuf=params.workSpace;
            const void* src=tensors.in;
            const void* fil=tensors.w;
            void* dst=tensors.out;
            uint8_t* a=reinterpret_cast<uint8_t*>(auxbuf);
            uint8_t* b=a+nbks*abks;
            uint8_t* c=b+nbks*bbks;
            float elapsed=0.f;
            dtr( handle, kernels[1], conv_params, a, src );
            if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            ftr( handle, kernels[2], conv_params, b, fil );
            if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            cgemm( handle, kernels[0], conv_params, c, a, b, alpha );
            if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            otr( handle, kernels[3], conv_params, dst, c );
            if(handle.IsProfilingEnabled()){
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}
InvokerFactory MakeCellfftInvokerFactoryGrad(const solver::cellfft_param_t& conv_params, float alpha)
{
    return [=]( const std::vector<Kernel>& kernels )
    {
        return [=]( const Handle& handle, const boost::any& prim_params )
        {
            const size_t abks=static_cast<size_t>(conv_params.abks);
            const size_t bbks=static_cast<size_t>(conv_params.bbks);
            const size_t cbks=static_cast<size_t>(conv_params.cbks);
            const size_t nbks=static_cast<size_t>(conv_params.nbanks)<<3;
            const size_t auxsize=nbks*(abks+bbks+cbks);
            const auto& params=boost::any_cast<WrWInvokeParams>(prim_params);
            if(params.workSpace==nullptr||params.workSpaceSize<auxsize)
                MIOPEN_THROW("Workspace is not enough for cellfft");
            const auto& tensors=params.tensors;
            auto& auxbuf=params.workSpace;
            const void* pin=tensors.x;
            const void* qin=tensors.dy;
            void* dst=tensors.dw;
            uint8_t* a=reinterpret_cast<uint8_t*>(auxbuf);
            uint8_t* b=a+nbks*abks;
            uint8_t* c=b+nbks*bbks;
            float elapsed=0.f;
            dtr( handle, kernels[1], conv_params, a, pin );
            if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            ftr( handle, kernels[2], conv_params, b, qin );
            if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            cgemm( handle, kernels[0], conv_params, c, a, b, alpha );
            if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            otr( handle, kernels[3], conv_params, dst, c );
            if(handle.IsProfilingEnabled()){
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}
} // namespace conv
} // namespace miopen
// clang-format on
