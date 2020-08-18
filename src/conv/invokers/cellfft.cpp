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

#include <miopen/conv/invokers/cellfft.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <boost/any.hpp>
#include "../../cellfft/include/cellfft_op.hpp"

// clang-format off
static void fft2d_r2c_a( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.pad_l|p.pad_t)!=0){
        miopen::cellfft::lk_fft2d_r2c_perm_pad( handle, kern, p, dst, src );
    } else {
        miopen::cellfft::lk_fft2d_r2c_perm_a( handle, kern, p, dst, src );
    }
}
static void fft2d_r2c_b( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.bnx==p.bny)&&((p.bnx==3)||(p.bnx==5))){
        miopen::cellfft::lk_fft2d_r2c_perm_s( handle, kern, p, dst, src );
    } else {
        miopen::cellfft::lk_fft2d_r2c_perm_b( handle, kern, p, dst, src );
    }
}
static void fft2d_r2c_grid_a( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.pad_l|p.pad_t)!=0){
        miopen::cellfft::lk_fft2d_r2c_grid_perm_pad( handle, kern, p, dst, src );
    } else {
        miopen::cellfft::lk_fft2d_r2c_grid_perm( handle, kern, p, dst, src );
    }
}
static void fft2d_r2c_grid_b( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, const void* src )
{
    miopen::cellfft::lk_fft2d_r2c_grid_perm_nov( handle, kern, p, dst, src );
}
static void fft2d_c2r( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, void* src )
{
    miopen::cellfft::lk_fft2d_c2r_perm( handle, kern, p, dst, src );
}
static void fft2d_c2r_grid( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, void* src )
{
    miopen::cellfft::lk_fft2d_c2r_grid_perm( handle, kern, p, dst, src );
}
static void fft2d_c2r_grad( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, void* src )
{
    bool cc0=(p.cnx==p.cny)&&((p.cnx==3)||(p.cnx==5)||(p.cnx==7));
    bool cc1=(p.cnx==1)&&((p.cny&0x1)!=0&&(p.cny>1)&&(p.cny<=9));
    bool cc2=(p.cny==1)&&((p.cnx&0x1)!=0&&(p.cnx>1)&&(p.cnx<=9));
    if(cc0||cc1||cc2){
        miopen::cellfft::lk_fft2d_c2r_grad_perm_s( handle, kern, p, dst, src );
    } else {
        miopen::cellfft::lk_fft2d_c2r_grad_perm( handle, kern, p, dst, src );
    }
}
static void cgemm( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* c, void* a, void* b, float alpha )
{
    miopen::cellfft::lk_cgemm( handle, kern, p, c, a, b, alpha );
}
static void dtr( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.grid_x|p.grid_y)>1){
        fft2d_r2c_grid_a( handle, kern, p, dst, src );
    } else {
        fft2d_r2c_a( handle, kern, p, dst, src );
    }
}
static void ftr( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, const void* src )
{
    if((p.dir==2)&&((p.grid_x|p.grid_y)>1)){
        fft2d_r2c_grid_b( handle, kern, p, dst, src );
    } else {
        fft2d_r2c_b( handle, kern, p, dst, src );
    }
}
static void otr( const miopen::Handle& handle, const miopen::Kernel& kern, const miopen::cellfft::cellfft_param_t& p, void* dst, void* src )
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

namespace miopen {
namespace conv {
InvokerFactory MakeCellfftInvokerFactory( const cellfft::cellfft_param_t& conv_params, float alpha )
{
    return [=]( const std::vector<Kernel>& kernels )
    {
        return [=]( const Handle& handle, const boost::any& prim_params )
        {
            const size_t abks=static_cast<size_t>(conv_params.abks);
            const size_t bbks=static_cast<size_t>(conv_params.bbks);
            const size_t cbks=static_cast<size_t>(conv_params.cbks);
            const size_t nbks=static_cast<size_t>(conv_params.nbanks);
            const size_t auxsize=(nbks<<3)*(abks+bbks+cbks);
            const auto& params=boost::any_cast<DataInvokeParams>(prim_params);
            if(params.workSpace==nullptr||params.workSpaceSize<auxsize)
                MIOPEN_THROW("Workspace is not enough for cellfft");
            const auto& tensors=params.tensors;
            auto& auxbuf=params.workSpace;
            const void* src=tensors.in;
            const void* fil=tensors.w;
            void* dst=tensors.out;
            uint8_t* a=reinterpret_cast<uint8_t*>(auxbuf);
            uint8_t* b=a+((nbks*abks)<<3);
            uint8_t* c=b+((nbks*bbks)<<3);
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
InvokerFactory MakeCellfftInvokerFactoryGrad( const cellfft::cellfft_param_t& conv_params, float alpha )
{
    return [=]( const std::vector<Kernel>& kernels )
    {
        return [=]( const Handle& handle, const boost::any& prim_params )
        {
            const size_t abks=static_cast<size_t>(conv_params.abks);
            const size_t bbks=static_cast<size_t>(conv_params.bbks);
            const size_t cbks=static_cast<size_t>(conv_params.cbks);
            const size_t nbks=static_cast<size_t>(conv_params.nbanks);
            const size_t auxsize=(nbks<<3)*(abks+bbks+cbks);
            const auto& params=boost::any_cast<WrWInvokeParams>(prim_params);
            if(params.workSpace==nullptr||params.workSpaceSize<auxsize)
                MIOPEN_THROW("Workspace is not enough for cellfft");
            const auto& tensors=params.tensors;
            auto& auxbuf=params.workSpace;
            const void* pin=tensors.x;
            const void* qin=tensors.dy;
            void* dst=tensors.dw;
            uint8_t* a=reinterpret_cast<uint8_t*>(auxbuf);
            uint8_t* b=a+((nbks*abks)<<3);
            uint8_t* c=b+((nbks*bbks)<<3);
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
