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

#include <miopen/conv/invokers/flexgemm.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <boost/any.hpp>

// clang-format off
namespace miopen {
namespace conv {
InvokerFactory MakeFlexgemmInvokerFactory( const flexgemm::param_ufconv_t& p, float alpha )
{
    return [=]( const std::vector<Kernel>& kernels )
    {
        return [=]( const Handle& handle, const boost::any& prim_params )
        {
            const auto& tensors=boost::any_cast<DataInvokeParams>(prim_params).tensors;
            lk_ufconv( handle, kernels[0], p, tensors.out, tensors.in, tensors.w, alpha );
            if(handle.IsProfilingEnabled()){
                float elapsed=handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}
InvokerFactory MakeFlexgemmInvokerFactory( const flexgemm::param_conv_t& p, float alpha )
{
    return [=]( const std::vector<Kernel>& kernels )
    {
        return [=]( const Handle& handle, const boost::any& prim_params )
        {
            const auto& params=boost::any_cast<DataInvokeParams>(prim_params);
            const size_t auxsize=flexgemm::get_auxbuf_size(p);
            if(params.workSpace==nullptr||params.workSpaceSize<auxsize)
                MIOPEN_THROW("Workspace is not enough for cellfft");
            const auto& tensors=params.tensors;
            float elapsed=0.f;
            uint8_t* a=reinterpret_cast<uint8_t*>(params.workSpace);
            uint8_t* b=a+p.spad;
            uint8_t* idx=b+p.sperm;
            const void* src=p.pad!=0?static_cast<const void*>(a):tensors.in;
            const void* fil=p.dir!=0?static_cast<const void*>(b):tensors.w;
            int ikern=0;
            if(p.pad!=0){
                lk_padding2d( handle, kernels[ikern++], p, a, tensors.in );
                if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            }
            if(p.dir!=0){
                lk_perm2d( handle, kernels[ikern++], p, b, tensors.w );
                if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }
            }
            lk_genidx2d( handle, kernels[ikern++], p, idx );
            if(handle.IsProfilingEnabled()){ elapsed+=handle.GetKernelTime(); }

            lk_conv( handle, kernels[ikern], p, tensors.out, src, fil, idx, alpha );
            if(handle.IsProfilingEnabled()){
                elapsed+=handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}
} // namespace conv
} // namespace miopen
// clang-format on
