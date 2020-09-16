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
#include <miopen/conv/invokers/flexgemm.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <boost/any.hpp>

namespace miopen {
static void lk_ufconv(const Handle& handle,
                      const Kernel& kern,
                      const solver::param_ufconv_t& p,
                      void* c,
                      const void* a,
                      const void* b,
                      float alpha)
{
    handle.Run(kern)(a, b, p.ng, p.m, p.n, p.k, p.amag, p.cmag, c, alpha, p.dimx);
}
static void lk_padding2d(const Handle& handle,
                         const Kernel& kern,
                         const solver::param_conv_t& p,
                         void* dst,
                         const void* src)
{
    handle.Run(kern)(dst, src, p.anx, p.any, p.pad, p.ng * p.inc, p.lda);
}
static void lk_perm2d(const Handle& handle,
                      const Kernel& kern,
                      const solver::param_conv_t& p,
                      void* dst,
                      const void* src)
{
    uint32_t bnn   = p.bnx * p.bny;
    uint32_t lda   = p.n * bnn;
    uint32_t align = p.id != 3 ? 7 : 15;
    handle.Run(kern)(dst, src, lda, (p.n + 3) & ~3, p.k, p.n, bnn, (p.k + align) & ~align);
}
static void
lk_genidx2d(const Handle& handle, const Kernel& kern, const solver::param_conv_t& p, void* dst)
{
    uint32_t npx = p.pnx * p.pny;
    uint32_t inc = p.ng * p.inc;
    uint32_t onc = p.ng * p.n;
    uint32_t ldx = npx * (p.pad == 0 ? inc : 1);
    handle.Run(kern)(
        dst, p.ntidx, p.pnx, p.sd, ldx, onc, p.m, p.cnx, p.cny, p.bnx, p.bny, p.lda, p.k);
}
static void lk_conv(const Handle& handle,
                    const Kernel& kern,
                    const solver::param_conv_t& p,
                    void* c,
                    const void* a,
                    const void* b,
                    const void* idx,
                    float alpha)
{
    const uint8_t* relo = static_cast<const uint8_t*>(idx) + (p.ntidx << 3);
    uint32_t n          = (p.pad != 0 ? 0x80000000 : 0) | p.n;
    if(p.dir == 0)
    {
        handle.Run(kern)(idx, relo, n, p.k, p.ags, alpha, a, b, c, p.m, p.ldc);
    }
    else
    {
        uint32_t align = p.id != 3 ? 7 : 15;
        uint32_t k     = (p.k + align) & ~align;
        handle.Run(kern)(idx, relo, (p.n + 3) & ~3, n, k, p.ags, a, b, c, alpha, p.m, p.ldc);
    }
}

namespace conv {
InvokerFactory MakeFlexgemmInvokerFactory(const solver::param_ufconv_t& p, float alpha)
{
    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const boost::any& prim_params) {
            const auto& tensors = boost::any_cast<DataInvokeParams>(prim_params).tensors;
            lk_ufconv(handle, kernels[0], p, tensors.out, tensors.in, tensors.w, alpha);
            if(handle.IsProfilingEnabled())
            {
                float elapsed = handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}
InvokerFactory MakeFlexgemmInvokerFactory(const solver::param_conv_t& p, float alpha)
{
    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const boost::any& prim_params) {
            const auto& params   = boost::any_cast<DataInvokeParams>(prim_params);
            const size_t auxsize = get_auxbuf_size(p);
            if(params.workSpace == nullptr || params.workSpaceSize < auxsize)
                MIOPEN_THROW("Workspace is not enough for flexgemm");
            const auto& tensors = params.tensors;
            float elapsed       = 0.f;
            uint8_t* a          = reinterpret_cast<uint8_t*>(params.workSpace);
            uint8_t* b          = a + p.spad;
            uint8_t* idx        = b + p.sperm;
            const void* src     = p.pad != 0 ? static_cast<const void*>(a) : tensors.in;
            const void* fil     = p.dir != 0 ? static_cast<const void*>(b) : tensors.w;
            int ikern           = 0;
            if(p.pad != 0)
            {
                lk_padding2d(handle, kernels[ikern++], p, a, tensors.in);
                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                }
            }
            if(p.dir != 0)
            {
                lk_perm2d(handle, kernels[ikern++], p, b, tensors.w);
                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                }
            }
            lk_genidx2d(handle, kernels[ikern++], p, idx);
            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
            }

            lk_conv(handle, kernels[ikern], p, tensors.out, src, fil, idx, alpha);
            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}
} // namespace conv
} // namespace miopen
