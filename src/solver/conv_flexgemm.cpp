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
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/idiv.hpp>
#include <boost/any.hpp>

namespace miopen {
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
    size_t s_pad;
    size_t s_idx;
    size_t s_perm;
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
static size_t get_auxbuf_size(const param_conv_t& p) { return (p.s_pad + p.s_perm + p.s_idx); }

static void lk_ufconv(const Handle& handle,
                      const Kernel& kern,
                      const param_ufconv_t& p,
                      void* c,
                      const void* a,
                      const void* b,
                      float alpha)
{
    handle.Run(kern)(a, b, p.ng, p.m, p.n, p.k, p.amag, p.cmag, c, alpha, p.dimx);
}
static void lk_padding2d(
    const Handle& handle, const Kernel& kern, const param_conv_t& p, void* dst, const void* src)
{
    handle.Run(kern)(dst, src, p.anx, p.any, p.pad, p.ng * p.inc, p.lda);
}
static void lk_perm2d(
    const Handle& handle, const Kernel& kern, const param_conv_t& p, void* dst, const void* src)
{
    uint32_t bnn   = p.bnx * p.bny;
    uint32_t lda   = p.n * bnn;
    uint32_t align = p.id != 3 ? 7 : 15;
    handle.Run(kern)(dst, src, lda, (p.n + 3) & ~3, p.k, p.n, bnn, (p.k + align) & ~align);
}
static void lk_genidx2d(const Handle& handle, const Kernel& kern, const param_conv_t& p, void* dst)
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
                    const param_conv_t& p,
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
InvokerFactory MakeFlexgemmInvokerFactory(const param_ufconv_t& p, float alpha)
{
    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& prim_params) {
            const auto& tensors = prim_params.CastTo<conv::DataInvokeParams>().tensors;
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
InvokerFactory MakeFlexgemmInvokerFactory(const param_conv_t& p, float alpha)
{
    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& prim_params) {
            const auto& params   = prim_params.CastTo<conv::DataInvokeParams>();
            const size_t auxsize = get_auxbuf_size(p);
            if(params.workSpace == nullptr || params.workSpaceSize < auxsize)
                MIOPEN_THROW("Workspace is not enough for flexgemm");
            const auto& tensors = params.tensors;
            float elapsed       = 0.f;
            uint8_t* a          = reinterpret_cast<uint8_t*>(params.workSpace);
            uint8_t* b          = a + p.s_pad;
            uint8_t* idx        = b + p.s_perm;
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

namespace solver {
static size_t get_auxbuf_size(const ConvolutionContext& ctx)
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
    temp          = id != 3 ? 7 : 15;
    size_t ags    = lda * inc;
    size_t s_pad  = static_cast<size_t>(ng << 2) * ags;
    size_t s_perm = static_cast<size_t>(ng << 2) * ((k + temp) & ~temp) * ((n + 3) & ~3);
    size_t s_idx  = (ntidx << 3) + (((k + 15) & ~15) << 2) + 128;
    s_pad         = (pu | pv) == 0 ? 0 : s_pad;
    s_perm        = ctx.direction.IsForward() ? 0 : s_perm;
    return (s_pad + s_perm + s_idx);
}

static void build_params_ufconv(param_ufconv_t& p, const ConvolutionContext& ctx)
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

static void build_params_conv(param_conv_t& p, const ConvolutionContext& ctx)
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
    temp     = p.id != 3 ? 7 : 15;
    p.ags    = p.lda * p.inc;
    p.s_pad  = static_cast<size_t>(p.ng << 2) * p.ags;
    p.s_perm = static_cast<size_t>(p.ng << 2) * ((p.k + temp) & ~temp) * ((p.n + 3) & ~3);
    p.s_idx  = (p.ntidx << 3) + (((p.k + 15) & ~15) << 2) + 128;
    p.s_pad  = p.pad == 0 ? 0 : p.s_pad;
    p.s_perm = p.dir == 0 ? 0 : p.s_perm;
}

static void fill_kernels_info(ConvSolution& sol,
                              const ConvolutionContext& ctx,
                              const param_ufconv_t& p,
                              const std::string& fname,
                              uint32_t relu)
{
    static const char* knames_ufco[] = {
        "suffco7x4_om", "suffco7x4_om_relu", "suffco7x4_dm", "suffco7x4_dm_relu",
        "suffco7x4_qm", "suffco7x4_qm_relu", "suffco8x5_om", "suffco8x5_om_relu",
        "suffco8x5_dm", "suffco8x5_dm_relu", "suffco8x5_qm", "suffco8x5_qm_relu",
        "suffco8x6_om", "suffco8x6_om_relu", "suffco8x6_dm", "suffco8x6_dm_relu",
        "suffco8x6_qm", "suffco8x6_qm_relu", "suffco7x7_om", "suffco7x7_om_relu",
        "suffco7x7_dm", "suffco7x7_dm_relu", "suffco7x7_qm", "suffco7x7_qm_relu",
        "sufbco7x4_om", "sufbco7x4_dm",      "sufbco7x4_qm", "sufbco8x5_om",
        "sufbco8x5_dm", "sufbco8x5_qm",      "sufbco8x6_om", "sufbco8x6_dm",
        "sufbco8x6_qm", "sufbco7x7_om",      "sufbco7x7_dm", "sufbco7x7_qm"};
    const uint32_t id   = p.id & 0xffff;
    const uint32_t mode = p.id >> 16;
    const uint32_t blk  = id > 0 ? 256 : 128;
    const uint32_t shx  = (id > 0) && (id < 3) ? 8 : 7;
    const uint32_t shy  = 4 + id;
    const uint32_t gdy  = (p.n + (1 << shy) - 1) >> shy;
    const uint32_t gdx  = p.ntidx >> shx;
    const uint32_t routine =
        p.dir * 24 + id * (3 * (1 + (1 ^ p.dir))) + ((mode << (1 ^ p.dir)) | relu);
    const std::vector<size_t> block{blk, 1, 1};
    const std::vector<size_t> grid{gdy * blk, gdx, p.ng};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    KernelInfo kinfo = {options.str(), block, grid, fname, knames_ufco[routine]};
    sol.construction_params.push_back(kinfo);
}
static void fill_kernels_info(ConvSolution& sol,
                              const ConvolutionContext& ctx,
                              const param_conv_t& p,
                              const std::string& fname,
                              uint32_t relu)
{
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    if(p.pad != 0)
    {
        const uint32_t gdx = (p.pnx * p.pny * p.ng * p.inc + 255) >> 8;
        const std::vector<size_t> block{256, 1, 1};
        const std::vector<size_t> grid{gdx << 8, p.bs, 1};
        KernelInfo kinfo = {options.str(), block, grid, fname, "padding2d"};
        sol.construction_params.push_back(kinfo);
    }

    if(p.dir != 0)
    {
        const uint32_t gdx = (((p.n + 3) & ~3) + 63) >> 6;
        const std::vector<size_t> block{64, 1, 1};
        const std::vector<size_t> grid{gdx << 6, p.inc, p.ng};
        KernelInfo kinfo = {options.str(), block, grid, fname, "perm2d_flip"};
        sol.construction_params.push_back(kinfo);
    }

    {
        const uint32_t srelo = ((p.k + 15) & ~15) + 32;
        const uint32_t ntid  = p.ntidx > srelo ? p.ntidx : srelo;
        const uint32_t gdx   = (ntid + 63) >> 6;
        const std::vector<size_t> block{64, 1, 1};
        const std::vector<size_t> grid{gdx << 6, 1, 1};
        KernelInfo kinfo = {options.str(), block, grid, fname, "genidx2d"};
        sol.construction_params.push_back(kinfo);
    }

    {
        static const char* knames_co[] = {"sfco7x4",
                                          "sfco7x4_relu",
                                          "sfco8x5",
                                          "sfco8x5_relu",
                                          "sfco8x6",
                                          "sfco8x6_relu",
                                          "sfco7x7",
                                          "sfco7x7_relu",
                                          "sfco",
                                          "sfco_relu",
                                          "sbco7x4",
                                          "sbco8x5",
                                          "sbco8x6",
                                          "sbco7x7"};
        const uint32_t routine = p.dir == 0 ? ((p.id << 1) | relu) : (10 + p.id);
        const uint32_t blk     = p.id == 0 ? 128 : 256;
        const uint32_t shx     = (0x87887 >> (p.id << 2)) & 0xf;
        const uint32_t shy     = (0x57654 >> (p.id << 2)) & 0xf;
        const uint32_t ngx     = p.ntidx >> shx;
        const uint32_t ngy     = (p.n + (1 << shy) - 1) >> shy;
        const uint32_t gdx     = p.pad != 0 ? ngx : ngy;
        const uint32_t gdy     = p.pad != 0 ? ngy : ngx;
        const std::vector<size_t> block{blk, 1, 1};
        const std::vector<size_t> grid{gdx * blk, gdy, p.ng};
        KernelInfo kinfo = {options.str(), block, grid, fname, knames_co[routine]};
        sol.construction_params.push_back(kinfo);
    }
}

bool ConvFlexgemm::IsApplicable(const ConvolutionContext& ctx) const
{
    if(MIOPEN_BACKEND_OPENCL)
        return false;
    const auto name = ctx.GetStream().GetDeviceName();
    if((name != "gfx900" && name != "gfx906") || ctx.direction.IsBackwardWrW())
        return false;
    uint32_t pad      = ctx.pad_w | ctx.pad_h | ctx.pad_d;
    uint32_t ksize    = ctx.kernel_size_w | ctx.kernel_size_h | ctx.kernel_size_d;
    uint32_t stride   = ctx.kernel_stride_w | ctx.kernel_stride_h | ctx.kernel_stride_d;
    uint32_t dilation = ctx.kernel_dilation_w | ctx.kernel_dilation_h | ctx.kernel_dilation_d;
    if(((ksize | stride | dilation) == 1) && (pad == 0) &&
       (((ctx.n_inputs / ctx.group_counts) & 7) == 0))
        return true;
    if(!ctx.Is2d())
        return false;
    int pu = ctx.pad_w;
    int pv = ctx.pad_h;
    int su = ctx.kernel_stride_w;
    int sv = ctx.kernel_stride_h;
    int du = ctx.kernel_dilation_w;
    int dv = ctx.kernel_dilation_h;
    if(ctx.direction.IsForward())
    {
        if(((ctx.in_width + (pu << 1)) < ctx.kernel_size_w) ||
           ((ctx.in_height + (pv << 1)) < ctx.kernel_size_h))
            return false;
        if((((ctx.in_width + (pu << 1) - du * (ctx.kernel_size_w - 1) - 1) / su + 1) <= 0) ||
           (((ctx.in_height + (pv << 1) - dv * (ctx.kernel_size_h - 1) - 1) / sv + 1) <= 0))
            return false;
    }
    else
    {
        pu = ctx.kernel_size_w - pu - 1;
        pv = ctx.kernel_size_h - pv - 1;
        if(((stride | dilation) != 1) || (pu < 0) || (pv < 0))
            return false;
    }
    return (ctx.IsFp32() && (ctx.in_layout == "NCHW") && (ctx.bias == 0));
}
size_t ConvFlexgemm::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    uint32_t pad      = ctx.pad_w | ctx.pad_h | ctx.pad_d;
    uint32_t ksize    = ctx.kernel_size_w | ctx.kernel_size_h | ctx.kernel_size_d;
    uint32_t stride   = ctx.kernel_stride_w | ctx.kernel_stride_h | ctx.kernel_stride_d;
    uint32_t dilation = ctx.kernel_dilation_w | ctx.kernel_dilation_h | ctx.kernel_dilation_d;
    if(((ksize | stride | dilation) == 1) && (pad == 0) &&
       (((ctx.n_inputs / ctx.group_counts) & 7) == 0))
        return 0;
    return get_auxbuf_size(ctx);
}
ConvSolution ConvFlexgemm::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution sol{};
    const std::string fname = "flexgemm_" + ctx.GetStream().GetDeviceName() + ".s";
    uint32_t pad            = ctx.pad_w | ctx.pad_h | ctx.pad_d;
    uint32_t ksize          = ctx.kernel_size_w | ctx.kernel_size_h | ctx.kernel_size_d;
    uint32_t stride         = ctx.kernel_stride_w | ctx.kernel_stride_h | ctx.kernel_stride_d;
    uint32_t dilation       = ctx.kernel_dilation_w | ctx.kernel_dilation_h | ctx.kernel_dilation_d;
    if(((ksize | stride | dilation) == 1) && (pad == 0) &&
       (((ctx.n_inputs / ctx.group_counts) & 7) == 0))
    {
        param_ufconv_t params{};
        build_params_ufconv(params, ctx);
        sol.workspce_sz = 0;
        fill_kernels_info(sol, ctx, params, fname, 0);
        sol.invoker_factory = conv::MakeFlexgemmInvokerFactory(params, 1.f);
    }
    else
    {
        param_conv_t params{};
        build_params_conv(params, ctx);
        sol.workspce_sz = params.s_pad + params.s_perm + params.s_idx;
        fill_kernels_info(sol, ctx, params, fname, 0);
        sol.invoker_factory = conv::MakeFlexgemmInvokerFactory(params, 1.f);
    }
    return sol;
}
} // namespace solver
} // namespace miopen
