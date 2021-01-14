#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/numeric.hpp>
#include <boost/any.hpp>

namespace miopen {
namespace conv {

float CallImplGemmDynamicForward(const miopen::Handle& handle,
                                 const ProblemDescription& conv_problem,
                                 ConstData_t src,
                                 Data_t dst,
                                 ConstData_t wei,
                                 const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());

    // clang-format off
    int hi          = conv_problem.GetInHeight();
    int wi          = conv_problem.GetInWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetOutChannels();
    int c           = conv_problem.GetInChannels();
    int ho          = conv_problem.GetOutHeight();
    int wo          = conv_problem.GetOutWidth();
    int stride_h    = conv_problem.GetKernelStrideH();
    int stride_w    = conv_problem.GetKernelStrideW();
    int dilation_h  = conv_problem.GetDilationH();
    int dilation_w  = conv_problem.GetDilationW();
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
    int y           = conv_problem.GetWeightsHeight();
    int x           = conv_problem.GetWeightsWidth();
    int __pack0     = 0;
    // clang-format on

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(wei);
    opArgs.emplace_back(dst);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(__pack0);

    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();
    return elapsed;
}

float CallImplGemmDynamicForward(const miopen::Handle& handle,
                                 const ProblemDescription& conv_problem,
                                 const solver::TunableImplicitGemmGTCDynamic_t& cfg,
                                 ConstData_t src,
                                 Data_t dst,
                                 ConstData_t wei,
                                 const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());

    // clang-format off
    int hi          = conv_problem.GetInHeight();
    int wi          = conv_problem.GetInWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetOutChannels();
    int c           = conv_problem.GetInChannels();
    int ho          = conv_problem.GetOutHeight();
    int wo          = conv_problem.GetOutWidth();
    int stride_h    = conv_problem.GetKernelStrideH();
    int stride_w    = conv_problem.GetKernelStrideW();
    int dilation_h  = conv_problem.GetDilationH();
    int dilation_w  = conv_problem.GetDilationW();
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
    int y           = conv_problem.GetWeightsHeight();
    int x           = conv_problem.GetWeightsWidth();
    int group       = conv_problem.GetGroupCount();
    int __pack0     = 0;
    // clang-format on

    int gemm_m =
        ((k / group + cfg.gemm_m_per_block - 1) / cfg.gemm_m_per_block) * cfg.gemm_m_per_block;
    int nxe = cfg.nxe;
    int nxb = cfg.nxb;
    int b =
        nxe == 0 ? (ho * wo) : ((ho * wo + nxb - 1) / nxb) * nxb; // pad to nxb modulo when nxe != 0

    // init magic division parameters
    uint32_t nb_n0          = cfg.tensor_b_cluster_lengths[2] * cfg.tensor_b_thread_lengths[2];
    uint32_t nb_n1b         = cfg.tensor_b_cluster_lengths[3] * cfg.tensor_b_thread_lengths[3];
    uint32_t unmerge_sub_n  = cfg.gemm_n_per_block / nxb;
    uint32_t unmerge_sub_n1 = unmerge_sub_n / nb_n0;

    magic_div_u32_t mdiv_0 = magic_div_u32_gen(gemm_m / cfg.gemm_m_per_block);
    magic_div_u32_t mdiv_1 = magic_div_u32_gen(b * unmerge_sub_n1 / nb_n1b);
    magic_div_u32_t mdiv_2 = magic_div_u32_gen(y * x);
    magic_div_u32_t mdiv_3 = magic_div_u32_gen(x);
    magic_div_u32_t mdiv_4 = magic_div_u32_gen(b);
    magic_div_u32_t mdiv_5 = magic_div_u32_gen(wo);
    magic_div_u32_t mdiv_6 =
        magic_div_u32_gen((n * b * (gemm_m)) / (cfg.gemm_m_per_block * cfg.gemm_n_per_block));

    uint32_t magic_0 = mdiv_0.magic;
    uint32_t magic_1 = mdiv_1.magic;
    uint32_t magic_2 = mdiv_2.magic;
    uint32_t magic_3 = mdiv_3.magic;
    uint32_t magic_4 = mdiv_4.magic;
    uint32_t magic_5 = mdiv_5.magic;
    uint32_t magic_6 = mdiv_6.magic;
    uint32_t shift_pack_0 =
        magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
    uint32_t shift_pack_1 = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, 0);

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(wei);
    opArgs.emplace_back(dst);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(group);
    opArgs.emplace_back(magic_0);
    opArgs.emplace_back(magic_1);
    opArgs.emplace_back(magic_2);
    opArgs.emplace_back(magic_3);
    opArgs.emplace_back(magic_4);
    opArgs.emplace_back(magic_5);
    opArgs.emplace_back(magic_6);
    opArgs.emplace_back(shift_pack_0);
    opArgs.emplace_back(shift_pack_1);
    opArgs.emplace_back(__pack0);

    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();
    return elapsed;
}

float CallImplGemmDynamicForward1x1(const miopen::Handle& handle,
                                    const ProblemDescription& conv_problem,
                                    ConstData_t src,
                                    Data_t dst,
                                    ConstData_t wei,
                                    const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());

    // clang-format off
    int hi          = conv_problem.GetInHeight();
    int wi          = conv_problem.GetInWidth();
    int n           = conv_problem.GetInBatchSize();
    int k           = conv_problem.GetOutChannels();
    int c           = conv_problem.GetInChannels();
    int ho          = conv_problem.GetOutHeight();
    int wo          = conv_problem.GetOutWidth();
    int stride_h    = conv_problem.GetKernelStrideH();
    int stride_w    = conv_problem.GetKernelStrideW();
    int dilation_h  = conv_problem.GetDilationH();
    int dilation_w  = conv_problem.GetDilationW();
    int pad_h       = conv_problem.GetPadH();
    int pad_w       = conv_problem.GetPadW();
    int __pack0     = 0;
    // clang-format on

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(wei);
    opArgs.emplace_back(dst);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(__pack0);

    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();
    return elapsed;
}

InvokerFactory MakeImplGemmDynamicForwardInvokerFactory(const ConvolutionContext& ctx)
{
    const auto& conv_problem = ctx.conv_problem;
    return [conv_problem](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            auto kernel             = handle.Run(kernels[0]);

            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k) { return handle.Run(k); });
            float elapsed = 0;
            elapsed       = CallImplGemmDynamicForward(
                handle, conv_problem, tensors.in, tensors.out, tensors.w, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory
MakeImplGemmDynamicForwardInvokerFactory(const ConvolutionContext& ctx,
                                         const solver::TunableImplicitGemmGTCDynamic_t& cfg)
{
    const auto& conv_problem = ctx.conv_problem;
    return [conv_problem, cfg](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            auto kernel             = handle.Run(kernels[0]);

            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k) { return handle.Run(k); });
            float elapsed = 0;
            elapsed       = CallImplGemmDynamicForward(
                handle, conv_problem, cfg, tensors.in, tensors.out, tensors.w, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeImplGemmDynamicForward1x1InvokerFactory(const ConvolutionContext& ctx)
{
    const auto& conv_problem = ctx.conv_problem;
    return [conv_problem](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            auto kernel             = handle.Run(kernels[0]);

            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k) { return handle.Run(k); });
            float elapsed = 0;
            elapsed       = CallImplGemmDynamicForward1x1(
                handle, conv_problem, tensors.in, tensors.out, tensors.w, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeImplGemmDynamicBackwardDataInvokerFactory(const ConvolutionContext& ctx)
{
    const auto& conv_problem = ctx.conv_problem;
    int hi                   = conv_problem.GetOutHeight();
    int wi                   = conv_problem.GetOutWidth();
    int n                    = conv_problem.GetInBatchSize();
    int k                    = conv_problem.GetInChannels();
    int c                    = conv_problem.GetOutChannels();
    int ho                   = conv_problem.GetInHeight();
    int wo                   = conv_problem.GetInWidth();
    int stride_h             = conv_problem.GetInHeight() > 1 ? conv_problem.GetKernelStrideH() : 1;
    int stride_w             = conv_problem.GetInWidth() > 1 ? conv_problem.GetKernelStrideW() : 1;
    int dilation_h = conv_problem.GetWeightsHeight() > 1 ? conv_problem.GetDilationH() : 1;
    int dilation_w = conv_problem.GetWeightsWidth() > 1 ? conv_problem.GetDilationW() : 1;
    int pad_h      = conv_problem.GetPadH();
    int pad_w      = conv_problem.GetPadW();
    int y          = conv_problem.GetWeightsHeight();
    int x          = conv_problem.GetWeightsWidth();

    int gcd_stride_dilation_h = gcd(stride_h, dilation_h);
    int gcd_stride_dilation_w = gcd(stride_w, dilation_w);
    int y_tilda               = stride_h / gcd_stride_dilation_h;
    int x_tilda               = stride_w / gcd_stride_dilation_w;

    int y_dot = (y + y_tilda - 1) / y_tilda;
    int x_dot = (x + x_tilda - 1) / x_tilda;

    int h_tilda = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
    int w_tilda = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

    int h_tilda_left = std::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
    int w_tilda_left = std::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

    int h_tilda_right = std::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
    int w_tilda_right = std::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

    int h_tilda_slice = h_tilda_right - h_tilda_left;
    int w_tilda_slice = w_tilda_right - w_tilda_left;

    int num_of_gemms = x_tilda * y_tilda;

    int dtile_dy      = dilation_h / gcd_stride_dilation_h;
    int dtile_dx      = dilation_w / gcd_stride_dilation_w;
    int dtile_y       = y_tilda;
    int dtile_x       = x_tilda;
    int dtile_h       = h_tilda;
    int dtile_w       = w_tilda;
    int dslice_h      = h_tilda_slice;
    int dslice_w      = w_tilda_slice;
    int dslice_h_left = h_tilda_left;
    int dslice_w_left = w_tilda_left;
    int pack_align    = 0;
    std::vector<int> dtile_iy_gid;
    std::vector<int> dtile_ix_gid;
    std::vector<int> y_dot_slice_gid;
    std::vector<int> x_dot_slice_gid;
    std::vector<bool> is_gemm_not_empty;
    for(int gemm_id = 0; gemm_id < num_of_gemms; gemm_id++)
    {
        dtile_iy_gid.emplace_back(gemm_id / x_tilda);
        dtile_ix_gid.emplace_back(gemm_id % x_tilda);
        y_dot_slice_gid.emplace_back((dtile_iy_gid[gemm_id] + 1) * y_dot <= y ? y_dot : y % y_dot);
        x_dot_slice_gid.emplace_back((dtile_ix_gid[gemm_id] + 1) * x_dot <= x ? x_dot : x % x_dot);
        const int gemm_k_gid = k * y_dot_slice_gid[gemm_id] * x_dot_slice_gid[gemm_id];
        is_gemm_not_empty.emplace_back(gemm_k_gid > 0);
    }
    bool need_set_zero = false;
    if(y < stride_h || x < stride_w || dilation_h != 1 || dilation_w != 1)
        need_set_zero = true;

    return [=](const std::vector<Kernel>& kernels) {
        const auto kernel = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            if(need_set_zero)
            {
                float zero = 0.f;
                SetTensor(handle, tensors.outDesc, tensors.out, &zero);

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }
            for(int gemm_id = 0; gemm_id < num_of_gemms; gemm_id++)
            {
                if(is_gemm_not_empty[gemm_id])
                {
                    handle.Run(kernel)(tensors.out,
                                       tensors.w,
                                       tensors.in,
                                       hi,
                                       wi,
                                       n,
                                       k,
                                       c,
                                       ho,
                                       wo,
                                       stride_h,
                                       stride_w,
                                       dilation_h,
                                       dilation_w,
                                       pad_h,
                                       pad_w,
                                       y,
                                       x,
                                       dtile_iy_gid[gemm_id],
                                       dtile_ix_gid[gemm_id],
                                       dtile_dy,
                                       dtile_dx,
                                       dtile_y,
                                       dtile_x,
                                       dtile_h,
                                       dtile_w,
                                       y_dot_slice_gid[gemm_id],
                                       x_dot_slice_gid[gemm_id],
                                       dslice_h,
                                       dslice_w,
                                       dslice_h_left,
                                       dslice_w_left,
                                       pack_align);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
            }
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

} // namespace conv
} // namespace miopen
