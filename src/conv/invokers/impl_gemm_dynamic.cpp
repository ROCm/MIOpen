#include <miopen/buffer_info.hpp>
#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/batched_transpose_sol.hpp>
#include <boost/any.hpp>

namespace miopen {
namespace conv {

static inline uint32_t igemm_find_tile_size_with_upper_bound(
    uint32_t out_size, size_t upper_bound, uint32_t stride, uint32_t dilation, uint32_t filter)
{
    // return tile size so that the required input tile(sec_in) is no larger than upper_bound
    uint32_t n_tiles = 1;
    if(n_tiles <= out_size)
    {
        for(; n_tiles <= out_size; n_tiles++)
        {
            uint32_t tile_size = (out_size + n_tiles - 1) / n_tiles;
            uint32_t sec_in    = (tile_size - 1) * stride + 1 + dilation * (filter - 1);
            if(sec_in <= upper_bound)
                break;
        }
    }
    else
        MIOPEN_THROW("out_size should not be less than one");

    return (out_size + n_tiles - 1) / n_tiles;
}

static float CallImplGemmDynamicForward1x1(const miopen::Handle& handle,
                                           const ProblemDescription& problem,
                                           ConstData_t src,
                                           Data_t dst,
                                           ConstData_t wei,
                                           const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    MIOPEN_LOG_I(kernel.GetName());

    // clang-format off
    int hi          = problem.GetInHeight();
    int wi          = problem.GetInWidth();
    int n           = problem.GetInBatchSize();
    int k           = problem.GetOutChannels();
    int c           = problem.GetInChannels();
    int ho          = problem.GetOutHeight();
    int wo          = problem.GetOutWidth();
    int stride_h    = problem.GetKernelStrideH();
    int stride_w    = problem.GetKernelStrideW();
    int dilation_h  = problem.GetDilationH();
    int dilation_w  = problem.GetDilationW();
    int pad_h       = problem.GetPadH();
    int pad_w       = problem.GetPadW();
    int gap_0     = 0;
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
    opArgs.emplace_back(gap_0);

    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();
    return elapsed;
}

InvokerFactory MakeImplGemmDynamicForward1x1InvokerFactory(const ProblemDescription& problem)
{
    return [problem](const std::vector<Kernel>& kernels) {
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
                handle, problem, tensors.in, tensors.out, tensors.w, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeImplGemmDynamicBackwardDataInvokerFactory(const ProblemDescription& problem,
                                                             const int cfg)
{
    int hi         = problem.GetOutHeight();
    int wi         = problem.GetOutWidth();
    int n          = problem.GetInBatchSize();
    int k          = problem.GetInChannels();
    int c          = problem.GetOutChannels();
    int ho         = problem.GetInHeight();
    int wo         = problem.GetInWidth();
    int stride_h   = problem.GetInHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int stride_w   = problem.GetInWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dilation_h = problem.GetWeightsHeight() > 1 ? problem.GetDilationH() : 1;
    int dilation_w = problem.GetWeightsWidth() > 1 ? problem.GetDilationW() : 1;
    int pad_h      = problem.GetPadH();
    int pad_w      = problem.GetPadW();
    int y          = problem.GetWeightsHeight();
    int x          = problem.GetWeightsWidth();

    int gcd_stride_dilation_h = solver::gcd(stride_h, dilation_h);
    int gcd_stride_dilation_w = solver::gcd(stride_w, dilation_w);
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
    int pack_align    = cfg;
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
    bool need_set_zero = true;

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

InvokerFactory
MakeImplGemmDynamicBackwardDataInvokerFactory(const ProblemDescription& problem,
                                              const solver::TunableImplicitGemmGTCDynamic_t& cfg)
{
    int hi         = problem.GetOutHeight();
    int wi         = problem.GetOutWidth();
    int n          = problem.GetInBatchSize();
    int k          = problem.GetInChannels();
    int c          = problem.GetOutChannels();
    int ho         = problem.GetInHeight();
    int wo         = problem.GetInWidth();
    int stride_h   = problem.GetOutHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int stride_w   = problem.GetOutWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dilation_h = problem.GetWeightsHeight() > 1 ? problem.GetDilationH() : 1;
    int dilation_w = problem.GetWeightsWidth() > 1 ? problem.GetDilationW() : 1;
    int pad_h      = problem.GetPadH();
    int pad_w      = problem.GetPadW();
    int y          = problem.GetWeightsHeight();
    int x          = problem.GetWeightsWidth();
    int group      = problem.GetGroupCount();

    int gcd_stride_dilation_h = solver::gcd(stride_h, dilation_h);
    int gcd_stride_dilation_w = solver::gcd(stride_w, dilation_w);
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
    bool need_set_zero = true;
    int nxb            = cfg.nxb;
    int b              = h_tilda_slice * w_tilda_slice;
    b = (cfg.nxe == 0) ? (b) : ((b + nxb - 1) / nxb) * nxb; // pad to nxb modulo when nxe != 0

    uint32_t nb_n0          = cfg.tensor_b_cluster_lengths[2] * cfg.tensor_b_thread_lengths[2];
    uint32_t nb_n1b         = cfg.tensor_b_cluster_lengths[3] * cfg.tensor_b_thread_lengths[3];
    uint32_t unmerge_sub_n  = cfg.gemm_n_per_block / cfg.nxb;
    uint32_t unmerge_sub_n1 = unmerge_sub_n / nb_n0;

    magic_div_u32_t mdiv_2 =
        magic_div_u32_gen(((c / group) * n * b) / (cfg.gemm_m_per_block * cfg.gemm_n_per_block));
    magic_div_u32_t mdiv_3 = magic_div_u32_gen((n * b) / cfg.gemm_n_per_block);
    magic_div_u32_t mdiv_4 = magic_div_u32_gen(b * unmerge_sub_n1 / nb_n1b);
    magic_div_u32_t mdiv_5 = magic_div_u32_gen(b);
    magic_div_u32_t mdiv_6 = magic_div_u32_gen(w_tilda_slice);

    std::vector<magic_div_u32_t> mdiv_0_vec;
    std::vector<magic_div_u32_t> mdiv_1_vec;
    std::vector<uint32_t> shift_pack_0_vec;
    uint32_t shift_pack_1;

    for(int gemm_id = 0; gemm_id < num_of_gemms; gemm_id++)
    {
        if(is_gemm_not_empty[gemm_id])
        {
            mdiv_0_vec.push_back(
                magic_div_u32_gen(y_dot_slice_gid[gemm_id] * x_dot_slice_gid[gemm_id]));
            mdiv_1_vec.push_back(magic_div_u32_gen(x_dot_slice_gid[gemm_id]));
        }
        else
        {
            mdiv_0_vec.push_back(magic_div_u32_t({0, 0}));
            mdiv_1_vec.push_back(magic_div_u32_t({0, 0}));
        };

        shift_pack_0_vec.push_back(magic_div_u32_pack_shift(
            mdiv_0_vec[gemm_id].shift, mdiv_1_vec[gemm_id].shift, mdiv_2.shift, mdiv_3.shift));
    };

    shift_pack_1 = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, 0);

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
                                       k / group,
                                       c / group,
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
                                       group,
                                       mdiv_0_vec[gemm_id].magic,
                                       mdiv_1_vec[gemm_id].magic,
                                       mdiv_2.magic,
                                       mdiv_3.magic,
                                       mdiv_4.magic,
                                       mdiv_5.magic,
                                       mdiv_6.magic,
                                       shift_pack_0_vec[gemm_id],
                                       shift_pack_1,
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

InvokerFactory MakeImplGemmDynamicForwardXdlopsNHWCInvokerFactory(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const solver::conv::PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC& config)
{
    int hi         = problem.GetInHeight();
    int wi         = problem.GetInWidth();
    int n          = problem.GetInBatchSize();
    int k          = problem.GetOutChannels();
    int c          = problem.GetInChannels();
    int ho         = problem.GetOutHeight();
    int wo         = problem.GetOutWidth();
    int stride_h   = problem.GetKernelStrideH();
    int stride_w   = problem.GetKernelStrideW();
    int dilation_h = problem.GetDilationH();
    int dilation_w = problem.GetDilationW();
    int pad_h      = problem.GetPadH();
    int pad_w      = problem.GetPadW();
    int y          = problem.GetWeightsHeight();
    int x          = problem.GetWeightsWidth();
    int group      = problem.GetGroupCount();
    int c_karg     = c / group;
    int y_karg     = y;
    int x_karg     = x;

    int splits_4G = solver::igemm_split_batch_size(
        hi, wi, ho, wo, n, k, c, miopen::GetTypeSize(problem.GetInDataType()));
    splits_4G = splits_4G == 0 ? n : splits_4G;

    uint32_t gemm_m = (n / splits_4G) * ho * wo;
    uint32_t gemm_n = k / group;
    magic_div_u32_t mdiv_0, mdiv_1, mdiv_2, mdiv_3, mdiv_4, mdiv_5;
    uint32_t shift_pack_0, shift_pack_1;
    uint32_t pack0 = 0;

    mdiv_0 = magic_div_u32_gen((gemm_n + config.gemm_n_per_block - 1) / config.gemm_n_per_block);
    mdiv_1 = magic_div_u32_gen(ho * wo);
    mdiv_2 = magic_div_u32_gen(wo);
    mdiv_3 = magic_div_u32_gen(((gemm_m + config.gemm_m_per_block - 1) / config.gemm_m_per_block) *
                               ((gemm_n + config.gemm_n_per_block - 1) / config.gemm_n_per_block));

    shift_pack_0 = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
    if(config.merge_e != 0)
    {
        mdiv_4       = magic_div_u32_gen(x * (c / group));
        mdiv_5       = magic_div_u32_gen(c / group);
        shift_pack_1 = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, 0, 0);

        uint32_t s_move_slice_k_y = (config.gemm_k_per_block / (x * (c / group))) % y;
        uint32_t s_move_slice_k_x = (config.gemm_k_per_block / (c / group)) % x;
        uint32_t s_move_slice_k_c = config.gemm_k_per_block % (c / group);
        y_karg                    = static_cast<int>((s_move_slice_k_y << 24) | y);
        x_karg                    = static_cast<int>((s_move_slice_k_x << 24) | x);
        c_karg                    = static_cast<int>((s_move_slice_k_c << 24) | (c / group));
    }
    else
    {
        mdiv_4       = magic_div_u32_gen(1);
        mdiv_5       = magic_div_u32_gen(1);
        shift_pack_1 = 0;
    }

    // Clear buffer for all condition to resolve the NaN issue.
    bool need_set_zero                 = true;
    bool use_fp32_global_split_on_fp16 = config.vector_store == 1 && config.gemm_k_global_split > 0;

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n / splits_4G);
    opArgs.emplace_back(k / group);
    opArgs.emplace_back(c_karg);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y_karg);
    opArgs.emplace_back(x_karg);
    opArgs.emplace_back(group);
    opArgs.emplace_back(mdiv_0.magic);
    opArgs.emplace_back(mdiv_1.magic);
    opArgs.emplace_back(mdiv_2.magic);
    opArgs.emplace_back(mdiv_3.magic);
    opArgs.emplace_back(mdiv_4.magic);
    opArgs.emplace_back(mdiv_5.magic);
    opArgs.emplace_back(shift_pack_0);
    opArgs.emplace_back(shift_pack_1);
    opArgs.emplace_back(config.gemm_k_global_split);
    opArgs.emplace_back(pack0);

    std::vector<std::vector<OpKernelArg>> opArgsTrans;

    const auto lowp_quant = problem.GetConv().lowp_quant;
    const auto isGfx90aFp16altSupport =
        (ctx.GetStream().GetDeviceName() == "gfx90a") && problem.IsFp16();

    const bool need_cast = [&]() {
        if(problem.GetOut().GetType() == miopenHalf)
            return use_fp32_global_split_on_fp16;
        if(problem.GetOut().GetType() == miopenBFloat16)
            return config.gemm_k_global_split > 0;
        return false;
    }();
    const auto is_nchw = problem.IsLayoutDefault();

    size_t trans_input_offset = 0;
    size_t trans_input_size   = 0;

    size_t trans_weight_offset = 0;
    size_t trans_weight_size   = 0;

    size_t trans_output_offset = 0;
    size_t trans_output_size   = 0;

    bool trans_input_skippable  = false;
    bool trans_weight_skippable = false;
    bool trans_output_skippable = false;

    int trans_input_idx  = -1;
    int trans_weight_idx = -1;
    int trans_output_idx = -1;

    if(is_nchw)
    {
        TransposeSolutionDefault2Nhwc trans_input(ctx, problem.GetInDataType(), n, c, hi, wi);
        TransposeSolutionDefault2Nhwc trans_weight(ctx,
                                                   problem.GetWeightsDataType(),
                                                   k,
                                                   c / group,
                                                   y,
                                                   x); // group * k_per_group as batch for weight
        TransposeSolutionNhwc2Default trans_output(ctx, problem.GetOutDataType(), n, k, ho, wo);

        trans_input_skippable  = trans_input.IsSkippable();
        trans_weight_skippable = trans_weight.IsSkippable();
        trans_output_skippable = trans_output.IsSkippable();

        if(!trans_input_skippable)
            opArgsTrans.emplace_back(trans_input.GetKernelArg());
        if(!trans_weight_skippable)
            opArgsTrans.emplace_back(trans_weight.GetKernelArg());
        if(!trans_output_skippable)
            opArgsTrans.emplace_back(trans_output.GetKernelArg());

        trans_input_size  = trans_input_skippable ? 0 : trans_input.GetOutputTensorSize();
        trans_weight_size = trans_weight_skippable ? 0 : trans_weight.GetOutputTensorSize();
        trans_output_size = trans_output_skippable ? 0 : trans_output.GetOutputTensorSize();

        int idx = 0;
        if(!trans_input_skippable)
            trans_input_idx = idx++;
        if(!trans_weight_skippable)
            trans_weight_idx = idx++;
        if(!trans_output_skippable)
            trans_output_idx = idx++;
    }

    const size_t cast_size = need_cast ? miopen::GetTypeSize(miopenFloat) * n * k * ho * wo : 0;

    MultiBufferWorkspaceTraits wt(
        {trans_input_size, trans_weight_size, trans_output_size, cast_size});

    trans_input_offset  = wt.GetOffset(0);
    trans_weight_offset = wt.GetOffset(1);
    trans_output_offset = wt.GetOffset(2);

    const size_t cast_offset = wt.GetOffset(3);

    const int kID_trans_start = isGfx90aFp16altSupport ? 2 : 1;

    const TensorDescriptor cast_desc(
        miopenFloat, problem.GetOut().GetLengths(), problem.GetOut().GetStrides());
    auto null_buf = shared<Data_t>{};

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            const auto& workSpace   = data_ctx.workSpace;
            const auto ker =
                handle.Run(kernels[(isGfx90aFp16altSupport && data_ctx.gfx90aFp16alt) ? 1 : 0]);
            float elapsed = 0;

            auto trans_input_buf =
                trans_input_size == 0
                    ? null_buf
                    : handle.CreateSubBuffer(workSpace, trans_input_offset, trans_input_size);
            auto trans_weight_buf =
                trans_weight_size == 0
                    ? null_buf
                    : handle.CreateSubBuffer(workSpace, trans_weight_offset, trans_weight_size);
            auto trans_output_buf =
                trans_output_size == 0
                    ? null_buf
                    : handle.CreateSubBuffer(workSpace, trans_output_offset, trans_output_size);
            auto cast_buf = cast_size == 0
                                ? null_buf
                                : handle.CreateSubBuffer(workSpace, cast_offset, cast_size);

            if(need_set_zero)
            {
                auto zero_buf = need_cast
                                    ? cast_buf.get()
                                    : ((is_nchw && !trans_output_skippable) ? trans_output_buf.get()
                                                                            : tensors.out);
                auto& zero_desc =
                    need_cast
                        ? cast_desc
                        : tensors.outDesc; // use the same desc for NCHW/NHWC for this dense tensor
                float zero = 0.f;

                SetTensor(handle, zero_desc, zero_buf, &zero);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            if(is_nchw)
            {
                if(!trans_input_skippable)
                {
                    auto& karg_input = opArgsTrans[trans_input_idx];
                    karg_input[0]    = OpKernelArg(trans_input_buf.get());
                    karg_input[1]    = OpKernelArg(tensors.in);
                    handle.Run(kernels[kID_trans_start + trans_input_idx])(karg_input);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                if(!trans_weight_skippable)
                {
                    auto& karg_weight = opArgsTrans[trans_weight_idx];
                    karg_weight[0]    = OpKernelArg(trans_weight_buf.get());
                    karg_weight[1]    = OpKernelArg(tensors.w);
                    handle.Run(kernels[kID_trans_start + trans_weight_idx])(karg_weight);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
            }

            opArgs[0] = (is_nchw && !trans_input_skippable) ? OpKernelArg(trans_input_buf.get())
                                                            : OpKernelArg(tensors.in);
            opArgs[1] = (is_nchw && !trans_weight_skippable) ? OpKernelArg(trans_weight_buf.get())
                                                             : OpKernelArg(tensors.w);

            opArgs[2] = need_cast ? OpKernelArg(cast_buf.get())
                                  : ((is_nchw && !trans_output_skippable)
                                         ? OpKernelArg(trans_output_buf.get())
                                         : OpKernelArg(tensors.out));
            ker(opArgs);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();

            if(need_cast)
            {
                CastTensor(handle,
                           &lowp_quant,
                           problem.IsDirectionForward(),
                           cast_desc,
                           cast_buf.get(),
                           tensors.outDesc,
                           (is_nchw && !trans_output_skippable) ? trans_output_buf.get()
                                                                : tensors.out,
                           0,
                           0);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            if(is_nchw && !trans_output_skippable)
            {
                auto& karg_output = opArgsTrans[trans_output_idx];
                karg_output[0]    = OpKernelArg(tensors.out);
                karg_output[1]    = OpKernelArg(trans_output_buf.get());
                handle.Run(kernels[kID_trans_start + trans_output_idx])(karg_output);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeImplGemmDynamicBackwardDataXdlopsNHWCInvokerFactory(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const solver::conv::PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC& config)
{
    int hi         = problem.GetOutHeight();
    int wi         = problem.GetOutWidth();
    int n          = problem.GetInBatchSize();
    int k          = problem.GetInChannels();
    int c          = problem.GetOutChannels();
    int ho         = problem.GetInHeight();
    int wo         = problem.GetInWidth();
    int stride_h   = problem.GetOutHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int stride_w   = problem.GetOutWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dilation_h = problem.GetWeightsHeight() > 1 ? problem.GetDilationH() : 1;
    int dilation_w = problem.GetWeightsWidth() > 1 ? problem.GetDilationW() : 1;
    int pad_h      = problem.GetPadH();
    int pad_w      = problem.GetPadW();
    int y          = problem.GetWeightsHeight();
    int x          = problem.GetWeightsWidth();
    int group      = problem.GetGroupCount();

    int gcd_stride_dilation_h = solver::gcd(stride_h, dilation_h);
    int gcd_stride_dilation_w = solver::gcd(stride_w, dilation_w);
    int y_tilda               = stride_h / gcd_stride_dilation_h;
    int x_tilda               = stride_w / gcd_stride_dilation_w;

    int h_tilda = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
    int w_tilda = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

    int h_tilda_left = std::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
    int w_tilda_left = std::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

    int h_tilda_right = std::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
    int w_tilda_right = std::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

    int h_tilda_slice = h_tilda_right - h_tilda_left;
    int w_tilda_slice = w_tilda_right - w_tilda_left;

    int num_of_gemms = x_tilda * y_tilda;

    int splits_4G = solver::igemm_split_batch_size(
        hi, wi, ho, wo, n, k, c, miopen::GetTypeSize(problem.GetInDataType()));
    int n_in_1_block = splits_4G == 0 ? 1 : (n / splits_4G);

    uint32_t gemm_m = n_in_1_block * h_tilda_slice * w_tilda_slice;
    uint32_t gemm_n = c / group;

    magic_div_u32_t mdiv_x_tilda  = magic_div_u32_gen(x_tilda);
    magic_div_u32_t mdiv_y_tilda  = magic_div_u32_gen(y_tilda);
    magic_div_u32_t mdiv_group_mn = magic_div_u32_gen(
        group * ((gemm_n + config.gemm_n_per_block - 1) / config.gemm_n_per_block) *
        ((gemm_m + config.gemm_m_per_block - 1) / config.gemm_m_per_block));

    magic_div_u32_t mdiv_0 =
        magic_div_u32_gen((gemm_n + config.gemm_n_per_block - 1) / config.gemm_n_per_block);
    magic_div_u32_t mdiv_1 =
        magic_div_u32_gen(((gemm_n + config.gemm_n_per_block - 1) / config.gemm_n_per_block) *
                          ((gemm_m + config.gemm_m_per_block - 1) / config.gemm_m_per_block));
    magic_div_u32_t mdiv_2 = magic_div_u32_gen(config.nxe != 0 ? w_tilda_slice : wi);
    magic_div_u32_t mdiv_3 = magic_div_u32_gen(h_tilda_slice * w_tilda_slice);
    uint32_t shift_pack_0 =
        magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);

    int dtile_iy = num_of_gemms > 1 ? static_cast<int>(mdiv_x_tilda.magic) : 0;
    int dtile_ix = num_of_gemms > 1 ? static_cast<int>(mdiv_x_tilda.shift) : 0;
    int dslice_y = num_of_gemms > 1 ? static_cast<int>(mdiv_y_tilda.magic) : y;
    int dslice_x = num_of_gemms > 1 ? static_cast<int>(mdiv_y_tilda.shift) : x;
    int dtile_h  = num_of_gemms > 1 ? static_cast<int>(mdiv_group_mn.magic) : h_tilda;
    int dtile_w  = num_of_gemms > 1 ? static_cast<int>(mdiv_group_mn.shift) : w_tilda;

    bool need_set_zero                 = true;
    bool use_fp32_global_split_on_fp16 = config.vector_store == 1 && config.gemm_k_global_split > 0;

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n_in_1_block);
    opArgs.emplace_back(k / group);
    opArgs.emplace_back(c / group);
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

    opArgs.emplace_back(dtile_iy);
    opArgs.emplace_back(dtile_ix);
    opArgs.emplace_back(dilation_h / gcd_stride_dilation_h);
    opArgs.emplace_back(dilation_w / gcd_stride_dilation_w);
    opArgs.emplace_back(y_tilda);
    opArgs.emplace_back(x_tilda);
    opArgs.emplace_back(dtile_h);
    opArgs.emplace_back(dtile_w);
    opArgs.emplace_back(dslice_y);
    opArgs.emplace_back(dslice_x);

    opArgs.emplace_back(h_tilda_slice);
    opArgs.emplace_back(w_tilda_slice);
    opArgs.emplace_back(h_tilda_left);
    opArgs.emplace_back(w_tilda_left);
    opArgs.emplace_back(group);

    opArgs.emplace_back(mdiv_0.magic);
    opArgs.emplace_back(mdiv_1.magic);
    opArgs.emplace_back(mdiv_2.magic);
    opArgs.emplace_back(mdiv_3.magic);
    opArgs.emplace_back(shift_pack_0);
    opArgs.emplace_back(config.gemm_k_global_split);

    std::vector<std::vector<OpKernelArg>> opArgsTrans;

    const auto lowp_quant = problem.GetConv().lowp_quant;
    const auto isGfx90aFp16altSupport =
        (ctx.GetStream().GetDeviceName() == "gfx90a") && problem.IsFp16();
    const bool need_cast = [&]() {
        if(problem.GetOut().GetType() == miopenHalf)
            return use_fp32_global_split_on_fp16;
        if(problem.GetOut().GetType() == miopenBFloat16)
        {
            return (y < stride_h || x < stride_w || dilation_h != 1 || dilation_w != 1 ||
                    config.gemm_k_global_split > 0);
        }
        return false;
    }();
    const auto is_nchw = problem.IsLayoutDefault();

    size_t trans_input_offset = 0;
    size_t trans_input_size   = 0;

    size_t trans_weight_offset = 0;
    size_t trans_weight_size   = 0;

    size_t trans_output_offset = 0;
    size_t trans_output_size   = 0;

    bool trans_input_skippable  = false;
    bool trans_weight_skippable = false;
    bool trans_output_skippable = false;

    int trans_input_idx  = -1;
    int trans_weight_idx = -1;
    int trans_output_idx = -1;

    if(is_nchw)
    {
        TransposeSolutionNhwc2Default trans_input(ctx, problem.GetOutDataType(), n, c, hi, wi);
        TransposeSolutionDefault2Nhwc trans_weight(ctx,
                                                   problem.GetWeightsDataType(),
                                                   k,
                                                   c / group,
                                                   y,
                                                   x); // group * k_per_group as batch for weight
        TransposeSolutionDefault2Nhwc trans_output(ctx, problem.GetInDataType(), n, k, ho, wo);

        trans_input_skippable  = trans_input.IsSkippable();
        trans_weight_skippable = trans_weight.IsSkippable();
        trans_output_skippable = trans_output.IsSkippable();

        if(!trans_input_skippable)
            opArgsTrans.emplace_back(trans_input.GetKernelArg());
        if(!trans_weight_skippable)
            opArgsTrans.emplace_back(trans_weight.GetKernelArg());
        if(!trans_output_skippable)
            opArgsTrans.emplace_back(trans_output.GetKernelArg());

        trans_input_size  = trans_input_skippable ? 0 : trans_input.GetOutputTensorSize();
        trans_weight_size = trans_weight_skippable ? 0 : trans_weight.GetOutputTensorSize();
        trans_output_size = trans_output_skippable ? 0 : trans_output.GetOutputTensorSize();

        int idx = 0;
        if(!trans_input_skippable)
            trans_input_idx = idx++;
        if(!trans_weight_skippable)
            trans_weight_idx = idx++;
        if(!trans_output_skippable)
            trans_output_idx = idx++;
    }

    const size_t cast_size = need_cast ? miopen::GetTypeSize(miopenFloat) * n * c * hi * wi : 0;

    MultiBufferWorkspaceTraits wt(
        {trans_input_size, trans_weight_size, trans_output_size, cast_size});

    trans_input_offset  = wt.GetOffset(0);
    trans_weight_offset = wt.GetOffset(1);
    trans_output_offset = wt.GetOffset(2);

    const size_t cast_offset = wt.GetOffset(3);

    const int kID_trans_start = isGfx90aFp16altSupport ? 2 : 1;

    const TensorDescriptor cast_desc(
        miopenFloat, problem.GetOut().GetLengths(), problem.GetOut().GetStrides());
    auto null_buf = shared<Data_t>{};

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            const auto& workSpace   = data_ctx.workSpace;
            const auto ker =
                handle.Run(kernels[(isGfx90aFp16altSupport && data_ctx.gfx90aFp16alt) ? 1 : 0]);
            float elapsed = 0;

            auto trans_input_buf =
                trans_input_size == 0
                    ? null_buf
                    : handle.CreateSubBuffer(workSpace, trans_input_offset, trans_input_size);
            auto trans_weight_buf =
                trans_weight_size == 0
                    ? null_buf
                    : handle.CreateSubBuffer(workSpace, trans_weight_offset, trans_weight_size);
            auto trans_output_buf =
                trans_output_size == 0
                    ? null_buf
                    : handle.CreateSubBuffer(workSpace, trans_output_offset, trans_output_size);
            auto cast_buf = cast_size == 0
                                ? null_buf
                                : handle.CreateSubBuffer(workSpace, cast_offset, cast_size);

            if(need_set_zero)
            {
                auto zero_buf = need_cast
                                    ? cast_buf.get()
                                    : ((is_nchw && !trans_input_skippable) ? trans_input_buf.get()
                                                                           : tensors.out);
                auto& zero_desc =
                    need_cast
                        ? cast_desc
                        : tensors.outDesc; // use the same desc for NCHW/NHWC for this dense tensor
                float zero = 0.f;

                SetTensor(handle, zero_desc, zero_buf, &zero);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            if(is_nchw)
            {
                if(!trans_output_skippable)
                {
                    auto& karg_output = opArgsTrans[trans_output_idx];
                    karg_output[0]    = OpKernelArg(trans_output_buf.get());
                    karg_output[1]    = OpKernelArg(tensors.in);
                    handle.Run(kernels[kID_trans_start + trans_output_idx])(karg_output);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                if(!trans_weight_skippable)
                {
                    auto& karg_weight = opArgsTrans[trans_weight_idx];
                    karg_weight[0]    = OpKernelArg(trans_weight_buf.get());
                    karg_weight[1]    = OpKernelArg(tensors.w);
                    handle.Run(kernels[kID_trans_start + trans_weight_idx])(karg_weight);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
            }

            opArgs[0] = need_cast ? OpKernelArg(cast_buf.get())
                                  : ((is_nchw && !trans_input_skippable)
                                         ? OpKernelArg(trans_input_buf.get())
                                         : OpKernelArg(tensors.out));
            opArgs[1] = (is_nchw && !trans_weight_skippable) ? OpKernelArg(trans_weight_buf.get())
                                                             : OpKernelArg(tensors.w);
            opArgs[2] = (is_nchw && !trans_output_skippable) ? OpKernelArg(trans_output_buf.get())
                                                             : OpKernelArg(tensors.in);

            ker(opArgs);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();

            if(need_cast)
            {
                CastTensor(handle,
                           &lowp_quant,
                           false,
                           cast_desc,
                           cast_buf.get(),
                           tensors.outDesc,
                           (is_nchw && !trans_input_skippable) ? trans_input_buf.get()
                                                               : tensors.out,
                           0,
                           0);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }
            if((is_nchw && !trans_input_skippable))
            {
                auto& karg_input = opArgsTrans[trans_input_idx];
                karg_input[0]    = OpKernelArg(tensors.out);
                karg_input[1]    = OpKernelArg(trans_input_buf.get());
                handle.Run(kernels[kID_trans_start + trans_input_idx])(karg_input);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeImplGemmDynamicForwardDlopsNCHWCInvokerFactory(
    const ProblemDescription& problem,
    const solver::conv::PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC& config)
{
    int hi         = problem.GetInHeight();
    int wi         = problem.GetInWidth();
    int n          = problem.GetInBatchSize();
    int k          = problem.GetOutChannels() * config.vector_c;
    int c          = problem.GetInChannels();
    int ks         = 1;
    int ho         = problem.GetOutHeight();
    int wo         = problem.GetOutWidth();
    int stride_h   = problem.GetKernelStrideH();
    int stride_w   = problem.GetKernelStrideW();
    int dilation_h = problem.GetDilationH();
    int dilation_w = problem.GetDilationW();
    int pad_h      = problem.GetPadH();
    int pad_w      = problem.GetPadW();
    int y          = problem.GetWeightsHeight();
    int x          = problem.GetWeightsWidth();
    int group      = problem.GetGroupCount();

    // Currentlly we do not tile in H/W dimension, using tile H/W as Ho/Wo, Thus Number of Tile
    // equal to one
    uint32_t upper_bound_h = 0xffff; // 16bit
    uint32_t upper_bound_w = 0xffff; // 16bit
    uint32_t tile_h =
        igemm_find_tile_size_with_upper_bound(ho, upper_bound_h, stride_h, dilation_h, y);
    uint32_t tile_w =
        igemm_find_tile_size_with_upper_bound(wo, upper_bound_w, stride_w, dilation_w, x);
    uint32_t ntile_h = 1;
    uint32_t ntile_w = 1;
    if(tile_h != 0 && tile_w != 0)
    {
        ntile_h = (ho + tile_h - 1) / tile_h;
        ntile_w = (wo + tile_w - 1) / tile_w;
    }
    else
        MIOPEN_THROW("tile_hw should not be zero");

    int tile_hw  = (tile_h << 16) | tile_w;
    int ntile_hw = (ntile_h << 16) | ntile_w;
    // Split K make no sense in vector format
    int stride_hw   = (stride_h << 16) | stride_w;
    int dilation_hw = (dilation_h << 16) | dilation_w;
    int pad_hw      = (pad_h << 16) | pad_w;
    int wei_hw      = (y << 16) | x;
    // Initialize here for better readibility
    uint32_t s_move_slice_k_y = (config.gemm_k_per_block / config.vector_c / x) % y;
    uint32_t s_move_slice_k_x = config.gemm_k_per_block / config.vector_c % x;
    uint32_t s_move_slice_k_c = (config.gemm_k_per_block / config.vector_c / (x * y)) % (c / group);
    int move_slice_k = (s_move_slice_k_y << 16) | (s_move_slice_k_x << 8) | s_move_slice_k_c;

    int splits_4G = solver::igemm_split_batch_size(
        hi, wi, ho, wo, n, k, c, miopen::GetTypeSize(problem.GetInDataType()));
    splits_4G       = (splits_4G == 0 ? n : splits_4G);
    uint32_t gemm_n = 1;
    uint32_t gemm_m = 1;
    if(splits_4G != 0)
    {
        gemm_n = (n / splits_4G) * tile_h * tile_w;
        gemm_m = k / group;
    }
    else
        MIOPEN_THROW("splits_4G should not be zero");
    magic_div_u32_t mdiv_0, mdiv_1, mdiv_2, mdiv_3, mdiv_4, mdiv_5, mdiv_6, mdiv_7;
    uint32_t shift_pack_0, shift_pack_1;

    mdiv_0 = magic_div_u32_gen((gemm_n + config.gemm_n_per_block - 1) / config.gemm_n_per_block);
    mdiv_1 = magic_div_u32_gen((gemm_m + config.gemm_m_per_block - 1) / config.gemm_m_per_block);
    mdiv_2 = magic_div_u32_gen(tile_h);
    mdiv_3 = magic_div_u32_gen(tile_w);
    mdiv_4 = magic_div_u32_gen(y);
    mdiv_5 = magic_div_u32_gen(x);
    mdiv_6 = magic_div_u32_gen(ntile_h);
    mdiv_7 = magic_div_u32_gen(ntile_w);
    shift_pack_0 = magic_div_u32_pack_shift(mdiv_0.shift, mdiv_1.shift, mdiv_2.shift, mdiv_3.shift);
    shift_pack_1 = magic_div_u32_pack_shift(mdiv_4.shift, mdiv_5.shift, mdiv_6.shift, mdiv_7.shift);

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(tile_hw);
    opArgs.emplace_back(ntile_hw);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n / splits_4G);
    opArgs.emplace_back(k / group);
    opArgs.emplace_back(c / group);
    opArgs.emplace_back(group);
    opArgs.emplace_back(ks);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_hw);
    opArgs.emplace_back(dilation_hw);
    opArgs.emplace_back(pad_hw);
    opArgs.emplace_back(wei_hw);
    opArgs.emplace_back(move_slice_k);
    opArgs.emplace_back(mdiv_0.magic);
    opArgs.emplace_back(mdiv_1.magic);
    opArgs.emplace_back(mdiv_2.magic);
    opArgs.emplace_back(mdiv_3.magic);
    opArgs.emplace_back(mdiv_4.magic);
    opArgs.emplace_back(mdiv_5.magic);
    opArgs.emplace_back(mdiv_6.magic);
    opArgs.emplace_back(mdiv_7.magic);
    opArgs.emplace_back(shift_pack_0);
    opArgs.emplace_back(shift_pack_1);

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            const auto ker          = handle.Run(kernels[0]);

            opArgs[0] = OpKernelArg(tensors.in);
            opArgs[1] = OpKernelArg(tensors.w);
            opArgs[2] = OpKernelArg(tensors.out);
            ker(opArgs);

            if(handle.IsProfilingEnabled())
            {
                float elapsed = 0;
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}
} // namespace conv
} // namespace miopen
