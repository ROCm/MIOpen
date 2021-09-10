/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/solver.hpp>

#include <miopen/env.hpp>
#include <miopen/kernel.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/sequences.hpp>

#include <boost/any.hpp>

#define WORKAROUND_ISSUE_1146 1 // check asm solver applicability for gfx90a

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS)

#define WINODATA 3
#define WINOFILTER 2
#define MAX_CU_LIMIT 512

static inline size_t Ceil(const size_t v, const size_t m)
{
    assert(m > 0);
    return (v + m - 1) / m;
}

static inline size_t RoundUpToMultiple(size_t val, size_t factor)
{
    return Ceil(val, factor) * factor;
}

/// \todo Consider re-using code from RxS_f2x3.
static inline int GetBestNGroupParam(const int R,
                                     const int S,
                                     const int R_stride,
                                     const int S_stride,
                                     const int C,
                                     const int K,
                                     const int OH,
                                     const int OW,
                                     const int pad_H,
                                     const int pad_W,
                                     const int N,
                                     const int idilation_w,
                                     const int idilation_h,
                                     const int n_groups,
                                     const int G)
{
    int o_tile     = WINODATA;
    int f_tile     = WINOFILTER;
    int r_factor   = f_tile * 2;
    int s_factor   = r_factor;
    int c_factor   = 2;
    int k_factor   = 32;
    int nwh_factor = 32;
    int w_factor   = o_tile * idilation_w * S_stride;
    int h_factor   = o_tile * idilation_h * R_stride;

    if(S_stride == 1 && idilation_w == 1 && S <= f_tile)
        s_factor = f_tile;
    if((R_stride == 1 && idilation_h == 1) || (R % (f_tile * 2)) == 1)
        r_factor = f_tile;
    if(S_stride == 2 || R_stride == 2 || idilation_w == 2 || idilation_h == 2)
        c_factor = 1;

    size_t g_s = RoundUpToMultiple(S, s_factor);
    size_t g_r = RoundUpToMultiple(R, r_factor);
    size_t g_c = RoundUpToMultiple(C, c_factor);
    size_t g_k = RoundUpToMultiple(K, k_factor);
    size_t g_w = OW;
    size_t g_h = OH;

    if((pad_W % 2 == 0) && (idilation_w > 1 || S_stride > 1))
        g_w += 1;
    if((pad_H % 2 == 1) && (idilation_h > 1 || R_stride > 1))
        g_h += 1;

    g_w            = RoundUpToMultiple(g_w, w_factor);
    g_h            = RoundUpToMultiple(g_h, h_factor);
    size_t g_n_w_h = RoundUpToMultiple(g_w * g_h * N, nwh_factor * w_factor * h_factor);

    int best_n_groups_cnt = 1;
    double min_param      = 0;
    for(auto i = 1; i < n_groups; ++i)
    {
        size_t g_n_w_h_k =
            RoundUpToMultiple(g_n_w_h * g_k, nwh_factor * w_factor * h_factor * k_factor * i);
        size_t granulated_mac_count = g_n_w_h_k * g_c * g_s * g_r;
        size_t n_groups_per_cu      = Ceil(i * G, n_groups);
        double perf_metric = static_cast<double>(n_groups_per_cu) * granulated_mac_count / i;
        if(static_cast<double>(granulated_mac_count) / i > 1.0e+7)
            perf_metric *= (1 + i * 0.003);
        else
            perf_metric *= (1 + i * 0.04);
        if(i == 1)
            min_param = perf_metric;
        if(min_param > perf_metric)
        {
            best_n_groups_cnt = i;
            min_param         = perf_metric;
        }
    }
    return best_n_groups_cnt;
}

namespace miopen {
namespace solver {

namespace {
// clang-format off
auto PerfFieldRules()
{
    return seq::MakeRuleSet(
        std::make_tuple(seq::Span<int, 1, MAX_CU_LIMIT>{}, &PerformanceConfigConvBinWinogradRxSf3x2::n_groups)
    );
}
// clang-format on

/// \todo Consider re-using code from RxS_f2x3.
inline bool IsShaderContraintsMet(const int R,
                                  const int S,
                                  const int C,
                                  const int K,
                                  const int H,
                                  const int W,
                                  const int OH,
                                  const int OW,
                                  const int N,
                                  const ConvolutionContext& params)
{
    // Padding for bwd data shall not be negative.
    /// \todo Either remove WrW related code or re-use function from RxS
    if(params.direction.IsBackwardData())
    {
        if(!(0 <= params.GetBackwardPadW() && params.GetBackwardPadW() < std::pow(2, 16)))
            return false;
        if(!(0 <= params.GetBackwardPadH() && params.GetBackwardPadH() < std::pow(2, 16)))
            return false;
    }

    const auto grid_workgroup_count_x = params.GetStream().GetMaxHardwareComputeUnits();

    // clang-format off
    // Check implementation limits.
    return N < std::pow(2, 16)
        && C < std::pow(2, 16)
        && K < std::pow(2, 16)
        && H < std::pow(2, 16)
        && W < std::pow(2, 16)
        && OH < std::pow(2, 16)
        && OW < std::pow(2, 16)
        && params.pad_w < std::pow(2, 16)
        && params.pad_h < std::pow(2, 16)
        && S < std::pow(2, 16)
        && R < std::pow(2, 16)
        && grid_workgroup_count_x < std::pow(2, 16)
        && (C * H * W) <= std::pow(2, 28)
        && (OH * OW) <= std::pow(2, 23)
        && (K * OH * OW) <= std::pow(2, 28)
        && (K * R * S) <= std::pow(2, 28)
        && (C * R * S) <= std::pow(2, 28);
    // clang-format on
}

} // namespace

PerformanceConfigConvBinWinogradRxSf3x2::PerformanceConfigConvBinWinogradRxSf3x2(int n_groups_)
    : n_groups(n_groups_)
{
}

void PerformanceConfigConvBinWinogradRxSf3x2::HeuristicInit(const ConvolutionContext& config)
{
    const auto n_inputs_per_group  = config.n_inputs / config.group_counts,
               n_outputs_per_group = config.n_outputs / config.group_counts;
    if(config.group_counts == 1)
    {
        n_groups = config.GetStream().GetMaxHardwareComputeUnits();
        return;
    }

    if(config.direction.IsBackwardWrW())
    {
        n_groups = GetBestNGroupParam(config.in_height,
                                      config.in_width,
                                      config.kernel_dilation_h,
                                      config.kernel_dilation_w,
                                      config.batch_sz,    // N
                                      n_inputs_per_group, // K
                                      config.kernel_size_h,
                                      config.kernel_size_w,
                                      config.pad_w,
                                      config.pad_h,
                                      n_outputs_per_group, // C
                                      config.kernel_stride_h,
                                      config.kernel_stride_w,
                                      config.GetStream().GetMaxHardwareComputeUnits(),
                                      config.group_counts);
    }
    else
    {
        n_groups = GetBestNGroupParam(config.kernel_size_h, // RxS
                                      config.kernel_size_w,
                                      config.kernel_stride_h,
                                      config.kernel_stride_w,
                                      n_inputs_per_group,  // C
                                      n_outputs_per_group, // K
                                      config.out_height,   // OHxOW
                                      config.out_width,
                                      config.pad_w,
                                      config.pad_h,
                                      config.batch_sz, // N
                                      config.kernel_dilation_h,
                                      config.kernel_dilation_w,
                                      config.GetStream().GetMaxHardwareComputeUnits(),
                                      config.group_counts);
    }
}

bool PerformanceConfigConvBinWinogradRxSf3x2::SetNextValue(const ConvolutionContext& /*config*/)
{
    return !PerfFieldRules().Next(*this);
}

bool PerformanceConfigConvBinWinogradRxSf3x2::IsValidValue() const
{
    return PerfFieldRules().IsIn(*this);
}

bool PerformanceConfigConvBinWinogradRxSf3x2::IsValid(const ConvolutionContext& config) const
{
    if(config.GetStream().GetMaxHardwareComputeUnits() < n_groups)
        return false;

    if(!IsValidValue())
        return false;
    return true;
}

inline bool PerformanceConfigConvBinWinogradRxSf3x2::operator==(
    const PerformanceConfigConvBinWinogradRxSf3x2& other) const
{
    return n_groups == other.n_groups;
}

std::string PerformanceConfigConvBinWinogradRxSf3x2::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvBinWinogradRxSf3x2
ConvBinWinogradRxSf3x2::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvBinWinogradRxSf3x2 pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvBinWinogradRxSf3x2::IsValidPerformanceConfig(
    const ConvolutionContext& problem, const PerformanceConfigConvBinWinogradRxSf3x2& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

PerformanceConfigConvBinWinogradRxSf3x2
ConvBinWinogradRxSf3x2::Search(const ConvolutionContext& context,
                               const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

bool ConvBinWinogradRxSf3x2::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.Is2d())
        return false;
    if(!(params.IsFp32() || params.IsFp16()))
        return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;
    if(!params.IsLayoutDefault())
        return false;

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const auto max_cu = params.GetStream().GetMaxHardwareComputeUnits();
    if(max_cu > MAX_CU_LIMIT)
        return false;

    const auto name = params.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9") || StartsWith(name, "gfx10")))
        return false;
#if WORKAROUND_ISSUE_1146
    if(name == "gfx90a")
        return false;
#endif

    if(params.IsFp16() &&
       !(StartsWith(name, "gfx906") || StartsWith(name, "gfx908") || StartsWith(name, "gfx1011") ||
         StartsWith(name, "gfx1012") || StartsWith(name, "gfx103")))
        return false;

    // clang-format off
    if (! (params.kernel_stride_w == 1
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0))
        return false;
    // clang-format on

    const auto n_inputs_per_group  = params.n_inputs / params.group_counts,
               n_outputs_per_group = params.n_outputs / params.group_counts;

    if(params.direction.IsBackwardWrW())
    {
        return IsShaderContraintsMet(params.in_height,
                                     params.in_width,
                                     params.batch_sz,    // N
                                     n_inputs_per_group, // K
                                     params.out_height,
                                     params.out_width,
                                     params.kernel_size_h,
                                     params.kernel_size_w,
                                     n_outputs_per_group, // C
                                     params);
    }
    else
    {
        return IsShaderContraintsMet(params.kernel_size_h, // RxS
                                     params.kernel_size_w,
                                     n_inputs_per_group,  // C
                                     n_outputs_per_group, // K
                                     params.in_height,    // HxW
                                     params.in_width,
                                     params.out_height, // OHxOW
                                     params.out_width,
                                     params.batch_sz, // N
                                     params);
    }
}

/// \todo Consider re-using code from RxS_f2x3.
ConvSolution
ConvBinWinogradRxSf3x2::GetSolution(const ConvolutionContext& params,
                                    const PerformanceConfigConvBinWinogradRxSf3x2& config,
                                    const bool disableConfigOverrideFromEnv) const
{
    const PerformanceConfigConvBinWinogradRxSf3x2* pcfg = &config;
    PerformanceConfigConvBinWinogradRxSf3x2 fromEnv;
    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(params))
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS: "
                                 "Bad format or invalid for the problem config: "
                                 << s);
                }
                else
                {
                    MIOPEN_LOG_I("Overridden from env: " << fromEnv.ToString());
                    pcfg = &fromEnv;
                }
            }
        }
    }

    ConvSolution result;
    KernelInfo kernel;

    const auto n_groups = pcfg->GetNGroups();
    const auto name     = params.GetStream().GetDeviceName();
    const auto is_gfx9  = StartsWith(name, "gfx9");
    size_t wg_size      = is_gfx9 ? 512 : 256;

    kernel.g_wk.push_back(wg_size * n_groups * params.group_counts);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    std::string kernel_name    = "miopenSp3AsmConv_v21_1_2";
    std::string kernel_file    = "Conv_Winograd_v21_1_2";
    std::string kernel_postfix = params.IsFp32() ? "_f3x2_fp32" : "_f3x2_fp16_dot2_edc";

    if(is_gfx9)
    {
        kernel_name += "_gfx9";
    }
    else // if(StartsWith(name, "gfx10"))
    {
        kernel_name += "_gfx10";
        kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");
    }

    if(params.kernel_stride_w == 1)
    {
        kernel_postfix += "_stride1";
    }

    kernel_postfix += "_group";
    kernel.kernel_name = kernel_name + kernel_postfix;
    kernel.kernel_file = kernel_file + kernel_postfix + ".s";

    result.construction_params.push_back(kernel);

    if(!params.direction.IsBackwardWrW())
    {
        const bool is_forward     = params.direction.IsForward();
        constexpr int F_REVERSE_R = 1 << 0;
        constexpr int F_REVERSE_S = 1 << 1;
        constexpr int F_FLIP_K_C  = 1 << 2;
        // These are not used yet. Nevertheless let's keep as a shader documentation.
        // constexpr int F_FLIP_DATA_N_C = 1 << 3; // Unsupported in f3x2.
        // constexpr int F_FLIP_OUT_N_K = 1 << 4; // Unsupported in f3x2.
        // constexpr int L_F_ADDR_INDIRECT  = 1 << 6;
        // constexpr int L_F_BIAS  = 1 << 7;
        // constexpr int L_F_LEAKY_RELU  = 1 << 8;
        constexpr int L_F_NKC_STRIDES   = 1 << 9;
        constexpr int L_F_GROUP_STRIDES = 1 << 10;
        // constexpr int L_F_FORCE_FILTER_TRAVERSE_MODE  = 1 << 11;
        // constexpr int L_F_FILTER_TRAVERSE_DUAL  = 1 << 12;
        // constexpr int L_F_TENSOR_OFFSETS  = 1 << 13;
        // constexpr int L_F_USE_EXTENDED_FLAGS_64  = 1 << 15;
        int reserved             = 0;
        uint64_t reserved_offset = 0;
        int* reserved_ptr        = nullptr;
        int ignore;

        int N, C, H, W, K, out_H, out_W, R, S, pad_H, pad_W;
        GetCompiledInParameters(
            params, &N, &C, &H, &W, &K, &ignore, &out_H, &out_W, &R, &S, &pad_H, &pad_W);
        const auto group_cnt = params.group_counts;
        C                    = C / group_cnt;
        K                    = K / group_cnt;
        int flags            = is_forward ? 0 : F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
        flags |= L_F_NKC_STRIDES + L_F_GROUP_STRIDES;

        // cppcheck-suppress unreadVariable
        BuffInfo d_buf(GetGroupConvLayout(GetMemLayout_t(params.in_layout), true),
                       N,
                       C,
                       H,
                       W,
                       group_cnt,
                       GetTypeSize(params.in_data_type)),
            // cppcheck-suppress unreadVariable
            o_buf(GetGroupConvLayout(GetMemLayout_t(params.out_layout), true),
                  N,
                  K,
                  out_H,
                  out_W,
                  group_cnt,
                  GetTypeSize(params.out_data_type)),
            // cppcheck-suppress unreadVariable
            f_buf(GetGroupConvLayout(is_forward ? (MemLayout_t::NCHW)
                                                : GetSwappedNCLayout(MemLayout_t::NCHW),
                                     false),
                  K,
                  C,
                  R,
                  S,
                  group_cnt,
                  GetTypeSize(params.weights_data_type));

        result.invoker_factory = [=](std::vector<Kernel> kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                const auto k         = handle.Run(kernels[0]);
                const auto& data_ctx = primitive_params.CastTo<conv::DataInvokeParams>();
                const auto& tensors  = data_ctx.tensors;

                // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                    << " d_buf.byte_stride.nk=" << d_buf.byte_stride.nk << " d_buf.byte_stride.c=" << d_buf.byte_stride.c
                    << " d_buf.byte_stride.h=" << d_buf.byte_stride.h << " d_buf.byte_stride.w=" << d_buf.byte_stride.w
                    << " f_buf.byte_stride.nk=" << f_buf.byte_stride.nk << " f_buf.byte_stride.c=" << f_buf.byte_stride.c
                    << " f_buf.byte_stride.h=" << f_buf.byte_stride.h << " f_buf.byte_stride.w=" << f_buf.byte_stride.w
                    << " o_buf.byte_stride.nk=" << o_buf.byte_stride.nk << " o_buf.byte_stride.c=" << o_buf.byte_stride.c
                    << " o_buf.byte_stride.h="  << o_buf.byte_stride.h <<  " o_buf.byte_stride.w=" << o_buf.byte_stride.w
                    << " d_buf.byte_stride.g=" << d_buf.byte_stride.g  << " o_buf.byte_stride.g="  << o_buf.byte_stride.g
                    << " f_buf.byte_stride.g=" << f_buf.byte_stride.g); // clang-format on

                k(N,
                  C,
                  H,
                  W,
                  K,
                  n_groups,
                  flags,
                  reserved,
                  tensors.in,
                  tensors.w,
                  tensors.out,
                  reserved_ptr, // Unused return_addr.
                  R,
                  S,
                  pad_H, // Like Fwd wino.
                  pad_W,
                  out_H,
                  out_W,
                  reserved_ptr,    // Unused bias_addr.
                  reserved,        // Unused relu_alpha.
                  reserved,        // Unused reserved2.
                  reserved_offset, // Unused d_offset.
                  reserved_offset, // Unused f_offset.
                  reserved_offset, // Unused o_offset.
                  reserved_offset, // Unused b_offset.
                  d_buf.byte_stride.nk,
                  d_buf.byte_stride.c,
                  d_buf.byte_stride.h,
                  d_buf.byte_stride.w,
                  f_buf.byte_stride.nk,
                  f_buf.byte_stride.c,
                  f_buf.byte_stride.h,
                  f_buf.byte_stride.w,
                  o_buf.byte_stride.nk,
                  o_buf.byte_stride.c,
                  o_buf.byte_stride.h,
                  o_buf.byte_stride.w,
                  group_cnt,
                  d_buf.byte_stride.g,
                  f_buf.byte_stride.g,
                  o_buf.byte_stride.g);
            };
        };
    }
    else
    {
        int unused = 0;
        int N, C, H, W, K, out_H, out_W, R, S;
        GetCompiledInParameters(
            params, &C, &K, &R, &S, &N, &unused, &H, &W, &out_H, &out_W, &unused, &unused);
        const auto group_cnt             = params.group_counts;
        static const int F_NKC_STRIDES   = 1 << 9;
        static const int F_GROUP_STRIDES = 1 << 10;
        int flags                        = F_NKC_STRIDES + F_GROUP_STRIDES;
        N                                = N / group_cnt;
        K                                = K / group_cnt;
        int pad_H                        = params.conv_problem.GetConv().GetConvPads()[0];
        int pad_W                        = params.conv_problem.GetConv().GetConvPads()[1];

        BuffInfo d_buf(
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(params.in_layout)), true),
            N,
            C,
            H,
            W,
            group_cnt,
            GetTypeSize(params.in_data_type)),
            o_buf(GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(params.out_layout)), false),
                  N,
                  K,
                  out_H,
                  out_W,
                  group_cnt,
                  GetTypeSize(params.out_data_type)),
            f_buf(GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true),
                  K,
                  C,
                  R,
                  S,
                  group_cnt,
                  GetTypeSize(params.weights_data_type));

        decltype(auto) batch_sz = params.batch_sz;
        decltype(auto) n_inputs = params.n_inputs;

        result.invoker_factory = [=](std::vector<Kernel> kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                decltype(auto) invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();
                const auto& tensors          = invoke_params.tensors;

                // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                    << " d_buf.byte_stride.nk=" << d_buf.byte_stride.nk << " d_buf.byte_stride.c=" << d_buf.byte_stride.c
                    << " d_buf.byte_stride.h=" << d_buf.byte_stride.h << " d_buf.byte_stride.w=" << d_buf.byte_stride.w
                    << " f_buf.byte_stride.nk=" << f_buf.byte_stride.nk << " f_buf.byte_stride.c=" << f_buf.byte_stride.c
                    << " f_buf.byte_stride.h=" << f_buf.byte_stride.h << " f_buf.byte_stride.w=" << f_buf.byte_stride.w
                    << " o_buf.byte_stride.nk=" << o_buf.byte_stride.nk << " o_buf.byte_stride.c=" << o_buf.byte_stride.c
                    << " o_buf.byte_stride.h="  << o_buf.byte_stride.h <<  " o_buf.byte_stride.w=" << o_buf.byte_stride.w
                    << " d_buf.byte_stride.g=" << d_buf.byte_stride.g  << " o_buf.byte_stride.g="  << o_buf.byte_stride.g
                    << " f_buf.byte_stride.g=" << f_buf.byte_stride.g); // clang-format on
                MIOPEN_LOG_I2(" ctx.batch_sz=" << batch_sz << "ctx.n_inputs=" << n_inputs);

                int reserved             = 0;
                uint64_t reserved_offset = 0;
                int* reserved_ptr        = nullptr;

                handle.Run(kernels[0])(N,
                                       C,
                                       H,
                                       W,
                                       K,
                                       n_groups,
                                       flags,
                                       reserved,
                                       tensors.x,
                                       tensors.dy,
                                       tensors.dw,
                                       reserved_ptr, // Unused return_addr.
                                       R,
                                       S,
                                       pad_H,
                                       pad_W,
                                       out_H,
                                       out_W,
                                       reserved_ptr,    // Unused bias_addr.
                                       reserved,        // Unused relu_alpha.
                                       reserved,        // Unused reserved2.
                                       reserved_offset, // Unused d_offset.
                                       reserved_offset, // Unused f_offset.
                                       reserved_offset, // Unused o_offset.
                                       reserved_offset, // Unused b_offset.
                                       d_buf.byte_stride.nk,
                                       d_buf.byte_stride.c,
                                       d_buf.byte_stride.h,
                                       d_buf.byte_stride.w,
                                       f_buf.byte_stride.nk,
                                       f_buf.byte_stride.c,
                                       f_buf.byte_stride.h,
                                       f_buf.byte_stride.w,
                                       o_buf.byte_stride.nk,
                                       o_buf.byte_stride.c,
                                       o_buf.byte_stride.h,
                                       o_buf.byte_stride.w,
                                       group_cnt,
                                       d_buf.byte_stride.g,
                                       f_buf.byte_stride.g,
                                       o_buf.byte_stride.g);
            };
        };
    }

    return result;
}

} // namespace solver
} // namespace miopen
