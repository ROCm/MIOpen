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

#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/sequences.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#include <boost/any.hpp>

#include <tuple>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS)

#define WINODATA 2
#define WINOFILTER 3
#define MAX_CU_LIMIT 512

static inline size_t Ceil(const size_t v, const size_t m)
{
    assert(m > 0);
    return (v + m - 1) / m;
}

static inline size_t quantize_up(size_t val, size_t factor) { return Ceil(val, factor) * factor; }

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

    size_t g_s = quantize_up(S, s_factor);
    size_t g_r = quantize_up(R, r_factor);
    size_t g_c = quantize_up(C, c_factor);
    size_t g_k = quantize_up(K, k_factor);
    size_t g_w = OW;
    size_t g_h = OH;

    if((pad_W % 2 == 0) && (idilation_w > 1 || S_stride > 1))
        g_w += 1;
    if((pad_H % 2 == 1) && (idilation_h > 1 || R_stride > 1))
        g_h += 1;

    g_w            = quantize_up(g_w, w_factor);
    g_h            = quantize_up(g_h, h_factor);
    size_t g_n_w_h = quantize_up(g_w * g_h * N, nwh_factor * w_factor * h_factor);

    int best_n_groups_cnt = 1;
    double min_param      = 0;
    for(auto i = 1; i < n_groups; ++i)
    {
        size_t g_n_w_h_k =
            quantize_up(g_n_w_h * g_k, nwh_factor * w_factor * h_factor * k_factor * i);
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
            std::make_tuple(seq::Span<int, 1, MAX_CU_LIMIT>{}, &PerformanceConfigConvBinWinogradRxSf2x3::n_groups)
        );
    }
// clang-format on

/// \todo Consider re-using code from RxS.
inline bool IsShaderContraintsMet(const int R,
                                  const int S,
                                  const int,
                                  const int,
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
    if(params.direction.IsBackwardData() || params.direction.IsBackwardWrW())
    {
        if(!(0 <= params.GetBackwardPadW() && params.GetBackwardPadW() < std::pow(2, 16)))
            return false;
        if(!(0 <= params.GetBackwardPadH() && params.GetBackwardPadH() < std::pow(2, 16)))
            return false;
    }
    const auto grid_workgroup_count_x = params.GetStream().GetMaxComputeUnits();
    assert(params.weights_layout.length() == 0);
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

PerformanceConfigConvBinWinogradRxSf2x3::PerformanceConfigConvBinWinogradRxSf2x3(int n_groups_)
    : n_groups(n_groups_)
{
}

void PerformanceConfigConvBinWinogradRxSf2x3::EuristicInit(const ConvolutionContext& config)
{
    const auto n_inputs_per_group  = config.n_inputs / config.group_counts,
               n_outputs_per_group = config.n_outputs / config.group_counts;
    if(config.group_counts == 1)
    {
        n_groups = config.GetStream().GetMaxComputeUnits();
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
                                      config.GetStream().GetMaxComputeUnits(),
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
                                      config.GetStream().GetMaxComputeUnits(),
                                      config.group_counts);
    }
}

bool PerformanceConfigConvBinWinogradRxSf2x3::SetNextValue()
{
    return !PerfFieldRules().Next(*this);
}

bool PerformanceConfigConvBinWinogradRxSf2x3::IsValidValue() const
{
    return PerfFieldRules().IsIn(*this);
}

bool PerformanceConfigConvBinWinogradRxSf2x3::IsValid(const ConvolutionContext& config) const
{
    if(config.GetStream().GetMaxComputeUnits() < n_groups)
        return false;

    if(!IsValidValue())
        return false;
    return true;
}

inline bool PerformanceConfigConvBinWinogradRxSf2x3::
operator==(const PerformanceConfigConvBinWinogradRxSf2x3& other) const
{
    return n_groups == other.n_groups;
}

std::string PerformanceConfigConvBinWinogradRxSf2x3::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvBinWinogradRxSf2x3
ConvBinWinogradRxSf2x3::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvBinWinogradRxSf2x3 pp;
    pp.EuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvBinWinogradRxSf2x3::IsValidPerformanceConfig(
    const ConvolutionContext& problem, const PerformanceConfigConvBinWinogradRxSf2x3& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

PerformanceConfigConvBinWinogradRxSf2x3
ConvBinWinogradRxSf2x3::Search(const ConvolutionContext& context) const
{
    if(context.direction.IsForward())
        return GenericSearchFwd(*this, context);
    else
        return GenericSearchBwd(*this, context);
}

inline void FillVarsFromConfig(int& H,
                               int& W,
                               int& R,
                               int& S,
                               int& R_stride,
                               int& S_stride,
                               int& C,
                               int& K,
                               int& out_H,
                               int& out_W,
                               int& pad_H,
                               int& pad_W,
                               int& N,
                               int& idilation_w,
                               int& idilation_h,
                               int& n_groups,
                               int& group_cnt,
                               const ConvolutionContext& config)
{
    group_cnt   = config.group_counts;
    n_groups    = config.GetStream().GetMaxComputeUnits();
    pad_H       = config.direction.IsForward() ? config.pad_h : config.GetBackwardPadH();
    pad_W       = config.direction.IsForward() ? config.pad_w : config.GetBackwardPadW();
    H           = config.in_height;
    W           = config.in_width;
    R           = config.kernel_size_h;
    S           = config.kernel_size_w;
    R_stride    = config.kernel_stride_h;
    S_stride    = config.kernel_stride_w;
    C           = config.n_inputs;
    K           = config.n_outputs;
    out_H       = config.out_height;
    out_W       = config.out_width;
    N           = config.batch_sz;
    idilation_w = config.kernel_dilation_h;
    idilation_h = config.kernel_dilation_w;
}

template <typename B, typename T, typename TW>
int ConvBinWinogradRxSf2x3::RunAndMeasureSolution(const miopen::Handle& profile_h,
                                                  B bot_ocl_buf,
                                                  T top_ocl_buf,
                                                  TW wei_ocl_buf,
                                                  ConstData_t bias_ocl_buf,
                                                  const ConvolutionContext& config,
                                                  const ConvSolution& solution,
                                                  float& elapsed_time) const
{
    assert(bias_ocl_buf == nullptr);
    (void)bias_ocl_buf;
    const KernelInfo k_info = solution.construction_params.back();
#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        // ConvolutionContext::general_compile_options is for OpenCL kernels
        // and thus not applicable for assembly.
        auto kernel = profile_h.AddKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);

        int reserved      = 0;
        int* reserved_ptr = nullptr;

        static const int F_REVERSE_R     = 1 << 0;
        static const int F_REVERSE_S     = 1 << 1;
        static const int F_FLIP_K_C      = 1 << 2;
        static const int F_NKC_STRIDES   = 1 << 9;
        static const int F_GROUP_STRIDES = 1 << 10;

        const auto is_forward = config.direction.IsForward();
        const auto is_wrw     = config.direction.IsBackwardWrW();

        int H, W, R, S, R_stride, S_stride, C, K, out_H, out_W, pad_H, pad_W, N, idilation_w,
            idilation_h, n_groups, group_cnt;
        // clang-format off
        FillVarsFromConfig(H, W, R, S, R_stride, S_stride, C, K, out_H, out_W, pad_H,
                           pad_W, N, idilation_w, idilation_h, n_groups, group_cnt, config);
        // clang-format on
        int flags = ((is_forward || is_wrw) ? 0 : F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C) +
                    F_NKC_STRIDES + F_GROUP_STRIDES;

        C = C / group_cnt;
        K = K / group_cnt;
        // cppcheck-suppress unreadVariable

        const auto in_layout = (is_wrw) ? GetSwappedNCLayout(GetMemLayout_t(config.in_layout))
                                        : GetMemLayout_t(config.in_layout);
        const auto out_layout = (is_wrw) ? GetSwappedNCLayout(GetMemLayout_t(config.out_layout))
                                         : GetMemLayout_t(config.out_layout);
        const auto wei_layout =
            (is_wrw) ? GetSwappedNCLayout(MemLayout_t::NCHW)
                     : (is_forward ? (MemLayout_t::NCHW) : GetSwappedNCLayout(MemLayout_t::NCHW));

        // cppcheck-suppress unreadVariable
        BuffInfo d_buf(GetGroupConvLayout(in_layout, true),
                       N,
                       C,
                       H,
                       W,
                       group_cnt,
                       GetTypeSize(config.in_data_type)),
            // cppcheck-suppress unreadVariable
            o_buf(GetGroupConvLayout(out_layout, true),
                  N,
                  K,
                  out_H,
                  out_W,
                  group_cnt,
                  GetTypeSize(config.out_data_type)),
            // cppcheck-suppress unreadVariable
            f_buf(GetGroupConvLayout(wei_layout, false),
                  K,
                  C,
                  R,
                  S,
                  group_cnt,
                  GetTypeSize(config.weights_data_type));

        if(k_info.l_wk[0] != 0)
            n_groups = solver::ConvBinWinogradRxSf2x3::GetNGroups(config.group_counts,
                                                                  k_info.g_wk[0] / k_info.l_wk[0]);
        else
            n_groups = solver::ConvBinWinogradRxSf2x3::GetNGroups(
                config.group_counts,
                k_info.g_wk[0] / 512); // For OCL runtime. Issue #1724

        kernel(N,
               C,
               H,
               W,
               K,
               n_groups,
               flags,
               reserved,
               bot_ocl_buf,
               wei_ocl_buf,
               top_ocl_buf,
               reserved_ptr, // Unused return_addr.
               R,
               S,
               pad_H, // Like Fwd wino.
               pad_W,
               out_H,
               out_W,
               reserved_ptr, // Unused bias_addr.
               reserved,     // Unused relu_alpha.
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

        elapsed_time = profile_h.GetKernelTime();
    }
#ifdef NDEBUG
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return -1;
    }
#endif
    return 0;
}

bool ConvBinWinogradRxSf2x3::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.Is2d())
        return false;
    if(!(params.IsFp32() || params.IsFp16()))
        return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;

    const auto name = params.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9")))
        return false;
    if(params.IsFp16() && !(StartsWith(name, "gfx906") || StartsWith(name, "gfx908")))
        return false;

    // clang-format off
    if (! ( (params.kernel_stride_w == 1 || params.kernel_stride_w == 2)
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0
        && params.in_layout == "NCHW"))
        return false;
    // clang-format on

    const auto n_inputs_per_group  = params.n_inputs / params.group_counts,
               n_outputs_per_group = params.n_outputs / params.group_counts;

    if(params.direction.IsBackwardWrW())
    {
        if(params.kernel_stride_w == 2)
            return false;
        return IsShaderContraintsMet(params.in_height,
                                     params.in_width,
                                     params.kernel_dilation_h,
                                     params.kernel_dilation_w,
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
                                     params.kernel_stride_h,
                                     params.kernel_stride_w,
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

ConvSolution
ConvBinWinogradRxSf2x3::GetSolution(const ConvolutionContext& params,
                                    const PerformanceConfigConvBinWinogradRxSf2x3& config,
                                    const bool disableConfigOverrideFromEnv) const
{
    const auto n_groups = config.n_groups;
    static bool IsWarned;
    if(!IsWarned)
    {
        if(params.GetStream().GetMaxComputeUnits() > MAX_CU_LIMIT)
            MIOPEN_LOG_WE(SolverDbId(*this) << ": GPU has "
                                            << params.GetStream().GetMaxComputeUnits()
                                            << "CUs, but this solver supports max "
                                            << MAX_CU_LIMIT
                                            << "and thus may show sub-optimal performance.");
        IsWarned = true;
    }

    ConvSolution result;

    const PerformanceConfigConvBinWinogradRxSf2x3* pcfg = &config;
    PerformanceConfigConvBinWinogradRxSf2x3 fromEnv;

    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(params))
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS: "
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

    KernelInfo kernel;

    kernel.g_wk.push_back(512 * pcfg->GetNGroups() * params.group_counts);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    std::string kernel_name    = "miopenSp3AsmConv";
    std::string kernel_file    = "Conv_Winograd";
    std::string kernel_postfix = "_v21_1_0_gfx9";

    if(params.IsFp32())
        kernel_postfix += "_fp32";
    else
    {
        kernel_postfix += "_fp16_dot2_edc";
    }
    if(params.kernel_stride_w == 1)
    {
        kernel_postfix += "_stride1";
    }
    else if(params.kernel_stride_w == 2 && !params.direction.IsBackwardData())
    {
        kernel_postfix += "_stride2";
    }
    else // if(params.kernel_dilation_h == 2)
    {
        kernel_postfix += "_dilation2";
    }

    if(params.group_counts != 1 || params.direction.IsBackwardWrW())
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
        int reserved                    = 0;
        int* reserved_ptr               = nullptr;
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
            return [=](const Handle& handle, const boost::any& primitive_params) {
                const auto k         = handle.Run(kernels[0]);
                const auto& data_ctx = boost::any_cast<conv::DataInvokeParams>(primitive_params);
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
                  reserved_ptr, // Unused bias_addr.
                  reserved,     // Unused relu_alpha.
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
            return [=](const Handle& handle, const boost::any& primitive_params) {
                decltype(auto) invoke_params =
                    boost::any_cast<conv::WrWInvokeParams>(primitive_params);
                const auto& tensors = invoke_params.tensors;

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

                int reserved      = 0;
                int* reserved_ptr = nullptr;

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
                                       pad_H, // Like Fwd wino.
                                       pad_W,
                                       out_H,
                                       out_W,
                                       reserved_ptr, // Unused bias_addr.
                                       reserved,     // Unused relu_alpha.
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
