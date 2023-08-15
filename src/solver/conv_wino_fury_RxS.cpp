/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/sequences.hpp>
//#include <miopen/stringutils.hpp>

#include <boost/any.hpp>
#include <boost/optional.hpp>

//#include <tuple>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2_PERF_VALS)

namespace miopen {
namespace solver {

namespace {
constexpr size_t max_cu_limit = 512;

template <typename T, typename S = std::enable_if_t<std::is_unsigned_v<T>, T>>
constexpr inline S Ceil(const T val, const T div)
{
    return (val - 1 + div) / div;
}

auto PerfFieldRules()
{
    return seq::MakeRuleSet(std::make_tuple(seq::Span<int, 1, int(max_cu_limit)>{},
                                            &PerformanceConfigConvWinoFuryRxS::n_groups));
}

template <size_t Winodata, size_t Winofilter>
inline boost::optional<PerformanceConfigConvWinoFuryRxS>
GetPerfConfFromEnv(const ConvolutionContext& ctx)
{
    PerformanceConfigConvWinoFuryRxS fromEnv;
    std::string s;
    const char* p_asciz = nullptr;
    const char* env_name;

    if(ConvWinoFuryRxS<Winodata, Winofilter>::is2x3())
    {
        p_asciz  = miopen::GetStringEnv(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3_PERF_VALS{});
        env_name = MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3_PERF_VALS::value();
    }
    else if(ConvWinoFuryRxS<Winodata, Winofilter>::is3x2())
    {
        p_asciz  = miopen::GetStringEnv(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2_PERF_VALS{});
        env_name = MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2_PERF_VALS::value();
    }

    if(p_asciz == nullptr)
        return {};

    s = std::string(p_asciz);

    if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(ctx))
    {
        MIOPEN_LOG_E(env_name << "Tuning config: Bad value or invalid format: `" << s << '\'');
        return boost::none;
    }

    MIOPEN_LOG_I("Overridden from env: " << fromEnv.ToString());

    return fromEnv;
}

class ShaderModel : public UnifiedDescriptionConv2d
{
    static constexpr uint32_t Tw  = 4;
    static constexpr uint32_t Toh = 2;
    static constexpr uint32_t Tow = 2;

    uint32_t Hs;
    uint32_t We;

    uint32_t W;
    uint32_t H;

    int32_t d_H_clip;
    int32_t d_W_clip;

    int32_t o_H_clip;
    int32_t o_W_clip;

    bool is_applicable{false};

public:
    uint16_t out_Hs;
    uint16_t out_We;

    int16_t d_H_clip_bot_neg;
    int16_t d_W_clip_bot_neg;

    int16_t o_H_clip_bot_neg;
    int16_t o_W_clip_bot_neg;

    uint16_t d_H_window;
    uint16_t d_W_window;

    uint16_t o_H_window;
    uint16_t o_W_window;

    uint32_t n_groups;

    ShaderModel(const ProblemDescription& problem,
                int groups,
                size_t /*Winodata*/,
                size_t /*Winofilter*/)
        : UnifiedDescriptionConv2d(problem),
          Hs{Ceil<uint32_t>(out_h, Toh)},
          We{Tow * (Ceil<uint32_t>(out_w, Tow) + Ceil(Tw, Tow) - 1)},

          W{static_cast<uint32_t>(problem.direction.IsBackwardWrW() ? problem.GetOutWidth()
                                                                    : problem.GetInHeight())},
          H{static_cast<uint32_t>(problem.direction.IsBackwardWrW() ? problem.GetOutHeight()
                                                                    : problem.GetInWidth())},

          d_H_clip{static_cast<int32_t>(static_cast<int64_t>(Hs * Toh) - pad_h)},
          d_W_clip{static_cast<int32_t>(We - pad_w)},

          o_H_clip{static_cast<int32_t>(Hs * Toh)},
          o_W_clip{static_cast<int32_t>(We - (Tw - Tow) / 2)},

          out_Hs{static_cast<uint16_t>(Hs)},
          out_We{static_cast<uint16_t>(We)},

          d_H_clip_bot_neg{static_cast<int16_t>(d_H_clip)},
          d_W_clip_bot_neg{static_cast<int16_t>(d_W_clip)},

          o_H_clip_bot_neg{static_cast<int16_t>(o_H_clip)},
          o_W_clip_bot_neg{static_cast<int16_t>(o_W_clip)},

          d_H_window{static_cast<uint16_t>(H)},
          d_W_window{static_cast<uint16_t>(W)},

          o_H_window{static_cast<uint16_t>(out_h)},
          o_W_window{static_cast<uint16_t>(out_w)},

          n_groups{static_cast<uint16_t>(groups)}
    {
        is_applicable = problem.IsFp16() && problem.Is2d() && problem.GetGroupCount() == 1 &&
                        problem.GetInLayout() == "NCHW";
    }

    float GetWTI() const { return -2.f; } // unknown

    bool isApplicable() const
    {
        return is_applicable
               // clang-format off
               && U == 1
               && V == 1
               && input_stride_h == 1
               && input_stride_w == 1
               && filter_stride_h == 1
               && filter_stride_w == 1
               && C <= 16
               && K <= 16
               && S <= 3
               && R <= 3
               && N < std::numeric_limits<uint16_t>::max()
               && pad_h <= std::numeric_limits<int16_t>::max()
               && pad_h >= std::numeric_limits<int16_t>::lowest()
               && pad_w <= std::numeric_limits<int16_t>::max()
               && pad_w >= std::numeric_limits<int16_t>::lowest()
               && n_groups < std::numeric_limits<uint16_t>::max()
               && static_cast<uint32_t>(out_Hs) == Hs
               && static_cast<uint32_t>(out_We) == We
               && static_cast<int32_t>(d_H_clip_bot_neg) == d_H_clip
               && static_cast<int32_t>(d_W_clip_bot_neg) == d_W_clip
               && static_cast<int32_t>(o_H_clip_bot_neg) == o_H_clip
               && static_cast<int32_t>(o_W_clip_bot_neg) == o_W_clip
               && static_cast<uint32_t>(d_H_window) == H
               && static_cast<uint32_t>(d_W_window) == W
               && static_cast<uint32_t>(o_H_window) == out_h
               && static_cast<uint32_t>(o_W_window) == out_w;
        // clang-format on
    }
};

} // namespace

PerformanceConfigConvWinoFuryRxS::PerformanceConfigConvWinoFuryRxS(int n_groups_)
    : n_groups(n_groups_)
{
}

void PerformanceConfigConvWinoFuryRxS::HeuristicInit(const ConvolutionContext& ctx,
                                                     const ProblemDescription& /*problem*/,
                                                     size_t /*Winodata*/,
                                                     size_t /*Winofilter*/)
{
    n_groups = ctx.GetStream().GetMaxHardwareComputeUnits();
}

bool PerformanceConfigConvWinoFuryRxS::SetNextValue(const ProblemDescription&)
{
    return !PerfFieldRules().Next(*this);
}

bool PerformanceConfigConvWinoFuryRxS::IsValidValue() const { return PerfFieldRules().IsIn(*this); }

bool PerformanceConfigConvWinoFuryRxS::IsValid(const ConvolutionContext& ctx) const
{
    if(ctx.GetStream().GetMaxHardwareComputeUnits() < n_groups)
        return false;

    if(!IsValidValue())
        return false;
    return true;
}

bool PerformanceConfigConvWinoFuryRxS::operator==(
    const PerformanceConfigConvWinoFuryRxS& other) const
{
    return n_groups == other.n_groups;
}

template <size_t Winodata, size_t Winofilter>
PerformanceConfigConvWinoFuryRxS ConvWinoFuryRxS<Winodata, Winofilter>::GetDefaultPerformanceConfig(
    const ConvolutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigConvWinoFuryRxS pp;
    pp.HeuristicInit(ctx, problem, Winodata, Winofilter);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

template <size_t Winodata, size_t Winofilter>
bool ConvWinoFuryRxS<Winodata, Winofilter>::IsValidPerformanceConfig(
    const ConvolutionContext& ctx,
    const ProblemDescription&,
    const PerformanceConfigConvWinoFuryRxS& config) const
{
    return config.IsValidValue() && config.IsValid(ctx);
}

template <size_t Winodata, size_t Winofilter>
PerformanceConfigConvWinoFuryRxS
ConvWinoFuryRxS<Winodata, Winofilter>::Search(const ConvolutionContext& ctx,
                                              const ProblemDescription& problem,
                                              const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

template <size_t Winodata, size_t Winofilter>
bool ConvWinoFuryRxS<Winodata, Winofilter>::IsApplicable(const ConvolutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    if(is2x3() && miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3{}))
        return false;

    if(is3x2() && miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2{}))
        return false;

    if(!ctx.use_asm_kernels)
        return false;
    if(!ctx.rmv.IsV3())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const auto name = ctx.GetStream().GetDeviceName();
    if(!StartsWith(name, "gfx11"))
        return false;

    int n_groups = ctx.GetStream().GetMaxHardwareComputeUnits();
    return ShaderModel(problem, n_groups, Winodata, Winofilter).isApplicable();
}

template <size_t Winodata, size_t Winofilter>
float ConvWinoFuryRxS<Winodata, Winofilter>::GetWti(const ConvolutionContext& ctx,
                                                    const ProblemDescription& problem) const
{
    int n_groups = ctx.GetStream().GetMaxHardwareComputeUnits();
    return ShaderModel(problem, n_groups, Winodata, Winofilter).GetWTI();
}

template <size_t Winodata, size_t Winofilter>
ConvSolution ConvWinoFuryRxS<Winodata, Winofilter>::GetSolution(
    const ConvolutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvWinoFuryRxS& config) const
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool IsWarned = false;
    if(!IsWarned)
    {
        if(ctx.GetStream().GetMaxHardwareComputeUnits() > max_cu_limit)
            MIOPEN_LOG_WE(SolverDbId()
                          << ": GPU has " << ctx.GetStream().GetMaxHardwareComputeUnits()
                          << "CUs, but this solver supports max " << max_cu_limit
                          << "and thus may show sub-optimal performance.");
        IsWarned = true;
    }

    ConvSolution result;

    const PerformanceConfigConvWinoFuryRxS* pcfg = &config;

    const auto fromEnv = GetPerfConfFromEnv<Winodata, Winofilter>(ctx);
    if(fromEnv)
    {
        pcfg = &(*fromEnv);
    }

    const auto name = ctx.GetStream().GetDeviceName();
    size_t wg_size  = 384;

    KernelInfo kernel;

    kernel.g_wk.push_back(wg_size * pcfg->GetNGroups());
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };

    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
    kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");

    std::stringstream kernel_name;
    std::stringstream kernel_file;

    std::string_view kernel_version = "_v1_0_0";
    kernel_name << "miopenSp3AsmConv_fury" << kernel_version;
    kernel_file << "Conv_Winograd_Fury" << kernel_version;

    kernel_name << "_gfx11";

    if(problem.IsFp32())
    {
        kernel_name << "_fp32";
        kernel_file << "_fp32";
    }
    else // if(problem.IsFp16())
    {
        kernel_name << "_fp16_fp16acc";
        kernel_file << "_fp16_fp16acc";
    }

    std::string_view kernel_postfix = is2x3() ? "_f2x3" : "_f3x2";

    kernel_name << kernel_postfix;
    kernel_file << kernel_postfix;

    if(problem.GetKernelStrideW() == 1)
    {
        kernel_name << "_stride1";
        kernel_file << "_stride1";
    }
    else if(problem.GetKernelStrideW() == 2 && !problem.direction.IsBackwardData())
    {
        kernel_name << "_stride2";
        kernel_file << "_stride2";
    }
    else // if(problem.GetDilationH() == 2)
    {
        kernel_name << "_dilation2";
        kernel_file << "_dilation2";
    }

    kernel.kernel_name = kernel_name.str();
    kernel_file << ".s";
    kernel.kernel_file = kernel_file.str();

    result.construction_params.push_back(kernel);

    constexpr uint8_t IDENTITY = 0;
    // todo: support fused version
    // constexpr uint8_t LEAKY_RELU = 1;
    // constexpr uint8_t SIGMOID    = 2;
    // constexpr uint8_t SCALED_TANH = 3;
    uint8_t activation_mode = IDENTITY;

    uint32_t flags = 0;

    const bool is_forward = problem.direction.IsForward();
    const bool is_backWrW = problem.direction.IsBackwardWrW();
    const int group_cnt   = problem.GetGroupCount();

    MemLayout_t d_layout, o_layout, f_layout;
    if(!is_backWrW)
    {
        d_layout = GetGroupConvLayout(GetMemLayout_t(problem.GetInLayout()), true);
        o_layout = GetGroupConvLayout(GetMemLayout_t(problem.GetOutLayout()), true);
        f_layout = GetGroupConvLayout(
            is_forward ? MemLayout_t::NCHW : GetSwappedNCLayout(MemLayout_t::NCHW), false);
    }
    else
    {
        d_layout =
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.GetInLayout())), true);
        o_layout =
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.GetOutLayout())), false);
        f_layout = GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true);
    }

    ShaderModel conv(problem, config.n_groups, Winodata, Winofilter);

    BuffInfo d_buf(d_layout,
                   conv.N,
                   conv.C,
                   conv.d_H_window,
                   conv.d_W_window,
                   group_cnt,
                   GetTypeSize(problem.GetInDataType()));
    BuffInfo o_buf(o_layout,
                   conv.N,
                   conv.K,
                   conv.out_h,
                   conv.out_w,
                   group_cnt,
                   GetTypeSize(problem.GetOutDataType()));
    BuffInfo f_buf(f_layout,
                   conv.K,
                   conv.C,
                   conv.R,
                   conv.S,
                   group_cnt,
                   GetTypeSize(problem.GetWeightsDataType()));

    const auto d_strides = d_buf.stride;
    const auto o_strides = o_buf.stride;

    auto f_strides       = f_buf.stride;
    f_strides.h          = is_forward || is_backWrW ? f_strides.h : -f_strides.h;
    f_strides.w          = is_forward || is_backWrW ? f_strides.w : -f_strides.w;
    uint32_t f_RS_offset = is_forward ? 0 : ((conv.R - 1) * conv.S + (conv.S - 1));

    result.invoker_factory = [=](std::vector<Kernel> kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto k = handle.Run(kernels[0]);
            const auto data_tensors =
                !is_backWrW ? primitive_params.CastTo<conv::DataInvokeParams>().tensors.in
                            : primitive_params.CastTo<conv::WrWInvokeParams>().tensors.x;
            const auto filter_tensors =
                !is_backWrW ? primitive_params.CastTo<conv::DataInvokeParams>().tensors.w
                            : primitive_params.CastTo<conv::WrWInvokeParams>().tensors.dy;
            const auto out_tensors =
                !is_backWrW ? primitive_params.CastTo<conv::DataInvokeParams>().tensors.out
                            : primitive_params.CastTo<conv::WrWInvokeParams>().tensors.dw;

            // clang-format off
            MIOPEN_LOG_I2(
              "\nn_groups=" << conv.n_groups <<
              "\nout_We=" << conv.out_We <<
              "\nout_Hs=" << conv.out_Hs <<
              "\nd_W_window=" << conv.d_W_window <<
              "\nd_H_window=" << conv.d_H_window <<
              "\nd_W_clip_bot_neg=" << conv.d_W_clip_bot_neg <<
              "\nd_H_clip_bot_neg=" << conv.d_H_clip_bot_neg <<
              "\nd_strides.nk=" << d_strides.nk <<
              "\nd_strides.h=" << d_strides.h <<
              "\nd_strides.c=" << d_strides.c <<
              "\no_W_window=" << conv.o_W_window <<
              "\no_H_window=" << conv.o_H_window <<
              "\no_W_clip_bot_neg=" << conv.o_W_clip_bot_neg <<
              "\no_H_clip_bot_neg=" << conv.o_H_clip_bot_neg <<
              "\no_strides.nk=" << o_strides.nk <<
              "\no_strides.h=" << o_strides.h <<
              "\no_strides.c=" << o_strides.c <<
              "\nN=" << conv.N <<
              "\nK=" << conv.K <<
              "\nC=" << conv.C <<
              "\nR=" << conv.R <<
              "\nS=" << conv.S <<
              "\nf_strides.nk=" << f_strides.nk <<
              "\nf_strides.c=" << f_strides.c <<
              "\nf_strides.h=" << f_strides.h <<
              "\nf_strides.w=" << f_strides.w <<
              "\nf_RS_offset=" << f_RS_offset <<
              "\nflags=" << flags <<
              "\nactivation_mode=" << static_cast<uint32_t>(activation_mode) <<
              "\n");
            // clang-format on

            k(conv.n_groups,
              conv.out_We,
              conv.out_Hs,
              conv.d_W_window,
              conv.d_H_window,
              conv.d_W_clip_bot_neg,
              conv.d_H_clip_bot_neg,
              static_cast<uint32_t>(d_strides.nk),
              static_cast<uint32_t>(d_strides.h), // todo: check < 2^16
              static_cast<uint32_t>(d_strides.c),
              conv.o_W_window,
              conv.o_H_window,
              conv.o_W_clip_bot_neg,
              conv.o_H_clip_bot_neg,
              static_cast<uint32_t>(o_strides.nk),
              static_cast<uint32_t>(o_strides.h), // todo: check < 2^16
              static_cast<uint32_t>(o_strides.c),
              data_tensors,
              out_tensors,
              filter_tensors,
              static_cast<uint32_t>(conv.N),
              static_cast<uint32_t>(conv.K),
              static_cast<uint32_t>(conv.C),
              static_cast<uint32_t>(conv.R),
              static_cast<uint32_t>(conv.S),
              static_cast<uint32_t>(f_strides.nk),
              static_cast<uint32_t>(f_strides.c),
              static_cast<int32_t>(f_strides.h),
              static_cast<int32_t>(f_strides.w),
              f_RS_offset,
              nullptr,
              nullptr,
              flags,
              activation_mode,
              static_cast<uint8_t>(0),
              static_cast<uint16_t>(0),
              0.0f);
        };
    };

    return result;
}

template struct ConvWinoFuryRxS<2, 3>;
// template struct ConvWinoFuryRxS<3, 2>;

} // namespace solver
} // namespace miopen
