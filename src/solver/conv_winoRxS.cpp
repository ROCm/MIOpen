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

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/sequences.hpp>
#include <miopen/stringutils.hpp>

#include <boost/any.hpp>
#include <boost/optional.hpp>

#include <tuple>

// ConvBinWinoRxS<2,3> is intended to handle group convolutions, but
// it is able to work with non-group convolutions and shows quite good performance,
// comparable with ConvBinWinogradRxSf2x3g1. Due to the issue in #1533 (fixed in #1634)
// AND some inaccuracy (jitter) of benchmarking, it is often shows the "best" performance
// with non-group convolutions, and therefore unexpected records where ConvBinWinoRxS<2,3>
// is the "best" leaked into find-db. That is why disabling ConvBinWinoRxS<2,3> for non-group
// convolutions leads to performance drops. Let's enable ConvBinWinoRxS<2,3> for non-group
// in order to quickly W/A the perf issue. When find-db fix is ready,
// we will keep ConvBinWinoRxS<2,3> for group convolutions only.
#define WORKAROUND_ISSUE_1681 0

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS)

#define MAX_CU_LIMIT 512

#define IS2X3 (Winodata == 2 && Winofilter == 3)
#define IS3X2 (Winodata == 3 && Winofilter == 2)

/// \todo The model is well-defined in for filters sized up to 5.
/// However, it seems producing valid results without this limitation,
/// when used against simple GEMM WTI model (to select the fastest solver).
/// This needs to be re-tested/re-considered when we have WTI
/// models for other solvers, OR when GEMM WTI model is improved.
/// --atamazov 2020-11-07.
#define WTI_MODEL_ALLOW_ANY_RS 1

static inline size_t Ceil(const size_t v, const size_t m)
{
    assert(m > 0);
    return (v + m - 1) / m;
}

static inline size_t RoundUpToMultiple(size_t val, size_t factor)
{
    return Ceil(val, factor) * factor;
}

// Let's avoid clang-tidy warnings without explicit casts.
static inline size_t RoundUpToMultiple(size_t val, int factor)
{
    return Ceil(val, factor) * factor;
}

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
                                     const int G,
                                     const int o_tile,
                                     const int f_tile)
{
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
        size_t n_groups_per_cu      = Ceil(static_cast<size_t>(i) * G, n_groups);
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
            std::make_tuple(seq::Span<int, 1, MAX_CU_LIMIT>{}, &PerformanceConfigConvBinWinogradRxS::n_groups)
        );
    }
// clang-format on

inline bool IsShaderContraintsMet(const ProblemDescription& problem,
                                  const int R,
                                  const int S,
                                  const int C,
                                  const int K,
                                  const int H,
                                  const int W,
                                  const int OH,
                                  const int OW,
                                  const int N)
{
    // Padding for bwd data shall not be negative.
    /// \todo Either remove WrW related code or re-use function from RxS
    if(problem.direction.IsBackwardData())
    {
        if(!(0 <= problem.GetBackwardPadW() && problem.GetBackwardPadW() < std::pow(2, 16)))
            return false;
        if(!(0 <= problem.GetBackwardPadH() && problem.GetBackwardPadH() < std::pow(2, 16)))
            return false;
    }
    if(!problem.IsLayoutDefault())
    {
        return false;
    }

    uint64_t o_K_stride      = static_cast<uint64_t>(OH) * OW;
    uint64_t o_N_stride      = o_K_stride * K;
    uint64_t o_N_stride_OHOW = o_N_stride + o_K_stride;

    uint64_t d_C_stride    = static_cast<uint64_t>(H) * W;
    uint64_t d_N_stride    = d_C_stride * C;
    uint64_t d_N_stride_HW = d_N_stride + d_C_stride;

    auto num_tiles  = Ceil(OH, 2) * Ceil(OW, 2);
    auto stride_one = problem.kernel_stride_h == 1 && problem.kernel_stride_w == 1 &&
                      problem.kernel_dilation_h == 1 && problem.kernel_dilation_w == 1;

    // clang-format off
    // Check implementation limits.
    return N < std::pow(2, 16)
        && C < std::pow(2, 16)
        && H < std::pow(2, 16)
        && W < std::pow(2, 16)
        && K < std::pow(2, 16)
        && S < std::pow(2, 16)
        && R < std::pow(2, 16)
        && OH < std::pow(2, 16)
        && OW < std::pow(2, 16)
        && problem.pad_w < std::pow(2, 16)
        && problem.pad_h < std::pow(2, 16)
        && C * R * S < std::pow(2, 22)
        && K * R * S < std::pow(2, 28)
        && ((o_N_stride_OHOW < std::pow(2, 29) && d_N_stride_HW < std::pow(2, 29))
           || (stride_one && o_N_stride < std::pow(2, 30) && d_N_stride < std::pow(2, 30)
           && (N == 1 || num_tiles % 16 == 0)));
}

} // namespace

PerformanceConfigConvBinWinogradRxS::PerformanceConfigConvBinWinogradRxS(int n_groups_)
    : n_groups(n_groups_)
{
}

template <int Winodata, int Winofilter>
void PerformanceConfigConvBinWinogradRxS::HeuristicInit(const ConvolutionContext& ctx,
                                                        const ProblemDescription& problem)
{
    const auto n_inputs_per_group  = problem.n_inputs / problem.group_counts,
               n_outputs_per_group = problem.n_outputs / problem.group_counts;
    if(problem.group_counts == 1)
    {
        n_groups = ctx.GetStream().GetMaxHardwareComputeUnits();
        return;
    }

    if(problem.direction.IsBackwardWrW())
    {
        n_groups = GetBestNGroupParam(problem.in_height,
                                      problem.in_width,
                                      problem.kernel_dilation_h,
                                      problem.kernel_dilation_w,
                                      problem.batch_sz,   // N
                                      n_inputs_per_group, // K
                                      problem.kernel_size_h,
                                      problem.kernel_size_w,
                                      problem.pad_w,
                                      problem.pad_h,
                                      n_outputs_per_group, // C
                                      problem.kernel_stride_h,
                                      problem.kernel_stride_w,
                                      ctx.GetStream().GetMaxHardwareComputeUnits(),
                                      problem.group_counts,
                                      Winodata,
                                      Winofilter);
    }
    else
    {
        n_groups = GetBestNGroupParam(problem.kernel_size_h, // RxS
                                      problem.kernel_size_w,
                                      problem.kernel_stride_h,
                                      problem.kernel_stride_w,
                                      n_inputs_per_group,  // C
                                      n_outputs_per_group, // K
                                      problem.out_height,  // OHxOW
                                      problem.out_width,
                                      problem.pad_w,
                                      problem.pad_h,
                                      problem.batch_sz, // N
                                      problem.kernel_dilation_h,
                                      problem.kernel_dilation_w,
                                      ctx.GetStream().GetMaxHardwareComputeUnits(),
                                      problem.group_counts,
                                      Winodata,
                                      Winofilter);
    }
}

bool PerformanceConfigConvBinWinogradRxS::SetNextValue(const ConvolutionContext& /*ctx*/)
{
    return !PerfFieldRules().Next(*this);
}

bool PerformanceConfigConvBinWinogradRxS::IsValidValue() const
{
    return PerfFieldRules().IsIn(*this);
}

bool PerformanceConfigConvBinWinogradRxS::IsValid(const ConvolutionContext& ctx) const
{
    if(ctx.GetStream().GetMaxHardwareComputeUnits() < n_groups)
        return false;

    if(!IsValidValue())
        return false;
    return true;
}

bool PerformanceConfigConvBinWinogradRxS::operator==(
    const PerformanceConfigConvBinWinogradRxS& other) const
{
    return n_groups == other.n_groups;
}

template <int Winodata, int Winofilter>
PerformanceConfigConvBinWinogradRxS
ConvBinWinoRxS<Winodata, Winofilter>::GetDefaultPerformanceConfig(
    const ConvolutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigConvBinWinogradRxS pp;
    pp.HeuristicInit<Winodata, Winofilter>(ctx, problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

template <int Winodata, int Winofilter>
bool ConvBinWinoRxS<Winodata, Winofilter>::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceConfigConvBinWinogradRxS& config) const
{
    return config.IsValidValue() && config.IsValid(ctx);
}

template <int Winodata, int Winofilter>
PerformanceConfigConvBinWinogradRxS
ConvBinWinoRxS<Winodata, Winofilter>::Search(const ConvolutionContext& ctx,
                                             const ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

template <int Winodata, int Winofilter>
class ShaderModel : public UnifiedDescriptionConv2d
{
    const size_t DATATYPE_BITS;    // S
    const size_t n_groups;         // BQ ~compute units
    const bool out_of_model_scope; // Shader model produces unreliable results.

public:
    ShaderModel(const ConvolutionContext& ctx, const ProblemDescription& problem)
        : UnifiedDescriptionConv2d(problem),
          DATATYPE_BITS(problem.IsFp16() ? 16 : 32),
          n_groups(ctx.GetStream()
                       .GetMaxHardwareComputeUnits()), /// \todo Take n_groups from PerfConfig.
          out_of_model_scope(!(problem.group_counts == 1) || //
                             !(U == 1) ||                    //
                             !(V == 1) ||                    //
                             !(input_stride_h == 1) ||       //
                             !(input_stride_w == 1) ||       //
                             !(filter_stride_h == 1) ||      //
                             !(filter_stride_w == 1) ||      //
#if !WTI_MODEL_ALLOW_ANY_RS
                             !(R <= 5) || //
                             !(S <= 5) || //
#endif
                             !(C >= 16) || //
                             !(K >= 16))
    {
        // Computations do not support negative padding.
        // Negative padding is not applicable, so let use simple assert here.
        assert(pad_h >= 0 && pad_w >= 0);
    }

    double ComputeWti() const
    {
        if(out_of_model_scope)
            return -1.0; // Shader model produces unreliable results.

        const auto direct_convolution_macs =
            static_cast<double>(C * N * K) / 1e+6 *
            static_cast<double>(RoundUpToMultiple(S * out_w / input_stride_w, 1)) *
            static_cast<double>(RoundUpToMultiple(R * out_h / input_stride_h, 1)); // AK

        constexpr size_t TILE_S = Winofilter; // AL
        constexpr size_t TILE_R = Winofilter; // AO
        assert(!(U > 2 && V > 2));
        const auto granulated_S =
            (U == 1 && input_stride_w == 1 && filter_stride_w == 1 && S <= TILE_S)
                ? TILE_S
                : RoundUpToMultiple(S, 2 * TILE_S); // AM
        const auto granulated_R = RoundUpToMultiple(
            R,
            (((V == 1 && input_stride_h == 1 && filter_stride_h == 1) || (R % (2 * TILE_R) == 1))
                 ? TILE_R
                 : 2 * TILE_R)); // AP

        constexpr size_t TILE_OUT_W = Winodata; // AR
        constexpr size_t TILE_OUT_H = Winodata; // AU
        const auto granulated_out_w =
            RoundUpToMultiple(out_w + ((input_stride_w == 2 && (pad_w % 2 != 0)) ? 1 : 0),
                              TILE_OUT_W * input_stride_w); // AS
        const auto granulated_out_h =
            RoundUpToMultiple(out_h + ((input_stride_h == 2 && (pad_h % 2 != 0)) ? 1 : 0),
                              TILE_OUT_H * input_stride_h); // AV

        constexpr size_t GRANULARITY_NHW_TILES = 32; // AY$2
        constexpr size_t GRANULARITY_K         = 32; // BC$2

        const auto NWH_tiles =
            granulated_out_w * granulated_out_h * N / TILE_OUT_H / TILE_OUT_W; // AX

        const auto granulated_NWH_tiles = RoundUpToMultiple(
            NWH_tiles,
            GRANULARITY_NHW_TILES * ((input_stride_w == 2 && input_stride_h == 2) ? 2 : 1)); // AY

        const auto granulated_C =
            RoundUpToMultiple(C,
                              ((U == 1 && S <= 3) ? 2 : 1) * 32 / DATATYPE_BITS); // BA

        const auto granulated_K = RoundUpToMultiple(
            K,
            GRANULARITY_K / ((input_stride_w == 2 && input_stride_h == 2) ? 2 : 1)); // BC

        const auto NKWH_tiles = granulated_NWH_tiles * granulated_K; // BE

        const auto granulated_NKWH_tiles =
            RoundUpToMultiple(NKWH_tiles,
                              n_groups * GRANULARITY_NHW_TILES * GRANULARITY_K); // BR

        const auto works_per_CU = granulated_NKWH_tiles / 32 / n_groups; // BY

        constexpr size_t MIN_FE_PER_WORK = 20; // BZ$2

        const auto fe_per_work =
            std::max(MIN_FE_PER_WORK,
                     granulated_S * granulated_R * granulated_C * DATATYPE_BITS / 32); // BZ

        const auto phases   = fe_per_work * works_per_CU; // CA
        const auto fe_calls = phases;                     // CC
        const auto be_calls = works_per_CU;               // CD

        constexpr double C0      = 43283;                                       // CB$2
        constexpr double C1      = 1.012;                                       // CC$2
        constexpr double C2      = 134.14;                                      // CD$2
        const auto GUI_predicted = (C0 + C1 * fe_calls + C2 * be_calls) / 1e+6; // CE

        if(GUI_predicted <= 0.1)
            return -1.0; // Unreliable, too small work to do for the shader.

        const auto N_MACS_PER_CU_PER_CLOCK = static_cast<size_t>(64) * 32 / DATATYPE_BITS;
        const auto WTI_predicted           = direct_convolution_macs /
                                   static_cast<double>(N_MACS_PER_CU_PER_CLOCK) /
                                   static_cast<double>(n_groups) / GUI_predicted; // similar to BW
        return WTI_predicted;
    }
};

template <int Winodata, int Winofilter>
static float GetWtiBase(const ConvolutionContext& ctx, const ProblemDescription& problem)
{
    constexpr auto WTI_UNKNOWN = -2.0;
    const auto rv              = ShaderModel<Winodata, Winofilter>(ctx, problem).ComputeWti();
    return rv < 0 ? WTI_UNKNOWN : rv;
}

static bool IsApplicableBase(const ConvolutionContext& ctx, const ProblemDescription& problem)
{
    if(!problem.Is2d())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16()))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!ctx.rmv.IsV3())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const auto name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9") || StartsWith(name, "gfx10")))
        return false;
    if(problem.IsFp16() &&
       !(StartsWith(name, "gfx906") || StartsWith(name, "gfx908") || StartsWith(name, "gfx90a") ||
         StartsWith(name, "gfx1011") || StartsWith(name, "gfx1012") || StartsWith(name, "gfx103")))
        return false;

    if(name == "gfx90a" && problem.conv_problem.IsGfx90aFp16altRequired())
        return false;

    // clang-format off
    if (!((problem.kernel_stride_w == 1 || problem.kernel_stride_w == 2)
        && problem.kernel_stride_w == problem.kernel_stride_h
        && problem.kernel_dilation_w == 1
        && problem.kernel_dilation_h == 1
        && problem.bias == 0
        && problem.in_layout == "NCHW"))
        return false;
    // clang-format on

    const auto n_inputs_per_group  = problem.n_inputs / problem.group_counts,
               n_outputs_per_group = problem.n_outputs / problem.group_counts;

    if(problem.direction.IsBackwardWrW())
    {
        if(problem.kernel_stride_w == 2)
            return false;
        return IsShaderContraintsMet(problem,
                                     problem.in_height,
                                     problem.in_width,
                                     problem.batch_sz,   // N
                                     n_inputs_per_group, // K
                                     problem.out_height,
                                     problem.out_width,
                                     problem.kernel_size_h,
                                     problem.kernel_size_w,
                                     n_outputs_per_group); // C
    }
    else
    {
        return IsShaderContraintsMet(problem,
                                     problem.kernel_size_h, // RxS
                                     problem.kernel_size_w,
                                     n_inputs_per_group,  // C
                                     n_outputs_per_group, // K
                                     problem.in_height,   // HxW
                                     problem.in_width,
                                     problem.out_height, // OHxOW
                                     problem.out_width,
                                     problem.batch_sz); // N
    }
}

template <int Winodata, int Winofilter>
bool ConvBinWinoRxS<Winodata, Winofilter>::IsApplicable(const ConvolutionContext& ctx,
                                                        const ProblemDescription& problem) const
{
    if(IS2X3)
    {
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3{}))
            return false;
#if !WORKAROUND_ISSUE_1681
        if(problem.group_counts == 1 && !problem.direction.IsBackwardWrW())
            return false;
#endif
    }
    if(IS3X2)
    {
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2{}))
            return false;
        if(problem.kernel_stride_w == 2) // f3x2 stride 2 not implemented yet
            return false;
    }
    return IsApplicableBase(ctx, problem);
}

template <int Winodata, int Winofilter>
static inline boost::optional<PerformanceConfigConvBinWinogradRxS>
GetPerfConfFromEnv(const ConvolutionContext& ctx)
{
    PerformanceConfigConvBinWinogradRxS fromEnv;
    std::string s;
    const char* p_asciz = nullptr;
    const char* env_name;

    if(IS2X3)
    {
        p_asciz  = miopen::GetStringEnv(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS{});
        env_name = MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS::value();
    }
    else if(IS3X2)
    {
        p_asciz  = miopen::GetStringEnv(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS{});
        env_name = MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS::value();
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

template <int Winodata, int Winofilter>
ConvSolution ConvBinWinoRxS<Winodata, Winofilter>::GetSolution(
    const ConvolutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvBinWinogradRxS& config) const
{
    const auto n_groups = config.n_groups;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool IsWarned;
    if(!IsWarned)
    {
        if(ctx.GetStream().GetMaxHardwareComputeUnits() > MAX_CU_LIMIT)
            MIOPEN_LOG_WE(SolverDbId()
                          << ": GPU has " << ctx.GetStream().GetMaxHardwareComputeUnits()
                          << "CUs, but this solver supports max " << MAX_CU_LIMIT
                          << "and thus may show sub-optimal performance.");
        IsWarned = true;
    }

    ConvSolution result;

    const PerformanceConfigConvBinWinogradRxS* pcfg = &config;

    const auto fromEnv = GetPerfConfFromEnv<Winodata, Winofilter>(ctx);
    if(fromEnv)
    {
        pcfg = &(*fromEnv);
    }

    const auto name    = ctx.GetStream().GetDeviceName();
    const auto is_gfx9 = StartsWith(name, "gfx9");
    size_t wg_size     = is_gfx9 ? 512 : 256;

    KernelInfo kernel;

    kernel.g_wk.push_back(wg_size * pcfg->GetNGroups() * problem.group_counts);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    std::string kernel_name = "miopenSp3AsmConv_v21_1_3";
    std::string kernel_file = "Conv_Winograd_v21_1_3";
    std::string kernel_postfix;

    if(is_gfx9)
    {
        kernel_name += "_gfx9";
    }
    else // if(StartsWith(name, "gfx10"))
    {
        kernel_name += "_gfx10";
        kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");
    }

    if(IS2X3)
    {
        kernel_postfix = problem.IsFp32() ? "_fp32" : "_fp16_dot2_edc";

        if(problem.kernel_stride_w == 1)
        {
            kernel_postfix += "_stride1";
        }
        else if(problem.kernel_stride_w == 2 && !problem.direction.IsBackwardData())
        {
            kernel_postfix += "_stride2";
        }
        else // if(problem.kernel_dilation_h == 2)
        {
            kernel_postfix += "_dilation2";
        }
        if(problem.group_counts != 1 || problem.direction.IsBackwardWrW())
        {
            kernel_postfix += "_group";
        }
    }
    else if(IS3X2)
    {
        kernel_postfix = problem.IsFp32() ? "_f3x2_fp32" : "_f3x2_fp16_dot2_edc";
        kernel_postfix += "_stride1"; // f3x2 stride 2 is not implemented yet
        kernel_postfix += "_group";
    }

    kernel.kernel_name = kernel_name + kernel_postfix;
    kernel.kernel_file = kernel_file + kernel_postfix + ".s";

    result.construction_params.push_back(kernel);

    if(!problem.direction.IsBackwardWrW())
    {
        const bool is_forward     = problem.direction.IsForward();
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
            ctx, problem, &N, &C, &H, &W, &K, &ignore, &out_H, &out_W, &R, &S, &pad_H, &pad_W);
        const auto group_cnt = problem.group_counts;
        C                    = C / group_cnt;
        K                    = K / group_cnt;
        int flags            = is_forward ? 0 : F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
        flags |= L_F_NKC_STRIDES + L_F_GROUP_STRIDES;

        // cppcheck-suppress unreadVariable
        BuffInfo d_buf(GetGroupConvLayout(GetMemLayout_t(problem.in_layout), true),
                       N,
                       C,
                       H,
                       W,
                       group_cnt,
                       GetTypeSize(problem.in_data_type)),
            // cppcheck-suppress unreadVariable
            o_buf(GetGroupConvLayout(GetMemLayout_t(problem.out_layout), true),
                  N,
                  K,
                  out_H,
                  out_W,
                  group_cnt,
                  GetTypeSize(problem.out_data_type)),
            // cppcheck-suppress unreadVariable
            f_buf(GetGroupConvLayout(is_forward ? (MemLayout_t::NCHW)
                                                : GetSwappedNCLayout(MemLayout_t::NCHW),
                                     false),
                  K,
                  C,
                  R,
                  S,
                  group_cnt,
                  GetTypeSize(problem.weights_data_type));

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
            ctx, problem, &C, &K, &R, &S, &N, &unused, &H, &W, &out_H, &out_W, &unused, &unused);
        const auto group_cnt             = problem.group_counts;
        static const int F_NKC_STRIDES   = 1 << 9;
        static const int F_GROUP_STRIDES = 1 << 10;
        int flags                        = F_NKC_STRIDES + F_GROUP_STRIDES;
        N                                = N / group_cnt;
        K                                = K / group_cnt;
        int pad_H                        = problem.conv_problem.GetConv().GetConvPads()[0];
        int pad_W                        = problem.conv_problem.GetConv().GetConvPads()[1];

        BuffInfo d_buf(
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.in_layout)), true),
            N,
            C,
            H,
            W,
            group_cnt,
            GetTypeSize(problem.in_data_type)),
            o_buf(GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.out_layout)), false),
                  N,
                  K,
                  out_H,
                  out_W,
                  group_cnt,
                  GetTypeSize(problem.out_data_type)),
            f_buf(GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true),
                  K,
                  C,
                  R,
                  S,
                  group_cnt,
                  GetTypeSize(problem.weights_data_type));

        decltype(auto) batch_sz = problem.batch_sz;
        decltype(auto) n_inputs = problem.n_inputs;

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

    return result;
}

bool ConvBinWinogradRxSf2x3g1::IsApplicable(const ConvolutionContext& ctx,
                                            const ProblemDescription& problem) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1{}))
        return false;
    return IsApplicableBase(ctx, problem) && problem.group_counts == 1;
}

float ConvBinWinogradRxSf2x3g1::GetWti(const ConvolutionContext& ctx,
                                       const ProblemDescription& problem) const
{
    return GetWtiBase<2, 3>(ctx, problem);
}

ConvSolution ConvBinWinogradRxSf2x3g1::GetSolution(const ConvolutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    const auto tunable = ConvBinWinoRxS<2, 3>{};
    return tunable.GetSolution(ctx, problem, tunable.GetDefaultPerformanceConfig(ctx, problem));
}

bool ConvBinWinogradRxSf2x3g1Fused::IsApplicable(const ConvolutionContext&) const
{
    return true; // Actual checks moved to FusionMDGraph.
}

ConvSolution ConvBinWinogradRxSf2x3g1Fused::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;
    KernelInfo kernel;

    const auto n_groups = ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto name     = ctx.GetStream().GetDeviceName();
    const auto is_gfx9  = StartsWith(name, "gfx9");
    size_t wg_size      = is_gfx9 ? 512 : 256;
    kernel.g_wk.push_back(wg_size * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
    if(!is_gfx9)
        kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");

    // File and name are defined in FusionMDGraph, so no need (and harmful)
    // to duplicate this information here.
    kernel.kernel_name = "<name not set>";
    kernel.kernel_file = "<file not set>";
    result.construction_params.push_back(kernel);
    return result;
}

template struct ConvBinWinoRxS<2, 3>;
template struct ConvBinWinoRxS<3, 2>;

} // namespace solver
} // namespace miopen
