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

#include <miopen/conv/solvers.hpp>

#include <miopen/buffer_info.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/problem_description.hpp>
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

/// \anchor disable_winograd_with_huge_granularity_loss
/// This is to exclude Winograd from PyTorch correctness tests.
/// These tests are written in incorrect manner.
/// They are very sensitive to the values in the output tensors,
/// but do not provide inputs that guarantee minimal deviation
/// (a numerical difference between kernel output and theoretically
/// "ideal" output). This is especially important for Winograd.
/// As a result, the PyTorch correctness tests often produce false
/// failures when Winograd algorithm is used. As PyTorch correctness
/// tests use very small convolutions and Winograd algorithm is
/// ineffective with such small configs due to huge granularity loss,
/// we can disable Winograd without any performance implications.
#define WORKAROUND_ISSUE_2492_GRANULARITY_LOSS 0
/// \anchor disable_winograd_with_small_tensor
/// This is alternative W/A for the same problem, which
/// resolves issues shown at
/// https://github.com/ROCm/MIOpen/issues/2492#issuecomment-1910928563
/// The idea is to disable the solver for very small tensors.
/// PyTorch tests use sizes like 2x2x6x6 (NCHW), but let's add
/// some spare space for N anc C and disable starting from 4x4x6x6.
#define WORKAROUND_ISSUE_2492_TINY_TENSOR 1

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_WORKAROUND_ISSUE_2492)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS)

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
#define WTI_MODEL_ALLOW_ANY_CK 1

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
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

namespace {

auto PerfFieldRules()
{
    return seq::MakeRuleSet(std::make_tuple(seq::Span<int, 1, MAX_CU_LIMIT>{},
                                            &PerformanceConfigConvBinWinogradRxS::n_groups));
}

// Winograd v21 is preferred on Vega10/Vega20 ASICs due to ~25% performance regression with Winograd
// v30. The exception is Winograd F(3,2) stride2 as this mode is unsupported in v21. Details:
// https://github.com/ROCm/MIOpen/pull/1927#issuecomment-1412741130
template <int Winodata, int Winofilter>
inline bool IsWinogradV21Preferred(const std::string& asic, const ProblemDescription& problem)
{
    return (StartsWith(asic, "gfx900") || StartsWith(asic, "gfx906")) &&
           !(IS3X2 && problem.GetKernelStrideW() == 2);
}

inline bool IsShaderConstraintsMetV21(const ProblemDescription& problem,
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
    uint64_t o_K_stride      = static_cast<uint64_t>(OH) * OW;
    uint64_t o_N_stride      = o_K_stride * K;
    uint64_t o_N_stride_OHOW = o_N_stride + o_K_stride;

    uint64_t d_C_stride    = static_cast<uint64_t>(H) * W;
    uint64_t d_N_stride    = d_C_stride * C;
    uint64_t d_N_stride_HW = d_N_stride + d_C_stride;

    auto num_tiles  = Ceil(OH, 2) * Ceil(OW, 2);
    auto stride_one = problem.GetKernelStrideH() == 1 && problem.GetKernelStrideW() == 1 &&
                      problem.GetDilationH() == 1 && problem.GetDilationW() == 1;

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
        && problem.GetPadW() < std::pow(2, 16)
        && problem.GetPadH() < std::pow(2, 16)
        && C * R * S < std::pow(2, 22)
        && K * R * S < std::pow(2, 28)
        && ((o_N_stride_OHOW < std::pow(2, 29) && d_N_stride_HW < std::pow(2, 29))
           || (stride_one && o_N_stride < std::pow(2, 30) && d_N_stride < std::pow(2, 30)
           && (N == 1 || num_tiles % 16 == 0)));
    // clang-format on
}

inline bool IsShaderConstraintsMetV30(const ProblemDescription& problem,
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
        && problem.GetPadW() < std::pow(2, 16)
        && problem.GetPadH() < std::pow(2, 16)
        && H * W < std::pow(2, 29)
        && K * R * S < std::pow(2, 28)
        && (C + 1) * H * W < std::pow(2, 30)
        && (C + 1) * R * S < std::pow(2, 22)
        && (K + 1) * OH * OW < std::pow(2, 30);
    // clang-format on
}

template <int Winodata, int Winofilter>
inline bool IsShaderConstraintsMet(const ProblemDescription& problem,
                                   const int R,
                                   const int S,
                                   const int C,
                                   const int K,
                                   const int H,
                                   const int W,
                                   const int OH,
                                   const int OW,
                                   const int N,
                                   const std::string& asic)
{
    // Padding for bwd data shall not be negative.
    /// \todo Either remove WrW related code or re-use function from RxS
    if(problem.IsDirectionBackwardData())
    {
        if(!(0 <= problem.GetBackwardPadW() && problem.GetBackwardPadW() < std::pow(2, 16)))
            return false;
        if(!(0 <= problem.GetBackwardPadH() && problem.GetBackwardPadH() < std::pow(2, 16)))
            return false;
    }

    if(!problem.IsLayoutDefault())
        return false;

    return IsWinogradV21Preferred<Winodata, Winofilter>(asic, problem)
               ? IsShaderConstraintsMetV21(problem, R, S, C, K, H, W, OH, OW, N)
               : IsShaderConstraintsMetV30(problem, R, S, C, K, H, W, OH, OW, N);
}

} // namespace

PerformanceConfigConvBinWinogradRxS::PerformanceConfigConvBinWinogradRxS(int n_groups_)
    : n_groups(n_groups_)
{
}

template <int Winodata, int Winofilter>
void PerformanceConfigConvBinWinogradRxS::HeuristicInit(const ExecutionContext& ctx,
                                                        const ProblemDescription& problem)
{
    const auto n_inputs_per_group  = problem.GetInChannels() / problem.GetGroupCount(),
               n_outputs_per_group = problem.GetOutChannels() / problem.GetGroupCount();
    if(problem.GetGroupCount() == 1)
    {
        n_groups = ctx.GetStream().GetMaxHardwareComputeUnits();
        return;
    }

    if(problem.IsDirectionBackwardWrW())
    {
        n_groups = GetBestNGroupParam(problem.GetInHeight(),
                                      problem.GetInWidth(),
                                      problem.GetDilationH(),
                                      problem.GetDilationW(),
                                      problem.GetBatchSize(), // N
                                      n_inputs_per_group,     // K
                                      problem.GetWeightsHeight(),
                                      problem.GetWeightsWidth(),
                                      problem.GetPadW(),
                                      problem.GetPadH(),
                                      n_outputs_per_group, // C
                                      problem.GetKernelStrideH(),
                                      problem.GetKernelStrideW(),
                                      ctx.GetStream().GetMaxHardwareComputeUnits(),
                                      problem.GetGroupCount(),
                                      Winodata,
                                      Winofilter);
    }
    else
    {
        n_groups = GetBestNGroupParam(problem.GetWeightsHeight(), // RxS
                                      problem.GetWeightsWidth(),
                                      problem.GetKernelStrideH(),
                                      problem.GetKernelStrideW(),
                                      n_inputs_per_group,     // C
                                      n_outputs_per_group,    // K
                                      problem.GetOutHeight(), // OHxOW
                                      problem.GetOutWidth(),
                                      problem.GetPadW(),
                                      problem.GetPadH(),
                                      problem.GetBatchSize(), // N
                                      problem.GetDilationH(),
                                      problem.GetDilationW(),
                                      ctx.GetStream().GetMaxHardwareComputeUnits(),
                                      problem.GetGroupCount(),
                                      Winodata,
                                      Winofilter);
    }
}

bool PerformanceConfigConvBinWinogradRxS::SetNextValue(const ProblemDescription&)
{
    return !PerfFieldRules().Next(*this);
}

bool PerformanceConfigConvBinWinogradRxS::IsValidValue() const
{
    return PerfFieldRules().IsIn(*this);
}

bool PerformanceConfigConvBinWinogradRxS::IsValid(const ExecutionContext& ctx) const
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
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigConvBinWinogradRxS pp;
    pp.HeuristicInit<Winodata, Winofilter>(ctx, problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

template <int Winodata, int Winofilter>
bool ConvBinWinoRxS<Winodata, Winofilter>::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription&,
    const PerformanceConfigConvBinWinogradRxS& config) const
{
    return config.IsValidValue() && config.IsValid(ctx);
}

template <int Winodata, int Winofilter>
PerformanceConfigConvBinWinogradRxS
ConvBinWinoRxS<Winodata, Winofilter>::Search(const ExecutionContext& ctx,
                                             const ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

namespace {

class ShaderModel : public UnifiedDescriptionConv2d
{
    static constexpr size_t NHW_tiles_factor = 32;

    size_t n_CU;
    size_t n_groups;

    double direct_convolution_macs;

    size_t R_loops;
    size_t S_loops;
    size_t C_loops;

    size_t n_works_per_CU;

    bool dstride2;
    bool ostride2;

    bool single_mode;

    bool is_fp16;
    bool is_2x3;

    bool out_of_model_scope; // Shader model produces unreliable results.
    double granularity_loss;

public:
    ShaderModel(const ExecutionContext& ctx,
                const ProblemDescription& problem,
                size_t Winodata,
                size_t Winofilter)
        : UnifiedDescriptionConv2d(problem),
          n_CU{ctx.GetStream().GetMaxHardwareComputeUnits()}, /// \todo Take n_CU from PerfConfig.
          direct_convolution_macs{static_cast<double>(C * N * K) / 1e+6 *
                                  static_cast<double>(Ceil(S * out_w, input_stride_w) *
                                                      Ceil(R * out_h, input_stride_h))},
          is_fp16{problem.IsFp16()},
          is_2x3{IS2X3},
          out_of_model_scope
    {
        !(problem.GetGroupCount() == 1) //
            || !(U == 1)                //
            || !(V == 1)                //
            || !(input_stride_h == 1)   //
            || !(input_stride_w == 1)   //
            || !(filter_stride_h == 1)  //
            || !(filter_stride_w == 1)  //
#if !WTI_MODEL_ALLOW_ANY_RS
            || !(R <= 5) //
            || !(S <= 5) //
#endif
#if !WTI_MODEL_ALLOW_ANY_CK
            || !(C >= 16) //
            || !(K >= 16)
#endif
    }
    {
        /// \todo add G to UnifiedDescriptionConv2d
        size_t G = static_cast<size_t>(problem.GetGroupCount());

        const size_t Ts = Winofilter;
        const size_t Tr = Winofilter; // Ts and Tr must be the same

        const size_t Tow = Winodata;
        const size_t Toh = Winodata; // Tow and Toh must be the same

        const bool dstride2_w{input_stride_w == 2};
        const bool dstride2_h{input_stride_h == 2};
        dstride2 = dstride2_w && dstride2_h;

        const bool ostride2_w{U == 2};
        const bool ostride2_h{V == 2};
        ostride2 = ostride2_w && ostride2_h;

        const bool stride1_w{!(dstride2_w || ostride2_w)};
        const bool stride1_h{!(dstride2_h || ostride2_h)};

        single_mode =
            stride1_w &&
            (S <= Ts /*|| (L_F_FORCE_FILTER_TRAVERSE_MODE == 1 && L_F_FILTER_TRAVERSE_DUAL == 0)*/);

        const auto R_factor = ((stride1_h || (R % (2 * Tr) == 1)) ? Tr : 2 * Tr);
        const auto Rg       = RoundUpToMultiple(R, R_factor);

        const auto S_factor = single_mode ? Ts : 2 * Ts;
        const auto Sg       = RoundUpToMultiple(S, S_factor);

        const auto C_factor = is_fp16 ? 2 : 1 /*fp32*/;
        const auto Cg       = RoundUpToMultiple(C, C_factor);

        const auto K_factor = dstride2 ? 16u : 32u;
        const auto Kg       = Ceil(K, K_factor);

        const auto OWg = RoundUpToMultiple(out_w + ((dstride2_w && (pad_w % 2 == 0)) ? 1 : 0),
                                           Tow * input_stride_w);
        const auto OHg = RoundUpToMultiple(out_h + ((dstride2_h && (pad_h % 2 == 1)) ? 1 : 0),
                                           Toh * input_stride_h);

        const auto NWH_tiles = N * (OHg / (Toh * input_stride_h)) * (OWg / (Tow * input_stride_w));
        const auto NHW_tiles_g = Ceil(NWH_tiles, NHW_tiles_factor);

        const auto n_works = Kg * NHW_tiles_g;

        S_loops = Sg / S_factor;
        R_loops = Rg / Tr;
        C_loops = Cg / C_factor;

        const auto NKWH_w = K_factor * NHW_tiles_factor * Toh * Tow;
        const auto grid_g = static_cast<double>(NKWH_w * Cg * Rg * Sg) / 1e6;

        auto compute_granularity_loss =
            [n_works, G, grid_g, CU = n_CU, dc_macs = direct_convolution_macs](size_t groups) {
                const auto works_per_CU = Ceil(n_works, groups) * Ceil(G * groups, CU);
                const auto n_works_g    = works_per_CU * CU;
                const auto macs_g       = static_cast<double>(n_works_g) * grid_g;
                return 1. - dc_macs / macs_g;
            };

        if(G == 1)
        {
            n_groups         = n_CU;
            n_works_per_CU   = Ceil(n_works, n_CU);
            granularity_loss = compute_granularity_loss(n_groups);
            return;
        }

        n_groups         = 1;
        granularity_loss = compute_granularity_loss(n_groups);
        for(auto i = n_groups + 1; i < n_CU; ++i)
        {
            auto loss = compute_granularity_loss(i);
            if(loss < granularity_loss)
            {
                n_groups         = i;
                granularity_loss = loss;
            }
        }

        n_works_per_CU = Ceil(n_works, n_groups) * Ceil(G * n_groups, n_CU);
    }

    [[maybe_unused]] size_t GetNGroups() const noexcept { return n_groups; }

    double ComputeWti() const noexcept
    {
        if(out_of_model_scope)
            return -1.0; // Shader model produces unreliable results.

        // perf tables for the following kernel binaries:
        // {stride1.f32, ostride2.f32, dstride2.f32} : {stride1.f16, ostride2.f16, dstride2.f16}
        // should be updated in case of any changes in kernels configurations or implementaion

        // F(2,3)
        constexpr auto const_2_3 =
            std::array<std::array<size_t, 3>, 2>{{{23652, 18349, 35391}, {28561, 58446, 40618}}};
        constexpr auto fe_2_3 =
            std::array<std::array<size_t, 3>, 2>{{{648, 690, 664}, {872, 737, 800}}};
        constexpr auto ph_2_3 =
            std::array<std::array<size_t, 3>, 2>{{{308, 326, 316}, {327, 337, 335}}};
        constexpr auto be_2_3 =
            std::array<std::array<size_t, 3>, 2>{{{2039, 2409, 1873}, {1852, 2533, 1593}}};

        // F(3,2)
        constexpr auto const_3_2 =
            std::array<std::array<size_t, 3>, 2>{{{37329, 38554, 39701}, {41417, 49740, 57820}}};
        constexpr auto fe_3_2 =
            std::array<std::array<size_t, 3>, 2>{{{950, 616, 550}, {799, 779, 573}}};
        constexpr auto ph_3_2 =
            std::array<std::array<size_t, 3>, 2>{{{295, 315, 308}, {309, 329, 325}}};
        constexpr auto be_3_2 =
            std::array<std::array<size_t, 3>, 2>{{{6295, 9900, 7201}, {4941, 4927, 5111}}};

        const size_t fp_idx     = is_fp16 ? 1 : 0;
        const size_t stride_idx = ostride2 ? 1 : dstride2 ? 2 : 0 /*stride1*/;

        const auto const_cost = (is_2x3 ? const_2_3 : const_3_2)[fp_idx][stride_idx];
        const auto fe_cost    = (is_2x3 ? fe_2_3 : fe_3_2)[fp_idx][stride_idx];
        const auto ph_cost    = (is_2x3 ? ph_2_3 : ph_3_2)[fp_idx][stride_idx];
        const auto be_cost    = (is_2x3 ? be_2_3 : be_3_2)[fp_idx][stride_idx];

        const auto fe_calls = n_works_per_CU * S_loops * R_loops;
        const auto be_calls = n_works_per_CU * (dstride2 ? 2 : 1);
        const auto phases   = fe_calls * C_loops * (!single_mode && !dstride2 ? 2 : 1);

        const auto cycles_predicted = static_cast<double>(const_cost + fe_cost * fe_calls +
                                                          ph_cost * phases + be_cost * be_calls) /
                                      1e6;

        if(cycles_predicted <= 0.1)
            return -1.0; // Unreliable, too small work to do for the shader.

        /// \todo Adjust for different architectures, probably it should be an external source for
        /// this metric
        const size_t macs_per_cu_per_clock = static_cast<size_t>(64) * (is_fp16 ? 2 : 1);

        const auto ideal_direct_cycles =
            direct_convolution_macs / static_cast<double>(n_CU * macs_per_cu_per_clock);
        const auto WTI_predicted = ideal_direct_cycles / cycles_predicted;

        return WTI_predicted;
    }

    [[maybe_unused]] double GetGranularityLoss() const { return granularity_loss; }
};

} // namespace

template <int Winodata, int Winofilter>
static float GetWtiBase(const ExecutionContext& ctx, const ProblemDescription& problem)
{
    constexpr auto WTI_UNKNOWN = -2.0;
    const auto rv              = ShaderModel(ctx, problem, Winodata, Winofilter).ComputeWti();
    return rv < 0 ? WTI_UNKNOWN : rv;
}

template <int Winodata, int Winofilter>
static bool IsApplicableBase(const ExecutionContext& ctx, const ProblemDescription& problem)
{
    if(!problem.Is2d())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16()))
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!ctx.rmv.IsV3())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const auto name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9") || StartsWith(name, "gfx10") || StartsWith(name, "gfx11")))
        return false;
    if(problem.IsFp16() &&
       !(StartsWith(name, "gfx906") || StartsWith(name, "gfx908") || StartsWith(name, "gfx90a") ||
         StartsWith(name, "gfx94") || StartsWith(name, "gfx1011") || StartsWith(name, "gfx1012") ||
         StartsWith(name, "gfx103") || StartsWith(name, "gfx11")))
        return false;

    if(name == "gfx90a" && problem.IsGfx90aFp16altRequired())
        return false;

    // clang-format off
    if (!((problem.GetKernelStrideW() == 1 || problem.GetKernelStrideW() == 2)
        && problem.GetKernelStrideW() == problem.GetKernelStrideH()
        && problem.GetDilationW() == 1
        && problem.GetDilationH() == 1
        && problem.GetBias() == 0
        && problem.GetInLayout() == "NCHW"))
        return false;
        // clang-format on

#if WORKAROUND_ISSUE_2492_GRANULARITY_LOSS
    if(!env::disabled(MIOPEN_DEBUG_WORKAROUND_ISSUE_2492) && !miopen::debug::IsWarmupOngoing)
    {
        constexpr double max_perf_drop_due_to_granularity = 200; // Times.
        const auto gl = ShaderModel(ctx, problem, Winodata, Winofilter).GetGranularityLoss();
        if(gl > (1.0 - 1.0 / max_perf_drop_due_to_granularity))
        {
            MIOPEN_LOG_I("granularity_loss =" << gl);
            return false;
        }
    }
#endif

#if WORKAROUND_ISSUE_2492_TINY_TENSOR
    if(!env::disabled(MIOPEN_DEBUG_WORKAROUND_ISSUE_2492) && !miopen::debug::IsWarmupOngoing)
    {
        // Group count is not taken into account intentionally.
        if(problem.GetInHeight() <= 6     //
           && problem.GetInWidth() <= 6   //
           && problem.GetBatchSize() <= 4 //
           && problem.GetInChannels() <= 4)
            return false;
    }
#endif

    const auto n_inputs_per_group  = problem.GetInChannels() / problem.GetGroupCount(),
               n_outputs_per_group = problem.GetOutChannels() / problem.GetGroupCount();

    if(problem.IsDirectionBackwardWrW())
    {
        if(problem.GetKernelStrideW() == 2)
            return false;
        return IsShaderConstraintsMet<Winodata, Winofilter>(problem,
                                                            problem.GetInHeight(),
                                                            problem.GetInWidth(),
                                                            problem.GetBatchSize(), // N
                                                            n_inputs_per_group,     // K
                                                            problem.GetOutHeight(),
                                                            problem.GetOutWidth(),
                                                            problem.GetWeightsHeight(),
                                                            problem.GetWeightsWidth(),
                                                            n_outputs_per_group, // C
                                                            name);
    }
    else
    {
        return IsShaderConstraintsMet<Winodata, Winofilter>(problem,
                                                            problem.GetWeightsHeight(), // RxS
                                                            problem.GetWeightsWidth(),
                                                            n_inputs_per_group,    // C
                                                            n_outputs_per_group,   // K
                                                            problem.GetInHeight(), // HxW
                                                            problem.GetInWidth(),
                                                            problem.GetOutHeight(), // OHxOW
                                                            problem.GetOutWidth(),
                                                            problem.GetBatchSize(), // N
                                                            name);
    }
}

template <int Winodata, int Winofilter>
bool ConvBinWinoRxS<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                        const ProblemDescription& problem) const
{
    if(IS2X3)
    {
        if(env::disabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3))
            return false;
#if !WORKAROUND_ISSUE_1681
        if(problem.GetGroupCount() == 1 && !problem.IsDirectionBackwardWrW())
            return false;
#endif
    }
    if(IS3X2)
    {
        if(env::disabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2))
            return false;
    }
    return IsApplicableBase<Winodata, Winofilter>(ctx, problem);
}

template <int Winodata, int Winofilter>
static inline boost::optional<PerformanceConfigConvBinWinogradRxS>
GetPerfConfFromEnv(const ExecutionContext& ctx)
{
    PerformanceConfigConvBinWinogradRxS fromEnv;
    std::string s;
    const char* env_name;

    if(IS2X3)
    {
        s        = env::value(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS);
        env_name = "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS";
    }
    else if(IS3X2)
    {
        s        = env::value(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS);
        env_name = "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_PERF_VALS";
    }

    if(s.empty())
        return {};

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
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvBinWinogradRxS& config) const
{
    const auto n_groups = config.n_groups;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool IsWarned = false;
    if(!IsWarned)
    {
        if(ctx.GetStream().GetMaxHardwareComputeUnits() > MAX_CU_LIMIT)
        {
            MIOPEN_LOG_WE(SolverDbId()
                          << ": GPU has " << ctx.GetStream().GetMaxHardwareComputeUnits()
                          << "CUs, but this solver supports max " << MAX_CU_LIMIT
                          << "and thus may show sub-optimal performance.");
        }
        IsWarned = true;
    }

    ConvSolution result;

    const PerformanceConfigConvBinWinogradRxS* pcfg = &config;

    const auto fromEnv = GetPerfConfFromEnv<Winodata, Winofilter>(ctx);
    if(fromEnv)
    {
        pcfg = &(*fromEnv);
    }

    const auto name     = ctx.GetStream().GetDeviceName();
    const auto is_gfx9  = StartsWith(name, "gfx9");
    const auto is_gfx10 = StartsWith(name, "gfx10");
    const auto is_v21   = IsWinogradV21Preferred<Winodata, Winofilter>(name, problem);
    size_t wg_size      = is_gfx9 ? 512 : 256;

    KernelInfo kernel;

    kernel.g_wk.push_back(wg_size * pcfg->GetNGroups() * problem.GetGroupCount());
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto force_cache_bypass = (name == "gfx940") || (name == "gfx941");

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
        {"FORCE_CACHE_BYPASS_ON_STORE", force_cache_bypass},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
    kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");

    const std::string kernel_version = is_v21 ? "_v21_1_3" : "_v30_3_1";
    std::string kernel_name          = "miopenSp3AsmConv" + kernel_version;
    std::string kernel_file          = "Conv_Winograd" + kernel_version;
    std::string kernel_postfix;

    if(is_gfx9)
    {
        kernel_name += "_gfx9";
    }
    else if(is_gfx10)
    {
        kernel_name += "_gfx10";
    }
    else // if(is_gfx11)
    {
        kernel_name += "_gfx11";
    }

    if(problem.IsFp32())
    {
        kernel_name += "_fp32";
        kernel_file += "_fp32";
    }
    else // if(problem.IsFp16())
    {
        kernel_name += is_gfx9 ? "_fp16_dot2_edc" : "_fp16_dot2";
        kernel_file += "_fp16_dot2";
    }

    kernel_postfix = IS2X3 ? "_f2x3" : "_f3x2";

    if(problem.GetKernelStrideW() == 1)
    {
        kernel_postfix += "_stride1";
    }
    else if(problem.GetKernelStrideW() == 2 && !problem.IsDirectionBackwardData())
    {
        kernel_postfix += "_stride2";
    }
    else // if(problem.GetDilationH() == 2)
    {
        kernel_postfix += "_dilation2";
    }

    kernel.kernel_name = kernel_name + kernel_postfix;
    kernel.kernel_file = kernel_file + kernel_postfix + ".s";

    result.construction_params.push_back(kernel);

    constexpr int F_REVERSE_R = 1 << 0;
    constexpr int F_REVERSE_S = 1 << 1;
    constexpr int F_FLIP_K_C  = 1 << 2;
    // These are not used yet. Nevertheless let's keep as a shader documentation.
    // constexpr int F_FLIP_DATA_N_C = 1 << 3; // Unsupported in f3x2.
    // constexpr int F_FLIP_OUT_N_K = 1 << 4; // Unsupported in f3x2.
    // constexpr int F_ADDR_INDIRECT  = 1 << 6;
    // constexpr int F_BIAS  = 1 << 7;
    // constexpr int F_LEAKY_RELU  = 1 << 8;
    constexpr int F_NKC_STRIDES   = 1 << 9;
    constexpr int F_GROUP_STRIDES = 1 << 10;
    // constexpr int F_FORCE_FILTER_TRAVERSE_MODE  = 1 << 11;
    // constexpr int F_FILTER_TRAVERSE_DUAL  = 1 << 12;
    // constexpr int F_TENSOR_OFFSETS  = 1 << 13;
    constexpr int F_USE_ACTIVATION_MODE = 1 << 14;
    // constexpr int F_USE_EXTENDED_FLAGS_64  = 1 << 15;
    int flags = F_NKC_STRIDES | F_GROUP_STRIDES | (is_v21 ? 0 : F_USE_ACTIVATION_MODE);

    constexpr uint8_t IDENTITY = 0;
    // constexpr uint8_t LEAKY_RELU = 1;
    // constexpr uint8_t SIGMOID = 2;
    // constexpr uint8_t SCALED_TANH = 3;
    uint8_t identity_activation = IDENTITY;

    int N, C, H, W, K, out_H, out_W, R, S, pad_H, pad_W;
    MemLayout_t d_layout, o_layout, f_layout;

    const bool is_forward = problem.IsDirectionForward();
    const bool is_backWrW = problem.IsDirectionBackwardWrW();
    const int group_cnt   = problem.GetGroupCount();

    if(!is_backWrW)
    {
        int unused_n_groups = 0;
        GetCompiledInParameters(ctx,
                                problem,
                                &N,
                                &C,
                                &H,
                                &W,
                                &K,
                                &unused_n_groups,
                                &out_H,
                                &out_W,
                                &R,
                                &S,
                                &pad_H,
                                &pad_W);
        C /= group_cnt;
        K /= group_cnt;

        d_layout = GetGroupConvLayout(GetMemLayout_t(problem.GetInLayout()), true);
        o_layout = GetGroupConvLayout(GetMemLayout_t(problem.GetOutLayout()), true);
        f_layout = GetGroupConvLayout(
            is_forward ? MemLayout_t::NCHW : GetSwappedNCLayout(MemLayout_t::NCHW), false);

        flags |= is_forward ? 0 : F_REVERSE_R | F_REVERSE_S | F_FLIP_K_C;
    }
    else
    {
        int unused_n_groups = 0;
        GetCompiledInParameters(ctx,
                                problem,
                                &C,
                                &K,
                                &R,
                                &S,
                                &N,
                                &unused_n_groups,
                                &H,
                                &W,
                                &out_H,
                                &out_W,
                                &pad_H,
                                &pad_W);
        N /= group_cnt;
        K /= group_cnt;
        pad_H = problem.GetConv().GetConvPads()[0];
        pad_W = problem.GetConv().GetConvPads()[1];

        d_layout =
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.GetInLayout())), true);
        o_layout =
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(problem.GetOutLayout())), false);
        f_layout = GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true);
    }

    // cppcheck-suppress unreadVariable
    BuffInfo d_buf(d_layout, N, C, H, W, group_cnt, GetTypeSize(problem.GetInDataType()));
    // cppcheck-suppress unreadVariable
    BuffInfo o_buf(o_layout, N, K, out_H, out_W, group_cnt, GetTypeSize(problem.GetOutDataType()));
    // cppcheck-suppress unreadVariable
    BuffInfo f_buf(f_layout, K, C, R, S, group_cnt, GetTypeSize(problem.GetWeightsDataType()));
    const auto d_strides = is_v21 ? d_buf.byte_stride : d_buf.stride;
    const auto f_strides = is_v21 ? f_buf.byte_stride : f_buf.stride;
    const auto o_strides = is_v21 ? o_buf.byte_stride : o_buf.stride;

    result.invoker_factory = [=](std::vector<Kernel> kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            int reserved             = 0;
            uint64_t reserved_offset = 0;
            int* reserved_ptr        = nullptr;

            const auto k = handle.Run(kernels[0]);
            const auto data_tensors =
                !is_backWrW ? primitive_params.CastTo<miopen::conv::DataInvokeParams>().tensors.in
                            : primitive_params.CastTo<miopen::conv::WrWInvokeParams>().tensors.x;
            const auto filter_tensors =
                !is_backWrW ? primitive_params.CastTo<miopen::conv::DataInvokeParams>().tensors.w
                            : primitive_params.CastTo<miopen::conv::WrWInvokeParams>().tensors.dy;
            const auto out_tensors =
                !is_backWrW ? primitive_params.CastTo<miopen::conv::DataInvokeParams>().tensors.out
                            : primitive_params.CastTo<miopen::conv::WrWInvokeParams>().tensors.dw;

            // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                    << " d_N_stride=" << d_strides.nk  << " d_C_stride=" << d_strides.c
                    << " d_H_stride=" << d_strides.h   << " d_W_stride=" << d_strides.w
                    << " f_K_stride=" << f_strides.nk  << " f_C_stride=" << f_strides.c
                    << " f_R_stride=" << f_strides.h   << " f_S_stride=" << f_strides.w
                    << " o_N_stride=" << o_strides.nk  << " o_K_stride=" << o_strides.c
                    << " o_H_stride=" << o_strides.h   << " o_W_stride=" << o_strides.w
                    << " d_G_stride=" << d_strides.g   << " f_G_stride=" << f_strides.g
                    << " o_G_stride=" << o_strides.g);
            // clang-format on
            if(is_v21)
            {
                k(N,
                  C,
                  H,
                  W,
                  K,
                  n_groups,
                  flags,
                  reserved,
                  data_tensors,
                  filter_tensors,
                  out_tensors,
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
                  d_strides.nk,
                  d_strides.c,
                  d_strides.h,
                  d_strides.w,
                  f_strides.nk,
                  f_strides.c,
                  f_strides.h,
                  f_strides.w,
                  o_strides.nk,
                  o_strides.c,
                  o_strides.h,
                  o_strides.w,
                  group_cnt,
                  d_strides.g,
                  f_strides.g,
                  o_strides.g);
            }
            else
            {
                k(N,
                  C,
                  H,
                  W,
                  K,
                  n_groups,
                  flags,
                  reserved,
                  data_tensors,
                  filter_tensors,
                  out_tensors,
                  reserved_ptr, // Unused return_addr.
                  R,
                  S,
                  pad_H,
                  pad_W,
                  out_H,
                  out_W,
                  reserved_ptr,    // Unused bias_addr.
                  reserved,        // Unused alpha.
                  reserved,        // Unused beta.
                  reserved_offset, // Unused d_offset.
                  reserved_offset, // Unused f_offset.
                  reserved_offset, // Unused o_offset.
                  reserved_offset, // Unused b_offset.
                  d_strides.nk,
                  d_strides.c,
                  d_strides.h,
                  d_strides.w,
                  f_strides.nk,
                  f_strides.c,
                  f_strides.h,
                  f_strides.w,
                  o_strides.nk,
                  o_strides.c,
                  o_strides.h,
                  o_strides.w,
                  group_cnt,
                  d_strides.g,
                  f_strides.g,
                  o_strides.g,
                  identity_activation);
            }
        };
    };

    return result;
}

bool ConvBinWinogradRxSf2x3g1::IsApplicable(const ExecutionContext& ctx,
                                            const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1))
        return false;
    return IsApplicableBase<2, 3>(ctx, problem) && problem.GetGroupCount() == 1;
}

float ConvBinWinogradRxSf2x3g1::GetWti(const ExecutionContext& ctx,
                                       const ProblemDescription& problem) const
{
    return GetWtiBase<2, 3>(ctx, problem);
}

ConvSolution ConvBinWinogradRxSf2x3g1::GetSolution(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    const auto tunable = ConvBinWinoRxS<2, 3>{};
    return tunable.GetSolution(ctx, problem, tunable.GetDefaultPerformanceConfig(ctx, problem));
}

template struct ConvBinWinoRxS<2, 3>;
template struct ConvBinWinoRxS<3, 2>;

} // namespace conv
} // namespace solver
} // namespace miopen
