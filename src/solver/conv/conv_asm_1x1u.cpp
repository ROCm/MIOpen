/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <sstream>
#include <limits>
#include <cassert>

#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/invokers/gcn_asm_1x1u.hpp>
#include <miopen/conv/invokers/gcn_asm_1x1u_ss.hpp>
#include <miopen/conv/invokers/gcn_asm_1x1u_us.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

namespace {

bool UseSubsample(const ProblemDescription& problem)
{
    return (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1) &&
           problem.IsDirectionForward();
}

bool UseUpsample(const ProblemDescription& problem)
{
    return (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1) &&
           problem.IsDirectionBackwardData();
}

/// After 2x subsampling kernel, image size on asm kernel input becomes 4x (2*2) smaller.
/// As padding = 0, we can simply re-use output image size (no computations required).
/// \note For backward convolutions input image size is held in
/// out_height/out_width and vice versa.
std::size_t AsmImgHeight(const ProblemDescription& problem)
{
    return UseSubsample(problem) ? problem.GetOutHeight() : problem.GetInHeight();
}

std::size_t AsmImgWidth(const ProblemDescription& problem)
{
    return UseSubsample(problem) ? problem.GetOutWidth() : problem.GetInWidth();
}

/// \todo move to separate header and use in other solvers.
template <int L, int H>
bool IsTwoPower(const int v)
{
    static_assert(L <= H, "L <= H");
    if(((v - 1) & v) != 0)
        return false;
    return L <= v && v <= H;
}

template <int L, int H>
bool NextTwoPower(int& v)
{
    static_assert((((L - 1) & L) == 0), "L is not power of 2");
    static_assert((((H - 1) & H) == 0), "H is not power of 2");
    assert((IsTwoPower<L, H>(v)));
    if(v == H)
    {
        v = L;
        return true;
    }
    v *= 2;
    return false;
}

template <int L, int H>
bool IsLinear(const int v)
{
    static_assert(L <= H, "L <= H");
    return L <= v && v <= H;
}

template <int L, int H>
bool NextLinear(int& v)
{
    assert((IsLinear<L, H>(v)));
    if(H == v)
    {
        v = L;
        return true;
    }
    ++v;
    return false;
}

// This range is like regular range [0,4,8...32], but 1 is used instead of 0.
bool Is_1_4_8_12_to_32(const int& v) { return v == 1 || (v % 4 == 0 && IsLinear<1, 8>(v / 4)); }

bool Next_1_4_8_12_to_32(int& v)
{
    assert(Is_1_4_8_12_to_32(v));
    int tmp        = v / 4;
    const bool ret = NextLinear<0, 8>(tmp);
    v              = ret ? 1 : tmp * 4;
    return ret;
}

#ifndef NDEBUG
bool Is_1_4(const int& v) { return v == 1 || v == 4; }
#endif

bool Next_1_4(int& v)
{
    assert(Is_1_4(v));
    if(v == 4)
    {
        v = 1;
        return true;
    }
    v = 4;
    return false;
}

} // namespace

bool PerformanceConfigConvAsm1x1U::SetNextValue(const ProblemDescription&)
{
    // Increment with wrap-around:
    do
    {
        if(!NextLinear<1, 4>(read_size))
            break;
        if(!env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED))
        {
            /// Narrow search space in optimized mode.
            if(use_spare_set ? !Next_1_4(k_mult) : !NextTwoPower<8, 32>(k_mult))
                break;
            if(!NextLinear<1, 8>(chunks_per_wave))
                break;
            if(use_spare_set ? !Next_1_4(chunk_size) : !NextTwoPower<16, 64>(chunk_size))
                break;
            if(!NextLinear<1, 4>(n_mult))
                break;
            if(!NextTwoPower<1, 4>(c_mult))
                break;
            if(!NextTwoPower<1, 4>(waves_c_in_group))
                break;
            if(!NextTwoPower<1, 8>(waves_k_in_group))
                break;
        }
        else
        {
            if(!Next_1_4_8_12_to_32(k_mult))
                break;
            if(!NextLinear<1, 16>(chunks_per_wave))
                break;
            if(!NextTwoPower<1, 64>(chunk_size))
                break;
            if(!NextLinear<1, 8>(n_mult))
                break;
            if(!NextTwoPower<1, 32>(c_mult))
                break;
            if(!NextLinear<1, 8>(waves_c_in_group))
                break;
            if(!NextTwoPower<1, 8>(waves_k_in_group))
                break;
        }
        // All the fields of performance config have wrapped around.
        return false;
    } while(false);
    return true;
}

PerformanceConfigConvAsm1x1U::PerformanceConfigConvAsm1x1U(bool spare)
    : PerformanceConfigConvAsm1x1U(1, 1, 1, 1, 1, 1, 1, 1, spare)
{
    if(!env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED))
    {
        k_mult     = spare ? 1 : 8;
        chunk_size = spare ? 1 : 16;
    }
}

PerformanceConfigConvAsm1x1U::PerformanceConfigConvAsm1x1U(int read_size_,
                                                           int k_mult_,
                                                           int chunks_per_wave_,
                                                           int chunk_size_,
                                                           int n_mult_,
                                                           int c_mult_,
                                                           int waves_c_in_group_,
                                                           int waves_k_in_group_,
                                                           bool use_spare_set_)
    : read_size(read_size_),
      k_mult(k_mult_),
      chunks_per_wave(chunks_per_wave_),
      chunk_size(chunk_size_),
      n_mult(n_mult_),
      c_mult(c_mult_),
      waves_c_in_group(waves_c_in_group_),
      waves_k_in_group(waves_k_in_group_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceConfigConvAsm1x1U::operator==(const PerformanceConfigConvAsm1x1U& other) const
{
    // clang-format off
    return read_size == other.read_size
        && k_mult == other.k_mult
        && chunks_per_wave == other.chunks_per_wave
        && chunk_size == other.chunk_size
        && n_mult == other.n_mult
        && c_mult == other.c_mult
        && waves_c_in_group == other.waves_c_in_group
        && waves_k_in_group == other.waves_k_in_group
        && use_spare_set == other.use_spare_set; // clang-format on
}

bool PerformanceConfigConvAsm1x1U::IsValidValueImpl(const int sequence_length) const
{
    if(sequence_length > 0)
    {
        if(!IsLinear<1, 4>(read_size))
            return false;
    }
    if(sequence_length > 1)
    {
        if(!Is_1_4_8_12_to_32(k_mult))
            return false;
    }
    if(sequence_length > 2)
    {
        if(!IsLinear<1, 16>(chunks_per_wave))
            return false;
    }
    if(sequence_length > 3)
    {
        if(!IsTwoPower<1, 64>(chunk_size))
            return false;
    }
    if(sequence_length > 4)
    {
        if(!IsLinear<1, 8>(n_mult))
            return false;
    }
    if(sequence_length > 5)
    {
        if(!IsTwoPower<1, 32>(c_mult))
            return false;
    }
    if(sequence_length > 6)
    {
        if(!IsLinear<1, 8>(waves_c_in_group))
            return false;
    }
    if(sequence_length > 7)
    {
        if(!IsTwoPower<1, 8>(waves_k_in_group))
            return false;
    }
    return true;
}

bool PerformanceConfigConvAsm1x1U::IsValidImpl(const ProblemDescription& problem,
                                               const int sequence_length) const
{
    const auto elements_in_dword = 4 / static_cast<int>(GetTypeSize(problem.GetInDataType()));
    if(elements_in_dword == 0) // For clang-tidy (DIV/0)
        MIOPEN_THROW(miopenStatusInternalError);
    const unsigned img_hw = problem.GetOutHeight() * problem.GetOutWidth();
    if(!IsValidValueImpl(sequence_length))
        return false;
    if(sequence_length > 1)
    {
        if((k_mult % elements_in_dword) != 0)
            return false;
        if(problem.IsDirectionBackwardData() && !(problem.GetOutChannels() % k_mult == 0))
            return false;
    }
    if(sequence_length > 2)
    {
        if(!(read_size * elements_in_dword <= chunks_per_wave))
            return false;
        if(chunks_per_wave % elements_in_dword != 0)
            return false;
    }
    if(sequence_length > 3)
    {
        const int total_chunks = (img_hw + chunk_size - 1) / chunk_size;
        if(!(chunks_per_wave <= total_chunks))
            return false;
    }
    if(sequence_length > 4)
    {
        const int total_n_blocks = (problem.GetBatchSize() + GetNPerGpr() - 1) / GetNPerGpr();
        if(!(n_mult <= total_n_blocks))
            return false;
    }
    const auto in_gprs =
        (chunks_per_wave * n_mult * c_mult + elements_in_dword - 1) / elements_in_dword;
    const auto acc_gprs = chunks_per_wave * n_mult * k_mult;
    // TODO last vgpr only for old card.
    // ADD if(option.machine_version_major == 9)
    // vgprs  = 4 + 2 * in_gprs + acc_gprs + (img_hw % elements_in_dword != 0 ? 1: 0);
    // else
    const auto vgprs = 4 + 2 * in_gprs + acc_gprs + (img_hw % elements_in_dword != 0 ? 1 : 0) + 1;
    if(sequence_length > 5)
    {
        if((c_mult % elements_in_dword) != 0)
            return false;
        if(!(vgprs < 256))
            return false;
        const auto sgprs = 25 + 2 * k_mult * c_mult;
        if(!(sgprs < 102)) /// \todo This is valid for Gfx8 and Gfx9. Check for newer parts.
            return false;
    }
    if(sequence_length > 6)
    {
        if(!(waves_c_in_group <= problem.GetInChannels()))
            return false;
        const int c_per_wave = (problem.GetInChannels() + waves_c_in_group - 1) / waves_c_in_group;
        const int c_per_last_wave =
            problem.GetInChannels() - static_cast<std::size_t>(c_per_wave * (waves_c_in_group - 1));
        if(c_per_wave % c_mult != 0 || c_per_last_wave % c_mult != 0)
            return false;
    }
    if(sequence_length > 7)
    {
        if(!(static_cast<std::size_t>(k_mult) * waves_k_in_group <= problem.GetOutChannels()))
            return false;
        if(!(waves_c_in_group * waves_k_in_group <= 16))
            return false;
        const auto max_waves_per_CU = (256 / vgprs) * 4;
        if(!(max_waves_per_CU >= waves_c_in_group * waves_k_in_group))
            return false;
    }
    return true;
}

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
bool PerformanceConfigConvAsm1x1U::ModelApplyToken(int index,
                                                   std::string value,
                                                   const ProblemDescription& problem)
{
    int val = stoi(value);
    switch(index)
    {
    case 0: read_size = val; break;
    case 1: k_mult = val; break;
    case 2: chunks_per_wave = val; break;
    case 3: chunk_size = val; break;
    case 4: n_mult = val; break;
    case 5: c_mult = val; break;
    case 6: waves_c_in_group = val; break;
    case 7: waves_k_in_group = val; break;
    default: return false;
    }
    // this function may leave PerformanceConfigConvAsm1x1U in a partially valid or invalid state
    return this->IsPartiallyValid(problem, index + 1);
}

static std::vector<float> TransformFeatures(const ProblemDescription& problem, std::size_t n)
{
    assert(n == 8); // n = 6 (numerical conv params) * 1 + 1 (nominal conv params) * 2(amount of
                    // values nominal param can take).
    std::vector<float> features(n * n, 0.0f);
    features[0]                   = problem.IsFp32() ? 2.0 : 1.0;
    int offset                    = (problem.IsDirectionForward() ? 0 : 1) + 1;
    features[(offset)*n + offset] = 1.0;
    features[3 * n + 3] =
        float(problem.IsDirectionForward() ? problem.GetInChannels() : problem.GetOutChannels());
    features[4 * n + 4] =
        float(problem.IsDirectionForward() ? problem.GetOutChannels() : problem.GetInChannels());
    features[5 * n + 5] = float(problem.GetInHeight());
    features[6 * n + 6] = float(problem.GetInWidth());
    features[7 * n + 7] = float(problem.GetBatchSize());
    return features;
}

bool PerformanceConfigConvAsm1x1U::RunParameterPredictionModel(const ExecutionContext& ctx,
                                                               const ProblemDescription& problem)
{
    static const std::size_t n      = 8;
    static const std::string& arch  = ctx.GetStream().GetDeviceName();
    static const std::string solver = "ConvAsm1x1U";
    std::vector<float> features     = TransformFeatures(problem, n);
    if(ai::tuning::ModelSetParams(
           arch, solver, problem.GetDirection(), features, true, [&](int idx, std::string value) {
               return this->ModelApplyToken(idx, value, problem);
           }))
    {
        MIOPEN_LOG_I("Params set by AI: " << ToString());
        return true;
    }
    return false;
}
#endif

void PerformanceConfigConvAsm1x1U::StaticHeuristic(const ProblemDescription& problem)
{
    const auto elements_in_dword = 4 / GetTypeSize(problem.GetInDataType());
    if(elements_in_dword == 0) // For clang-tidy (DIV/0)
        MIOPEN_THROW(miopenStatusInternalError);
    read_size        = 4;
    k_mult           = 16;
    chunks_per_wave  = static_cast<int>(read_size * elements_in_dword);
    chunk_size       = 16;
    n_mult           = 2;
    c_mult           = elements_in_dword;
    waves_c_in_group = 1;
    waves_k_in_group = 1;

    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        read_size  = 1;
        k_mult     = 4;
        chunk_size = 1;
        n_mult     = 1;
    }
    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        k_mult = (elements_in_dword == 1) ? 1 : 4;
        c_mult = 2;
    }
    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        chunks_per_wave = 2;
        c_mult          = elements_in_dword;
    }
    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        chunks_per_wave = elements_in_dword;
    }
    if(!IsValid(problem))
    {
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
    }
}

bool PerformanceConfigConvAsm1x1U::IsModelApplicable(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR))
        return false;
    if(ctx.GetStream().GetDeviceName() != "gfx908")
        return false;
    if(problem.GetKernelStrideH() != 1)
        return false;
    return true;
}

void PerformanceConfigConvAsm1x1U::HeuristicInit([[maybe_unused]] const ExecutionContext& ctx,
                                                 const ProblemDescription& problem)
{
    if(problem.GetInDataType() == miopenDouble)
        MIOPEN_THROW("Double data type is not supported by ConvAsm1x1U");
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(IsModelApplicable(ctx, problem))
    {

        if(RunParameterPredictionModel(ctx, problem))
            return;
    }
#endif
    StaticHeuristic(problem);
    MIOPEN_LOG_I(ToString());
}

PerformanceConfigConvAsm1x1U
ConvAsm1x1U::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                         const ProblemDescription& problem) const
{
    PerformanceConfigConvAsm1x1U pp;
    pp.HeuristicInit(ctx, problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsm1x1U::IsValidPerformanceConfig(const ExecutionContext&,
                                           const ProblemDescription& problem,
                                           const PerformanceConfigConvAsm1x1U& config) const
{
    return config.IsValidValue() && config.IsValid(problem);
}

bool ConvAsm1x1U::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!(problem.IsDirectionForward() || problem.IsDirectionBackwardData()))
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!ctx.rmv.IsV2orV3())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16()))
        return false;

    if(problem.IsTensorsCasted())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = ctx.GetStream().GetDeviceName();
    if(name.find("gfx9") == std::string::npos)
        return false;
    if(!problem.IsLayoutDefault())
        return false;

    if(problem.IsTensorsCasted() || problem.IsFp8() || problem.IsBfp8())
        return false;

    if(name == "gfx90a" && problem.IsGfx90aFp16altRequired())
        return false;

    const auto elements_in_dword = 4 / GetTypeSize(problem.GetInDataType());
    if(elements_in_dword == 0) // For clang-tidy (false positive DIV/0)
        MIOPEN_THROW(miopenStatusInternalError);
    // clang-format off
    const int img_hw = problem.GetOutHeight() * problem.GetOutWidth();
    bool ok = (problem.GetPadW() == 0       // -q  pad_w
        && problem.GetPadH() == 0           // -p  pad_h
        && problem.GetKernelStrideW() <= 2  // -u  stride_w
        && problem.GetKernelStrideW() == problem.GetKernelStrideH()
        && problem.GetWeightsWidth() == 1   // -x  S wei_w
        && problem.GetWeightsHeight() == 1  // -y  R wei_h
        && problem.GetDilationW() == 1
        && problem.GetDilationH() == 1
        && problem.GetBias() == 0
        && problem.GetInChannels() % elements_in_dword == 0
        && problem.GetOutChannels() % elements_in_dword == 0
        && problem.GetInLayout() == "NCHW"
        && problem.GetGroupCount() == 1
        && img_hw >= elements_in_dword
        && (elements_in_dword == 1 || problem.GetOutChannels() >= 4));
    if(problem.IsDirectionBackwardData() && elements_in_dword != 1)
        ok = ok && (problem.GetOutChannels() % 4 == 0);
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    /// \todo Ilya: The checks below look adequate but needs to be double-checked.
    {
        const uint64_t input_line_size = 4 * problem.GetInWidth();
        const uint64_t input_feature_map_size = input_line_size * problem.GetInHeight();
        const uint64_t input_stack_size = input_feature_map_size * problem.GetInChannels();
        if (! (input_stack_size < (1U << 24)))
            return false;
    }
    {
        const uint64_t output_line_size = 4 * problem.GetOutWidth();
        const uint64_t output_feature_map_size = output_line_size * problem.GetOutHeight();
        const uint64_t output_stack_size = output_feature_map_size * problem.GetOutChannels();
        if (! (output_stack_size < (1U << 24)))
            return false;
    }
    // Check limits:
    const auto h_w = AsmImgHeight(problem) * AsmImgWidth(problem);
    const auto r_s     = problem.GetWeightsHeight() * problem.GetWeightsWidth();
    const auto c_h_w   = problem.GetInChannels() * h_w;  // C*H*W
    const auto k_h_w   = problem.GetOutChannels() * h_w; // K*H*W
    const auto n_c_h_w = problem.GetBatchSize() * c_h_w; // N*C*H*W
    const auto n_k_h_w = problem.GetBatchSize() * k_h_w; // N*K*H*W
    const auto c_k_r_s = problem.GetInChannels() * problem.GetOutChannels() * r_s; // C*K*R*S
    ok = problem.GetBatchSize() < std::pow(2, 16)       // -n   N batch_size
         && problem.GetInChannels() < std::pow(2, 16)   // -c   C input_channels
         && problem.GetOutChannels() < std::pow(2, 16)  // -k   K output_channels
         && c_h_w < std::pow(2, 24)
         && k_h_w < std::pow(2, 24)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29); // clang-format on
    return ok;
}

size_t ConvAsm1x1U::GetWorkspaceSize(const ExecutionContext&,
                                     const ProblemDescription& problem) const
{
    if(UseSubsample(problem) || UseUpsample(problem))
    {
        auto in_batch_stride =
            AsmImgWidth(problem) * AsmImgHeight(problem) *
            (UseSubsample(problem) ? problem.GetInChannels() : problem.GetOutChannels());
        auto data_len = GetTypeSize(problem.GetOutDataType());
        return in_batch_stride * problem.GetBatchSize() * data_len;
    }
    return 0;
}
static int divide_round_plus_inf(const int x, const int y)
{
    assert(x >= 0 && y > 0);
    if(x % y != 0)
        return x / y + 1;
    return x / y;
}

ConvSolution ConvAsm1x1U::GetSolution(const ExecutionContext& ctx,
                                      const ProblemDescription& problem,
                                      const PerformanceConfigConvAsm1x1U& config) const
{
    ConvSolution result;

    std::ostringstream options;

    KernelInfo ss_us_kernel;
    int data_len = GetTypeSize(problem.GetOutDataType());
    if(UseSubsample(problem) || UseUpsample(problem))
    {
        // subsampled input, in_height equals to image size after downsampling
        auto in_batch_stride =
            AsmImgWidth(problem) * AsmImgHeight(problem) *
            (UseSubsample(problem) ? problem.GetInChannels() : problem.GetOutChannels());
        unsigned write_unit = (AsmImgWidth(problem) % 4 == 0)   ? 4
                              : (AsmImgWidth(problem) % 3 == 0) ? 3
                              : (AsmImgWidth(problem) % 2 == 0) ? 2
                                                                : 1;

        int n_grp0_size0 = 256;

        // clang-format off
        const auto subsample_kernel_compilation_options =
            std::string(" -DDATA_TYPE=") +
            (problem.GetInDataType() == miopenHalf ? "ushort" : "float") +
            std::string(" -DMLO_GRP0_SZ0=") + std::to_string(n_grp0_size0) +
            std::string(" -DMLO_GRP0_SZ1=1 ") + std::string(" -DMLO_GRP0_SZ2=1 ") +
            std::string(" -DMLO_FILTER0_STRIDE0=") + std::to_string(problem.GetKernelStrideW()) +
            std::string(" -DMLO_FILTER0_STRIDE1=") + std::to_string(problem.GetKernelStrideH()) +
            std::string(" -DMLO_WRITE_UNIT=") + std::to_string(write_unit) +
            std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(problem.GetOutChannelStride()) +
            std::string(" -DMLO_OUT_STRIDE=") + std::to_string(problem.GetOutStrideH()) +
            std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(in_batch_stride) +
            std::string(" -DMLO_IN0_BATCH_STRIDE=") +
            std::to_string(problem.IsDirectionForward() ? problem.GetInBatchStride()
                                                         : problem.GetOutBatchStride()) +
            std::string(" -DMLO_IN0_CHANNEL_STRIDE=") + std::to_string(problem.GetInChannelStride()) +
            std::string(" -DMLO_IN0_STRIDE=") + std::to_string(problem.GetInStrideH()) +
            ctx.general_compile_options;
        // clang-format on

        ss_us_kernel.l_wk.push_back(n_grp0_size0);
        ss_us_kernel.l_wk.push_back(1);
        ss_us_kernel.l_wk.push_back(1);
        // output is number of subsampled input maps
        size_t gbl_wk0 = (in_batch_stride / write_unit);
        size_t gbl_wk1 = problem.GetBatchSize();
        size_t gbl_wk2 = 1;

        ss_us_kernel.g_wk.push_back(gbl_wk0);
        ss_us_kernel.g_wk.push_back(gbl_wk1);
        ss_us_kernel.g_wk.push_back(gbl_wk2);

        ss_us_kernel.kernel_file = "MIOpenUtilKernels3.cl";

        if(UseSubsample(problem))
            ss_us_kernel.kernel_name = "SubSample";
        else
            ss_us_kernel.kernel_name = "UpSample";

        ss_us_kernel.comp_options = subsample_kernel_compilation_options;
    }
    result.workspace_sz = GetWorkspaceSize(ctx, problem);

    GenerateClangDefsym(options, "stride_h", 1);
    GenerateClangDefsym(options, "stride_w", 1);
    GenerateClangDefsym(options, "img_h", AsmImgHeight(problem)); // H
    GenerateClangDefsym(options, "img_w", AsmImgWidth(problem));  // W

    // Note that problem.n_outputs and problem.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "batch_size", problem.GetBatchSize());        // N
    GenerateClangDefsym(options, "input_channels", problem.GetInChannels());   // C
    GenerateClangDefsym(options, "output_channels", problem.GetOutChannels()); // K
    GenerateClangDefsym(options, "wei_h", problem.GetWeightsHeight());         // R
    GenerateClangDefsym(options, "wei_w", problem.GetWeightsWidth());          // S
    GenerateClangDefsym(options, "pad_h", problem.GetPadH());
    GenerateClangDefsym(options, "pad_w", problem.GetPadW());
    GenerateClangDefsym(options, "weights_layout", problem.IsDirectionForward() ? 0 : 1);

    GenerateClangDefsym(options, "vec_c_in", 1);
    GenerateClangDefsym(options, "vec_k_out", 1);
    GenerateClangDefsym(options, "vec_c_filter", 1);

    GenerateClangDefsym(options, "acc_type", 1);
    GenerateClangDefsym(options, "buf_type", (data_len == 2 ? 2 : 1));
    enum class MemLayout : int
    {
        NCHW = 0,
        CNHW = 1,
    };

    struct buff_info
    {
        size_t total_byte_size;
        struct
        {
            int nk, c, h, w;
        } stride{}, byte_stride{}, size{};

        buff_info(MemLayout layout, int nk, int c, int h, int w, int vec_c, int data_len_t)
        {
            int c_hi        = (c + vec_c - 1) / vec_c;
            auto count      = static_cast<size_t>(nk) * c_hi * h * w * vec_c;
            total_byte_size = count * data_len_t;
            size.nk         = nk;
            size.c          = c;
            size.h          = h;
            size.w          = w;

            switch(layout)
            {
            case MemLayout::NCHW:
                stride.w  = 1;
                stride.h  = w;
                stride.c  = w * h;
                stride.nk = w * h * c_hi;
                break;
            case MemLayout::CNHW:
                stride.w  = 1;
                stride.h  = w;
                stride.nk = w * h;
                stride.c  = w * h * nk;
                break;
            }
            stride.nk *= vec_c;
            stride.c *= vec_c;
            stride.h *= vec_c;
            stride.w *= vec_c;
            byte_stride.nk = stride.nk * data_len_t;
            byte_stride.c  = stride.c * data_len_t;
            byte_stride.h  = stride.h * data_len_t;
            byte_stride.w  = stride.w * data_len_t;
        }
    };

    // cppcheck-suppress unreadVariable
    buff_info ibuf(MemLayout::NCHW,
                   problem.GetBatchSize(),
                   problem.GetInChannels(),
                   AsmImgHeight(problem),
                   AsmImgWidth(problem),
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info obuf(MemLayout::NCHW,
                   problem.GetBatchSize(),
                   problem.GetOutChannels(),
                   AsmImgHeight(problem),
                   AsmImgWidth(problem),
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info fbuf(problem.IsDirectionForward() ? MemLayout::NCHW : MemLayout::CNHW,
                   problem.GetOutChannels(),
                   problem.GetInChannels(),
                   1,
                   1,
                   1,
                   data_len);

    GenerateClangDefsym(options, "input_n_stride", ibuf.byte_stride.nk);
    GenerateClangDefsym(options, "input_c_stride", ibuf.byte_stride.c);
    GenerateClangDefsym(options, "input_h_stride", ibuf.byte_stride.h);
    GenerateClangDefsym(options, "input_w_stride", ibuf.byte_stride.w);

    GenerateClangDefsym(options, "output_n_stride", obuf.byte_stride.nk);
    GenerateClangDefsym(options, "output_k_stride", obuf.byte_stride.c);
    GenerateClangDefsym(options, "output_h_stride", obuf.byte_stride.h);
    GenerateClangDefsym(options, "output_w_stride", obuf.byte_stride.w);

    GenerateClangDefsym(options, "filter_k_stride", fbuf.byte_stride.nk);
    GenerateClangDefsym(options, "filter_c_stride", fbuf.byte_stride.c);
    GenerateClangDefsym(options, "filter_h_stride", fbuf.byte_stride.h);
    GenerateClangDefsym(options, "filter_w_stride", fbuf.byte_stride.w);
    GenerateClangDefsym(options, "input_buffer_size", ibuf.total_byte_size);
    GenerateClangDefsym(options, "filter_buffer_size", fbuf.total_byte_size);
    GenerateClangDefsym(options, "output_buffer_size", obuf.total_byte_size);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    const PerformanceConfigConvAsm1x1U* pcfg = &config;

    PerformanceConfigConvAsm1x1U fromEnv;
    {
        const auto s = env::value(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS);
        if(!s.empty()) // else nothing to parse.
        {
            if(!fromEnv.Deserialize(s) || !fromEnv.IsValidValue())
            {
                MIOPEN_LOG_E("MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS: "
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

    GenerateClangDefsym(options, "read_size", pcfg->GetReadSize());
    GenerateClangDefsym(options, "k_mult", pcfg->GetKMult());
    GenerateClangDefsym(options, "chunks_per_wave", pcfg->GetChunksPerWave());
    GenerateClangDefsym(options, "chunk_size", pcfg->GetChunkSize());
    GenerateClangDefsym(options, "n_mult", pcfg->GetNMult());
    GenerateClangDefsym(options, "c_mult", pcfg->GetCMult());
    GenerateClangDefsym(options, "waves_c_in_group", pcfg->GetWavesCInGroup());
    GenerateClangDefsym(options, "waves_k_in_group", pcfg->GetWavesKInGroup());

    KernelInfo main_kernel;
    main_kernel.comp_options = options.str();

    const int waves_in_group = pcfg->GetWavesCInGroup() * pcfg->GetWavesKInGroup();
    main_kernel.l_wk.clear(); // workgroupsize
    main_kernel.l_wk.push_back(64ULL * waves_in_group);
    main_kernel.l_wk.push_back(1);
    main_kernel.l_wk.push_back(1);

    main_kernel.g_wk.clear(); // gridsize
    const int hw_per_wave = pcfg->GetChunksPerWave() * pcfg->GetChunkSize();

    main_kernel.g_wk.push_back(
        main_kernel.l_wk[0] *
        divide_round_plus_inf(AsmImgHeight(problem) * AsmImgWidth(problem), hw_per_wave));

    main_kernel.g_wk.push_back(divide_round_plus_inf(problem.GetOutChannels(),
                                                     pcfg->GetKMult() * pcfg->GetWavesKInGroup()));
    const int n_images_per_wave = pcfg->GetNMult() * pcfg->GetNPerGpr();
    main_kernel.g_wk.push_back(divide_round_plus_inf(problem.GetBatchSize(), n_images_per_wave));

    main_kernel.kernel_file = "conv1x1u.s";
    main_kernel.kernel_name = "miopenGcnAsmConv1x1U";

    if(UseSubsample(problem))
        result.construction_params.push_back(ss_us_kernel);

    result.construction_params.push_back(main_kernel);

    if(UseUpsample(problem))
        result.construction_params.push_back(ss_us_kernel);

    if(UseSubsample(problem))
    {
        int N, C, H, W, K, n_groups, out_H, out_W;
        GetCompiledInParameters(ctx, problem, &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W);
        result.invoker_factory = miopen::conv::MakeGcnAsm1x1USSInvokerFactory(
            N, C, K, n_groups, out_H, out_W, result.workspace_sz);
    }
    else if(UseUpsample(problem))
    {
        int N, C, H, W, K, n_groups;
        GetCompiledInParameters(ctx, problem, &N, &C, &H, &W, &K, &n_groups);
        result.invoker_factory = miopen::conv::MakeGcnAsm1x1UUSInvokerFactory(
            N, C, K, n_groups, H, W, result.workspace_sz);
    }
    else
    {
        int N, C, H, W, K, n_groups;
        GetCompiledInParameters(ctx, problem, &N, &C, &H, &W, &K, &n_groups);
        result.invoker_factory =
            miopen::conv::MakeGcnAsm1x1UInvokerFactory(N, C, H, W, K, n_groups);
    }

    return result;
}

PerformanceConfigConvAsm1x1U ConvAsm1x1U::Search(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem,
                                                 const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

} // namespace conv
} // namespace solver
} // namespace miopen
