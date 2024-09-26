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
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

static inline bool UseSubsample(const ProblemDescription& problem)
{
    return problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1;
}

/// After 2x subsampling kernel, image size on asm kernel input becomes 4x (2*2) smaller.
/// As padding = 0, we can simply re-use output image size (no computations required).
/// \note For backward convolutions input image size is held in
/// out_height/out_width and vice versa.
static inline int AsmImgHeight(const ProblemDescription& problem)
{
    return UseSubsample(problem) ? problem.GetInHeight() : problem.GetOutHeight();
}

static inline int AsmImgWidth(const ProblemDescription& problem)
{
    return UseSubsample(problem) ? problem.GetInWidth() : problem.GetOutWidth();
}

inline static bool Inc_1_2_4_8_16(int& v)
{
    assert(v == 1 || v == 2 || v == 4 || v == 8 || v == 16);
    if(v == 16)
    {
        v = 1;
        return true;
    }
    v = v * 2;
    return false;
}

/// \todo Rework, factor out to separate header and use in other solvers.
/// \todo Clarify functions semantics.
template <int first, int... others>
inline static bool IsFromPackContinue(const int v)
{
    return (v == first) || IsFromPackContinue<others...>(v);
}

template <int first, int... others>
inline static bool IsFromPack(const int v)
{
    return IsFromPackContinue<first, others..., 0, 0>(v);
}

template <>
inline bool IsFromPackContinue<0, 0>(const int)
{
    return false;
}

template <int next, int... others>
inline static bool IncPackNext(int& v)
{
    v = next;
    return true;
}

template <int first, int... others>
inline static bool IncPackContinue(int& v)
{
    return ((v == first) && IncPackNext<others...>(v)) || IncPackContinue<others...>(v);
}

template <>
inline bool IncPackContinue<0, 0>(int&)
{
    return false;
}

template <int first, int... others>
inline static bool IncPack(int& v)
{
    assert((IsFromPack<first, others...>(v)));
    IncPackContinue<first, others..., first, 0, 0>(v);

    return (v == first);
}

inline static bool Is_1_2_4_8_16(const int& v)
{
    return v == 1 || v == 2 || v == 4 || v == 8 || v == 16;
}

inline static bool Inc_1_2_4(int& v)
{
    assert(v == 1 || v == 2 || v == 4);
    if(v == 4)
    {
        v = 1;
        return true;
    }
    v = v * 2;
    return false;
}

inline static bool Is_1_2_4(const int& v) { return v == 1 || v == 2 || v == 4; }

bool PerformanceConfigConvAsmBwdWrW1x1::SetNextValue(const ProblemDescription&)
{
    // Increment with wrap-around:
    // select fast or full method
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_SEARCH_OPTIMIZED))
    {
        do
        {
            if(++read_size <= 4)
                break;
            read_size = 1;
            if(++n_part_cnt <= 8)
                break;
            n_part_cnt = 1;
            if(!Inc_1_2_4_8_16(chunk_size))
                break;
            if(!Inc_1_2_4_8_16(c_per_gpr))
                break;
            if(!Inc_1_2_4_8_16(c_mult))
                break;
            if(!Inc_1_2_4_8_16(k_per_gpr))
                break;
            if(!Inc_1_2_4_8_16(k_mult))
                break;
            if(!Inc_1_2_4(n_per_gpr))
                break;
            if(!IncPack<1, 0>(short_store))
                break;
            if(!IncPack<1, 0, 2, 3, 4>(data_prefetch))
                break;
            // All the fields (components) of performance confic have wrapped around.
            return false;
        } while(false);
    }
    else
    {
        if(!use_spare_set) // first or second perfParam pack
        {
            // if SetNextValue executed from default class state
            if((read_size == 1) && (c_per_gpr == 1) && (c_mult == 1) && (k_mult == 1) &&
               (k_per_gpr == 1) && (chunk_size == 1) && (n_per_gpr == 1) && (n_part_cnt == 1) &&
               (short_store == 1) && (data_prefetch == 1))
            {
                read_size   = 2;
                c_per_gpr   = 2;
                c_mult      = 2;
                k_mult      = 2;
                short_store = 0;
                return true;
            }

            do
            {
                if(!IncPack<2, 4>(read_size))
                    break;
                if(!IncPack<1, 2, 4>(chunk_size))
                    break;
                if(!IncPack<2, 4, 8, 16>(c_per_gpr))
                    break;
                if(!IncPack<2, 4, 8>(c_mult))
                    break;
                if(!IncPack<1, 2, 4, 8>(k_per_gpr))
                    break;
                if(!IncPack<2, 4, 8>(k_mult))
                    break;
                if(!IncPack<1, 2>(n_per_gpr))
                    break;
                if(!IncPack<1, 2, 4>(n_part_cnt))
                    break;
                if(!IncPack<0>(short_store))
                    break;
                if(!IncPack<1, 0, 2, 3>(data_prefetch))
                    break;
                return false;
            } while(false);
        }
        else
        {
            do
            {
                if(!IncPack<1, 2, 4>(read_size))
                    break;
                if(!IncPack<1, 2, 4, 8>(chunk_size))
                    break;
                if(!IncPack<1, 2, 4, 8, 16>(c_per_gpr))
                    break;
                if(!IncPack<1, 2, 4, 8>(c_mult))
                    break;
                if(!IncPack<1, 2, 4, 8>(k_per_gpr))
                    break;
                if(!IncPack<1, 2, 4, 8>(k_mult))
                    break;
                if(!IncPack<1, 2>(n_per_gpr))
                    break;
                if(!IncPack<1, 2, 4>(n_part_cnt))
                    break;
                if(!IncPack<1, 0>(short_store))
                    break;
                if(!IncPack<1, 0, 2, 3>(data_prefetch))
                    break;
                return false;
            } while(false);
        }
    }
    return true;
}

PerformanceConfigConvAsmBwdWrW1x1::PerformanceConfigConvAsmBwdWrW1x1(int chunk_size_,
                                                                     int c_per_gpr_,
                                                                     int c_mult_,
                                                                     int k_per_gpr_,
                                                                     int k_mult_,
                                                                     int n_per_gpr_,
                                                                     int n_part_cnt_,
                                                                     int read_size_,
                                                                     int short_store_,
                                                                     int data_prefetch_,
                                                                     bool use_spare_set_)
    : chunk_size(chunk_size_),
      c_per_gpr(c_per_gpr_),
      c_mult(c_mult_),
      k_per_gpr(k_per_gpr_),
      k_mult(k_mult_),
      n_per_gpr(n_per_gpr_),
      n_part_cnt(n_part_cnt_),
      read_size(read_size_),
      short_store(short_store_),
      data_prefetch(data_prefetch_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceConfigConvAsmBwdWrW1x1::operator==(
    const PerformanceConfigConvAsmBwdWrW1x1& other) const
{
    // clang-format off
    return chunk_size == other.chunk_size
        && c_per_gpr == other.c_per_gpr
        && c_mult == other.c_mult
        && k_per_gpr == other.k_per_gpr
        && k_mult == other.k_mult
        && n_per_gpr == other.n_per_gpr
        && n_part_cnt == other.n_part_cnt
        && read_size == other.read_size
        && short_store == other.short_store
        && data_prefetch == other.data_prefetch
        && use_spare_set == other.use_spare_set; // clang-format on
}

bool PerformanceConfigConvAsmBwdWrW1x1::IsValidValue() const
{
    // clang-format off
    return Is_1_2_4_8_16(c_per_gpr)
        && Is_1_2_4_8_16(c_mult)
        && Is_1_2_4_8_16(k_per_gpr)
        && Is_1_2_4_8_16(k_mult)
        && (1 <= read_size && read_size <= 4)
        && (IsFromPack<1,2,4>(n_per_gpr))
        && (n_part_cnt >= 1 && n_part_cnt <= 8)
        && Is_1_2_4(GetHWPerGpr())
        && Is_1_2_4_8_16(chunk_size)
        && IsFromPack<0, 1>(short_store)
        && IsFromPack<0, 1, 2, 3, 4>(data_prefetch); // clang-format on
}

bool PerformanceConfigConvAsmBwdWrW1x1::IsValid(const ExecutionContext& ctx,
                                                const ProblemDescription& problem) const
{
    if(!IsValidValue())
        return false;
    if(!((chunk_size * c_per_gpr) >= 16 && ((chunk_size == 1 || c_per_gpr * chunk_size <= 16))))
        return false;

    if(!(k_per_gpr <= c_per_gpr))
        return false;

    if(!(c_per_gpr * n_per_gpr * GetHWPerGpr() * chunk_size == wave_size))
        return false;
    if(problem.GetOutDataType() == miopenHalf || problem.GetOutDataType() == miopenBFloat16)
    {
        if(short_store == 0)
        {
            const int sequential_channels = 2;
            if((c_mult % sequential_channels) != 0 ||
               (problem.GetOutChannels() % sequential_channels) != 0)
                return false;
        }
    }
    else
    {
        if(short_store == 1)
            return false;
    }

    int acc_gprs      = c_mult * k_mult * k_per_gpr;
    int bfp16_convert = 0;

    const std::string name = ctx.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos && name.find("gfx9") == std::string::npos)
    {
        bfp16_convert = 0;
    }
    else
    {
        bfp16_convert =
            (problem.GetOutDataType() == miopenBFloat16) ? ((c_mult + k_mult) * read_size) : 0;
    }

    if(!(acc_gprs + 12 + (c_mult + k_mult) * read_size * (data_prefetch + 1) + bfp16_convert <=
         (n_part_cnt > 4 ? 128 : 256)))
    {
        return false;
    }
    if(n_part_cnt > 1)
    {
        auto lds_size =
            (n_part_cnt - 1) * solver::wave_size * static_cast<int>(sizeof(float)) * acc_gprs;
        if(!(lds_size <= (1 << 16)))
            return false;
    }
    return true;
}

void PerformanceConfigConvAsmBwdWrW1x1::HeuristicInit(const ExecutionContext& ctx,
                                                      const ProblemDescription& problem)
{
    short_store =
        (problem.GetOutDataType() == miopenHalf || problem.GetOutDataType() == miopenBFloat16) ? 1
                                                                                               : 0;
    read_size = 4;
    n_per_gpr =
        (problem.GetBatchSize() >= 4 && (AsmImgHeight(problem) * AsmImgWidth(problem)) <= 128) ? 4
                                                                                               : 1;
    data_prefetch      = 1;
    const auto c_k_256 = problem.GetOutChannels() * problem.GetInChannels() / 256; // C*K/256
    if(c_k_256 < 2)
    {
        c_per_gpr  = 1;
        chunk_size = 16 / c_per_gpr;
        c_mult     = 1;
        k_per_gpr  = 1;
        k_mult     = 1;
        n_per_gpr  = 1;
        n_part_cnt = 1;
        read_size  = 1;
    }
    else if(c_k_256 < static_cast<std::size_t>(2 * 4))
    {
        c_per_gpr  = 1;
        chunk_size = 16 / c_per_gpr;
        c_mult     = 2;
        k_per_gpr  = 1;
        k_mult     = 2;
        n_per_gpr  = 1;
        n_part_cnt = 1;
        read_size  = 1;
    }
    else if(c_k_256 < static_cast<std::size_t>(2 * 4 * 4))
    {
        c_per_gpr  = 2;
        chunk_size = 16 / c_per_gpr;
        c_mult     = 2;
        k_per_gpr  = 2;
        k_mult     = 2;
        n_per_gpr  = 2;
        n_part_cnt = 2;
        read_size  = 2;
    }
    else if(c_k_256 < static_cast<std::size_t>(2 * 4 * 4 * 4))
    {
        c_per_gpr  = 2;
        chunk_size = 16 / c_per_gpr;
        c_mult     = 4;
        k_per_gpr  = 2;
        k_mult     = 4;
        n_per_gpr  = 2;
        n_part_cnt = 2;
        read_size  = 4;
    }
    else
    {
        c_per_gpr  = 2;
        chunk_size = 16 / c_per_gpr;
        c_mult     = 4;
        k_per_gpr  = 2;
        k_mult     = 4;
        n_per_gpr  = 4;
        n_part_cnt = 4;
        read_size  = 4;
    }

    if(!IsValid(ctx, problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");

        c_per_gpr     = 2;
        chunk_size    = 16 / c_per_gpr;
        c_mult        = 1;
        k_per_gpr     = 2;
        k_mult        = 1;
        n_per_gpr     = 1;
        n_part_cnt    = 1;
        read_size     = 1;
        data_prefetch = 0;
        assert(IsValid(ctx, problem));
    }
    MIOPEN_LOG_I(ToString());
}

PerformanceConfigConvAsmBwdWrW1x1
ConvAsmBwdWrW1x1::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                              const ProblemDescription& problem) const
{
    PerformanceConfigConvAsmBwdWrW1x1 pp;
    pp.HeuristicInit(ctx, problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsmBwdWrW1x1::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvAsmBwdWrW1x1& config) const
{
    return config.IsValidValue() && config.IsValid(ctx, problem);
}

bool ConvAsmBwdWrW1x1::IsApplicable(const ExecutionContext& ctx,
                                    const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!ctx.rmv.IsV2orV3())
        return false;
    if(problem.IsTensorsCasted())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = ctx.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos && name.find("gfx9") == std::string::npos)
        return false;
    if(!problem.IsLayoutDefault())
        return false;

    if(name == "gfx90a" && problem.IsGfx90aFp16altRequired())
        return false;

    // clang-format off
    bool ok = (problem.GetPadW() == 0       // -q  pad_w
        && problem.GetPadH() == 0           // -p  pad_h
        && problem.GetKernelStrideW() <= 2  // -v  stride_w
        && problem.GetKernelStrideH() <= 2  // -u  stride_h
        && problem.GetKernelStrideW() == problem.GetKernelStrideH()
        && problem.GetWeightsWidth() == 1   // -x  S wei_w
        && problem.GetWeightsHeight() == 1  // -y  R wei_h
        && problem.GetDilationW() == 1
        && problem.GetDilationH() == 1
        && problem.GetBias() == 0
        && (problem.IsFp32() || problem.IsFp16() || problem.IsBfp16())
        && problem.GetInLayout() == "NCHW"
        && problem.GetGroupCount() == 1);
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    // Check limits:
    const auto h_w     = static_cast<uint64_t>(AsmImgHeight(problem)) * AsmImgWidth(problem);
    const auto r_s     = problem.GetWeightsHeight() * problem.GetWeightsWidth();
    const auto c_h_w   = problem.GetOutChannels() * h_w; // C*H*W
    const auto k_h_w   = problem.GetInChannels() * h_w;  // K*H*W
    const auto n_c_h_w = problem.GetBatchSize() * c_h_w; // N*C*H*W
    const auto n_k_h_w = problem.GetBatchSize() * k_h_w; // N*K*H*W
    const auto c_k_r_s = problem.GetOutChannels() * problem.GetInChannels() * r_s; // C*K*R*S
    ok = problem.GetBatchSize() < std::pow(2, 16)       // -n   N batch_size
         && problem.GetOutChannels() < std::pow(2, 16)  // -c   C input_channels
         && problem.GetInChannels() < std::pow(2, 16)   // -k   K output_channels
         && c_h_w < std::pow(2, 24)
         && k_h_w < std::pow(2, 24)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29); // clang-format on
    return ok;
}

static int divide_round_plus_inf(const int x, const int y)
{
    assert(x >= 0 && y > 0);
    if(x % y != 0)
        return x / y + 1;
    return x / y;
}

size_t ConvAsmBwdWrW1x1::GetWorkspaceSize(const ExecutionContext&,
                                          const ProblemDescription& problem) const
{
    if(UseSubsample(problem))
    {
        auto data_len = GetTypeSize(problem.GetOutDataType());
        auto in_batch_stride =
            problem.GetInStrideH() * problem.GetInHeight() * problem.GetOutChannels();
        return in_batch_stride * problem.GetBatchSize() * data_len;
    }
    else
        return 0;
}

ConvSolution ConvAsmBwdWrW1x1::GetSolution(const ExecutionContext& ctx,
                                           const ProblemDescription& problem,
                                           const PerformanceConfigConvAsmBwdWrW1x1& config) const
{

    ConvSolution result;
    std::ostringstream options;

    assert(problem.GetPadH() == 0 && problem.GetPadW() == 0);
    int data_len = GetTypeSize(problem.GetOutDataType());
    if(UseSubsample(problem))
    {
        // subsampled input, in_height equals to image size after downsampling
        int in_batch_stride =
            problem.GetInStrideH() * problem.GetInHeight() * problem.GetOutChannels();
        int write_unit   = (problem.GetInWidth() % 4 == 0)   ? 4
                           : (problem.GetInWidth() % 3 == 0) ? 3
                           : (problem.GetInWidth() % 2 == 0) ? 2
                                                             : 1;
        int n_grp0_size0 = 256;

        // clang-format off
        const auto subsample_kernel_compilation_options =
            std::string(" -DMLO_GRP0_SZ0=") + std::to_string(n_grp0_size0) +
            std::string(" -DMLO_GRP0_SZ1=1 ") + std::string(" -DMLO_GRP0_SZ2=1 ") +
            std::string(" -DMLO_FILTER0_STRIDE0=") + std::to_string(problem.GetKernelStrideW()) +
            std::string(" -DMLO_FILTER0_STRIDE1=") + std::to_string(problem.GetKernelStrideH()) +
            std::string(" -DMLO_WRITE_UNIT=") + std::to_string(write_unit) +
            std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(problem.GetInChannelStride()) +
            std::string(" -DMLO_OUT_STRIDE=") + std::to_string(problem.GetInStrideH()) +
            std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(in_batch_stride) +
            std::string(" -DMLO_IN0_BATCH_STRIDE=") + std::to_string(problem.GetOutBatchStride()) +
            std::string(" -DMLO_IN0_CHANNEL_STRIDE=") + std::to_string(problem.GetOutChannelStride()) +
            std::string(" -DMLO_IN0_STRIDE=") + std::to_string(problem.GetOutStrideH()) +
            ctx.general_compile_options;
        // clang-format on

        KernelInfo kernel;

        kernel.l_wk.push_back(n_grp0_size0);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        // output is number of subsampled input maps
        size_t gbl_wk0 = (in_batch_stride / write_unit);
        size_t gbl_wk1 = problem.GetBatchSize();
        size_t gbl_wk2 = 1;

        kernel.g_wk.push_back(gbl_wk0);
        kernel.g_wk.push_back(gbl_wk1);
        kernel.g_wk.push_back(gbl_wk2);

        kernel.kernel_file = "MIOpenUtilKernels3.cl";

        kernel.kernel_name = "SubSample";

        kernel.comp_options = subsample_kernel_compilation_options;

        result.construction_params.push_back(kernel);
    }
    result.workspace_sz = GetWorkspaceSize(ctx, problem);
    GenerateClangDefsym(options, "stride_h", 1);
    GenerateClangDefsym(options, "stride_w", 1);
    GenerateClangDefsym(options, "img_h", AsmImgHeight(problem)); // H
    GenerateClangDefsym(options, "img_w", AsmImgWidth(problem));  // W
    GenerateClangDefsym(options, "out_h", AsmImgHeight(problem)); // output H
    GenerateClangDefsym(options, "out_w", AsmImgWidth(problem));  // output W

    GenerateClangDefsym(options, "batch_size", problem.GetBatchSize()); // N
    // Note that problem.n_outputs and problem.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "input_channels", problem.GetOutChannels()); // C
    GenerateClangDefsym(options, "output_channels", problem.GetInChannels()); // K
    GenerateClangDefsym(options, "wei_h", problem.GetWeightsHeight());        // R
    GenerateClangDefsym(options, "wei_w", problem.GetWeightsWidth());         // S
    GenerateClangDefsym(options, "pad_h", problem.GetPadH());
    GenerateClangDefsym(options, "pad_w", problem.GetPadW());
    GenerateClangDefsym(options, "weights_layout", 0);
    GenerateClangDefsym(options, "reverse_weights", 0);
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    // Perf tune:
    GenerateClangDefsym(options, "do_not_use_default_perf_params", 1);

    GenerateClangDefsym(options, "acc_type", 1);
    const unsigned int buf_type = problem.GetOutDataType() == miopenHalf    ? 2
                                  : problem.GetOutDataType() == miopenFloat ? 1
                                                                            : 3;
    GenerateClangDefsym(options, "buf_type", buf_type);

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
                   problem.GetOutChannels(),
                   AsmImgHeight(problem),
                   AsmImgWidth(problem),
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info obuf(MemLayout::NCHW,
                   problem.GetBatchSize(),
                   problem.GetInChannels(),
                   AsmImgHeight(problem),
                   AsmImgWidth(problem),
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info fbuf(
        MemLayout::NCHW, problem.GetInChannels(), problem.GetOutChannels(), 1, 1, 1, data_len);
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

    const PerformanceConfigConvAsmBwdWrW1x1* pcfg = &config;

    PerformanceConfigConvAsmBwdWrW1x1 fromEnv;
    {
        const auto s = env::value(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS);
        if(!s.empty()) // else nothing to parse.
        {
            if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(ctx, problem))
            {
                MIOPEN_LOG_E("MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS: "
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

    GenerateClangDefsym(options, "short_store", pcfg->GetShortStore());
    GenerateClangDefsym(options, "chunk_size", pcfg->GetChunkSize());
    GenerateClangDefsym(options, "c_per_gpr", pcfg->GetCPerGpr());
    GenerateClangDefsym(options, "c_mult", pcfg->GetCMult());
    GenerateClangDefsym(options, "k_per_gpr", pcfg->GetKPerGpr());
    GenerateClangDefsym(options, "k_mult", pcfg->GetKMult());
    GenerateClangDefsym(options, "n_per_gpr", pcfg->GetNPerGpr());
    GenerateClangDefsym(options, "n_part_cnt", pcfg->GetNPartCnt());
    GenerateClangDefsym(options, "hw_per_gpr", pcfg->GetHWPerGpr());
    GenerateClangDefsym(options, "read_size", pcfg->GetReadSize());
    GenerateClangDefsym(options, "data_prefetch", pcfg->GetDataPrefetch());

    KernelInfo kernel;

    kernel.comp_options = options.str();

    kernel.l_wk.clear(); // workgroupsize
    kernel.l_wk.push_back(static_cast<size_t>(solver::wave_size) * pcfg->GetNPartCnt());
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.clear(); // gridsize
    kernel.g_wk.push_back(static_cast<size_t>(solver::wave_size) * pcfg->GetNPartCnt());
    kernel.g_wk.push_back(
        divide_round_plus_inf(problem.GetOutChannels(), pcfg->GetCPerGpr() * pcfg->GetCMult()));
    kernel.g_wk.push_back(
        divide_round_plus_inf(problem.GetInChannels(), pcfg->GetKPerGpr() * pcfg->GetKMult()));

    kernel.kernel_file = "conv1x1wrw.s";
    kernel.kernel_name = "miopenGcnAsmConv1x1WrW";

    result.construction_params.push_back(kernel);

    int N, C, H, W, K, n_groups;
    GetCompiledInParameters(ctx, problem, &N, &C, &H, &W, &K, &n_groups);

    if(UseSubsample(problem))
    {
        result.invoker_factory = [N, C, H, W, K, n_groups](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                const auto ss_kernel   = handle.Run(kernels[0]);
                const auto main_kernel = handle.Run(kernels[1]);
                const auto& invoke_params =
                    primitive_params.CastTo<miopen::conv::WrWInvokeParams>();
                const auto& x         = invoke_params.tensors.x;
                const auto& dy        = invoke_params.tensors.dy;
                const auto& dw        = invoke_params.tensors.dw;
                const auto& workSpace = invoke_params.workSpace;
                auto elapsed          = 0.f;

                if(invoke_params.type != InvokeType::AutoTune)
                {
                    ss_kernel(x, workSpace);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }

                int unused       = 0;
                int* return_addr = nullptr;
                main_kernel(
                    N, C, H, W, K, n_groups, unused, unused, workSpace, dw, dy, return_addr);
                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    }
    else
    {
        result.invoker_factory = [N, C, H, W, K, n_groups](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                const auto main_kernel = handle.Run(kernels[0]);
                const auto& invoke_params =
                    primitive_params.CastTo<miopen::conv::WrWInvokeParams>();
                const auto& x    = invoke_params.tensors.x;
                const auto& dy   = invoke_params.tensors.dy;
                const auto& dw   = invoke_params.tensors.dw;
                int unused       = 0;
                int* return_addr = nullptr;
                main_kernel(N, C, H, W, K, n_groups, unused, unused, x, dw, dy, return_addr);
            };
        };
    }

    return result;
}

PerformanceConfigConvAsmBwdWrW1x1 ConvAsmBwdWrW1x1::Search(const ExecutionContext& ctx,
                                                           const ProblemDescription& problem,
                                                           const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

} // namespace conv
} // namespace solver
} // namespace miopen
