/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

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

template <int L, int H>
inline static bool IsLinear(const int v)
{
    static_assert(L <= H, "L <= H");
    return L <= v && v <= H;
}

static inline size_t divide_round_plus_inf(const size_t x, const unsigned y)
{
    if(x % y != 0)
        return x / y + 1;
    return x / y;
}

enum class MemLayout : int
{
    NCHW = 0,
    CNHW = 1,
};

struct config_helper
{
    config_helper(const ProblemDescription& problem, const PerformanceConfigConvAsm1x1UV2& config)
    {
        if(problem.IsDirectionForward())
        {
            stride_w   = problem.GetKernelStrideW();
            stride_h   = problem.GetKernelStrideH();
            dilation_w = 1;
            dilation_h = 1;
        }
        else
        {
            stride_w   = 1;
            stride_h   = 1;
            dilation_w = problem.GetKernelStrideW();
            dilation_h = problem.GetKernelStrideH();
        }

        in_strided_w = divide_round_plus_inf(problem.GetInWidth(), stride_w);
        in_strided_h = divide_round_plus_inf(problem.GetInHeight(), stride_h);

        w_per_wave  = static_cast<int>(divide_round_plus_inf(config.dwords_per_ld, stride_w) *
                                      config.w_mult * (config.chunk_size / config.h_per_chunk));
        h_per_wave  = config.h_per_chunk * config.h_mult;
        gid_hw_size = static_cast<int>(divide_round_plus_inf(in_strided_w, h_per_wave) *
                                       divide_round_plus_inf(in_strided_h, w_per_wave));
    }

    int stride_w, stride_h;
    // data dilation
    int dilation_w, dilation_h;

    int in_strided_w, in_strided_h;
    int w_per_wave, h_per_wave;
    int gid_hw_size;
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

bool PerformanceConfigConvAsm1x1UV2::SetNextValue(const ProblemDescription&)
{
    // Increment with wrap-around:
    do
    {
        if(!env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_SEARCH_OPTIMIZED))
        {
            if(!IncPack<16, 32, 64>(chunk_size))
                break;
            if(!IncPack<1, 2, 3>(dwords_per_ld))
                break;
            if(use_spare_set ? (!IncPack<1, 2, 3, 4>(k_mult)) : (!IncPack<8, 16>(k_mult)))
                break;
            if(!IncPack<1, 2, 4>(c_mult))
                break;
            if(!IncPack<1, 2>(n_mult))
                break;
            if(!IncPack<2, 4, 8>(h_per_chunk))
                break;
            if(!IncPack<1, 2>(w_mult))
                break;
            if(!IncPack<1, 2>(h_mult))
                break;
            if(!IncPack<2, 4>(waves_k_in_group))
                break;
            if(!IncPack<1, 2>(waves_c_in_group))
                break;
        }
        else
        {
            if(++k_mult <= 16)
                break;
            k_mult = 1;
            if(++c_mult <= 16)
                break;
            c_mult = 1;
            if(++n_mult <= 8)
                break;
            n_mult = 1;
            if(++w_mult <= 8)
                break;
            w_mult = 1;
            if(++h_mult <= 8)
                break;
            h_mult = 1;
            if(++waves_k_in_group <= 8)
                break;
            waves_k_in_group = 1;
            if(++waves_c_in_group <= 8)
                break;
            waves_c_in_group = 1;
            if(!IncPack<1, 2, 4, 8, 16, 32, 64>(chunk_size))
                break;
            if(!IncPack<1, 2, 3, 4>(dwords_per_ld))
                break;
            if(!IncPack<1, 2, 4, 8, 16, 32, 64>(h_per_chunk))
                break;
        }
        return false;
    } while(false);
    return true;
}

PerformanceConfigConvAsm1x1UV2::PerformanceConfigConvAsm1x1UV2(bool spare)
    : PerformanceConfigConvAsm1x1UV2(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, spare)
{
    if(!env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_SEARCH_OPTIMIZED))
    {
        k_mult           = spare ? 1 : 8;
        chunk_size       = 16;
        h_per_chunk      = 2;
        waves_k_in_group = 2;
    }
}

PerformanceConfigConvAsm1x1UV2::PerformanceConfigConvAsm1x1UV2(int chunk_size_,
                                                               int dwords_per_ld_,
                                                               int k_mult_,
                                                               int c_mult_,
                                                               int n_mult_,
                                                               int w_mult_,
                                                               int h_mult_,
                                                               int h_per_chunk_,
                                                               int waves_k_in_group_,
                                                               int waves_c_in_group_,
                                                               bool use_spare_set_)
    : chunk_size(chunk_size_),
      dwords_per_ld(dwords_per_ld_),
      k_mult(k_mult_),
      c_mult(c_mult_),
      n_mult(n_mult_),
      w_mult(w_mult_),
      h_mult(h_mult_),
      h_per_chunk(h_per_chunk_),
      waves_k_in_group(waves_k_in_group_),
      waves_c_in_group(waves_c_in_group_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceConfigConvAsm1x1UV2::operator==(const PerformanceConfigConvAsm1x1UV2& other) const
{
    // clang-format off
    return chunk_size == other.chunk_size
        && dwords_per_ld == other.dwords_per_ld
        && c_mult == other.c_mult
        && k_mult == other.k_mult
        && n_mult == other.n_mult
        && w_mult == other.w_mult
        && h_mult == other.h_mult
        && h_per_chunk == other.h_per_chunk
        && waves_k_in_group == other.waves_k_in_group
        && waves_c_in_group == other.waves_c_in_group
        && use_spare_set == other.use_spare_set; // clang-format on
}

bool PerformanceConfigConvAsm1x1UV2::IsValidValue() const
{
    // clang-format off
    return IsFromPack<1, 2, 4, 8, 16, 32, 64>(chunk_size)
        && IsFromPack<1, 2, 3, 4>(dwords_per_ld)
        && IsLinear<1,32>(c_mult)
        && IsLinear<1,32>(k_mult)
        && IsLinear<1,32>(n_mult)
        && IsLinear<1,32>(w_mult)
        && IsLinear<1,32>(h_mult)
        && IsFromPack<1, 2, 4, 8, 16, 32, 64>(h_per_chunk)
        && IsLinear<1, 8>(waves_k_in_group)
        && IsLinear<1,8>(waves_c_in_group)
        ; // clang-format on
}

bool PerformanceConfigConvAsm1x1UV2::IsValid(const ProblemDescription& problem) const
{
    const auto elements_in_dword = 4 / static_cast<int>(GetTypeSize(problem.GetInDataType()));

    if(!IsValidValue())
        return false;
    if(!(waves_c_in_group * waves_k_in_group <= 16))
        return false;
    if(!(waves_c_in_group <= problem.GetInChannels()))
        return false;
    if(!(h_per_chunk <= chunk_size))
        return false;
    if(!(static_cast<std::size_t>(k_mult) * waves_k_in_group <= problem.GetOutChannels()))
        return false;

    // cppcheck-suppress unreadVariable
    const config_helper uv_lj(problem, *this);

    const auto elements_per_ld = (dwords_per_ld * elements_in_dword);
    const auto active_elements =
        static_cast<int>(divide_round_plus_inf(elements_per_ld, uv_lj.stride_w));

    const auto in_gprs  = dwords_per_ld * w_mult * h_mult * c_mult * n_mult * 2;
    const auto acc_gprs = active_elements * w_mult * h_mult * k_mult * n_mult;

    auto vgpr_ex                  = 0;
    const auto fp32_divider_limit = 1 << 16;
    // if this statement true:
    // asm macros get_gloabl_H_W_tids will use 64bit divizion
    // which uses more vtmp registers than 32bit version, so need to alocate  missing part
    if(uv_lj.gid_hw_size >= fp32_divider_limit)
        vgpr_ex++;

    if(uv_lj.dilation_w > 1)
    {
        const auto store_buffer_size = (dwords_per_ld == 1) ? 2 : 4;
        if((in_gprs - 1) < store_buffer_size)
            vgpr_ex += store_buffer_size - (in_gprs - 1);
    }
    else if(uv_lj.gid_hw_size >= fp32_divider_limit && in_gprs < 2)
    {
        vgpr_ex++;
    }

    const auto vgprs = 5 + in_gprs + acc_gprs + vgpr_ex + 1;
    if(!(vgprs < 256))
        return false;
    const auto max_waves_per_CU = (256 / vgprs) * 4;
    if(!(max_waves_per_CU >= waves_c_in_group * waves_k_in_group))
        return false;
    const auto sgprs = 25 + 2 * k_mult * c_mult;
    if(!(sgprs < 102))
        return false;
    const auto total_n_blocks = (problem.GetBatchSize() + GetNPerGpr() - 1) / GetNPerGpr();
    if(!(n_mult <= total_n_blocks))
        return false;

    const auto c_per_wave = (problem.GetInChannels() + waves_c_in_group - 1) / waves_c_in_group;
    const auto c_per_last_wave = problem.GetInChannels() - (c_per_wave * (waves_c_in_group - 1));

    if(problem.IsDirectionBackwardData() && !(problem.GetOutChannels() % k_mult == 0))
        return false;

    {
        // cppcheck-suppress unreadVariable
        buff_info ibuf(MemLayout::NCHW,
                       problem.GetBatchSize(),
                       problem.GetInChannels(),
                       problem.GetInHeight(),
                       problem.GetInWidth(),
                       1,
                       GetTypeSize(problem.GetInDataType()));
        // cppcheck-suppress unreadVariable
        buff_info obuf(MemLayout::NCHW,
                       problem.GetBatchSize(),
                       problem.GetOutChannels(),
                       problem.GetOutHeight(),
                       problem.GetOutWidth(),
                       1,
                       GetTypeSize(problem.GetOutDataType()));
        int n_miss = n_mult * GetNPerGpr() - 1;
        if((static_cast<int64_t>(problem.GetInChannels()) + n_miss) * ibuf.byte_stride.nk >=
               (1LL << 31) ||
           (static_cast<int64_t>(problem.GetOutChannels()) + n_miss) * obuf.byte_stride.nk >=
               (1LL << 31))
            return false;
    }
    return (c_per_wave % c_mult == 0) && (c_per_last_wave % c_mult == 0);
}

void PerformanceConfigConvAsm1x1UV2::HeuristicInit(const ProblemDescription& problem)
{
    int c_check   = problem.IsDirectionForward() ? problem.GetInChannels() : 0;
    int k_check   = problem.IsDirectionForward() ? 0 : problem.GetInChannels();
    chunk_size    = 16;
    dwords_per_ld = 1;
    c_mult        = (c_check % 2 == 0) ? 2 : ((c_check % 3 == 0) ? 3 : 1);
    k_mult      = (k_check % 8 == 0) ? 8 : ((k_check % 4 == 0) ? 4 : ((k_check % 3 == 0) ? 3 : 1));
    n_mult      = 1;
    w_mult      = 1;
    h_mult      = 1;
    h_per_chunk = 4;
    waves_c_in_group = 1;
    waves_k_in_group = 1;

    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        h_per_chunk = chunk_size;
    }
    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        c_mult = 1;
        k_mult = 1;
    }
    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        h_per_chunk = 1;
    }
    if(!IsValid(problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString());
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
    }
    else
    {
        MIOPEN_LOG_I(ToString());
    }
}

PerformanceConfigConvAsm1x1UV2
ConvAsm1x1UV2::GetDefaultPerformanceConfig(const ExecutionContext&,
                                           const ProblemDescription& problem) const
{
    PerformanceConfigConvAsm1x1UV2 pp;
    pp.HeuristicInit(problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsm1x1UV2::IsValidPerformanceConfig(const ExecutionContext&,
                                             const ProblemDescription& problem,
                                             const PerformanceConfigConvAsm1x1UV2& config) const
{
    return config.IsValidValue() && config.IsValid(problem);
}

bool ConvAsm1x1UV2::IsApplicable(const ExecutionContext& ctx,
                                 const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsDirectionForward() || problem.IsDirectionBackwardData()))
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!ctx.rmv.IsV2orV3())
        return false;
    if(!problem.IsFp32())
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

    if(problem.IsTensorsCasted() || problem.IsFp8() || problem.IsBfp8())
        return false;

    const auto elements_in_dword = 4 / GetTypeSize(problem.GetInDataType());
    // clang-format off
    const auto img_hw = problem.GetOutHeight() * problem.GetOutWidth();
    bool ok = (problem.GetPadW() == 0
        && problem.GetPadH() == 0
        && problem.GetWeightsWidth() == 1
        && problem.GetWeightsHeight() == 1
        && problem.GetKernelStrideW() <= 2
        && problem.GetKernelStrideW() == problem.GetKernelStrideH()
        && problem.GetDilationW() == 1
        && problem.GetDilationH() == 1
        && problem.GetBias() == 0
        && problem.GetInLayout() == "NCHW"
        && problem.GetGroupCount() == 1
        && img_hw >= elements_in_dword);

    if(problem.GetKernelStrideW() == 1){
        // save search time for non-strided convolutions
        // now this kernel much slower than conv1x1u.s
        ok = false;
    }
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }

    // Check limits:
    const auto h_w     = problem.GetInHeight() * problem.GetInWidth();
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
    if(ok)
    {
        /// Hotfix for issue 1810 (HeuristicInit problem of this solver).
        /// Modified copy from PerformanceConfigConvAsm1x1UV2::IsValid()
        /// \todo Refactor this.
        const auto& config = problem; // alias
        // cppcheck-suppress unreadVariable
        buff_info ibuf(MemLayout::NCHW,
                       config.GetBatchSize(),
                       config.GetInChannels(),
                       config.GetInHeight(),
                       config.GetInWidth(),
                       1,
                       GetTypeSize(config.GetInDataType()));
        // cppcheck-suppress unreadVariable
        buff_info obuf(MemLayout::NCHW,
                       config.GetBatchSize(),
                       config.GetOutChannels(),
                       config.GetOutHeight(),
                       config.GetOutWidth(),
                       1,
                       GetTypeSize(config.GetOutDataType()));

        const int eurictic_init_min_n_mult     = 1;
        const int eurictic_init_max_chunk_size = 16;
        const int n_miss = eurictic_init_min_n_mult * (64 / eurictic_init_max_chunk_size) - 1;

        if((static_cast<int64_t>(config.GetInChannels()) + n_miss) * ibuf.byte_stride.nk >=
               (1LL << 31) ||
           (static_cast<int64_t>(config.GetOutChannels()) + n_miss) * obuf.byte_stride.nk >=
               (1LL << 31))
            ok = false;
    }
    return ok;
}

ConvSolution ConvAsm1x1UV2::GetSolution(const ExecutionContext& ctx,
                                        const ProblemDescription& problem,
                                        const PerformanceConfigConvAsm1x1UV2& config) const
{
    ConvSolution result;
    std::ostringstream options;

    result.workspace_sz = 0;

    int data_len                               = GetTypeSize(problem.GetOutDataType());
    const PerformanceConfigConvAsm1x1UV2* pcfg = &config;

    PerformanceConfigConvAsm1x1UV2 fromEnv;
    {
        const auto s = env::value(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_PERF_VALS);
        if(!s.empty()) // else nothing to parse.
        {
            if(!fromEnv.Deserialize(s) || !fromEnv.IsValidValue())
            {
                MIOPEN_LOG_E("MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_PERF_VALS: "
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

    // cppcheck-suppress unreadVariable
    const config_helper uv_lj(problem, *pcfg);

    GenerateClangDefsym(options, "stride_h", uv_lj.stride_h);
    GenerateClangDefsym(options, "stride_w", uv_lj.stride_w);

    GenerateClangDefsym(options, "idilation_h", uv_lj.dilation_h);
    GenerateClangDefsym(options, "idilation_w", uv_lj.dilation_w);

    GenerateClangDefsym(options, "img_h", problem.GetInHeight()); // H
    GenerateClangDefsym(options, "img_w", problem.GetInWidth());  // W

    GenerateClangDefsym(options, "out_h", problem.GetOutHeight()); // H
    GenerateClangDefsym(options, "out_w", problem.GetOutWidth());  // W

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

    // cppcheck-suppress unreadVariable
    buff_info ibuf(MemLayout::NCHW,
                   problem.GetBatchSize(),
                   problem.GetInChannels(),
                   problem.GetInHeight(),
                   problem.GetInWidth(),
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info obuf(MemLayout::NCHW,
                   problem.GetBatchSize(),
                   problem.GetOutChannels(),
                   problem.GetOutHeight(),
                   problem.GetOutWidth(),
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

    GenerateClangDefsym(options, "chunk_size", pcfg->GetChunkSize());
    GenerateClangDefsym(options, "dwords_per_ld", pcfg->GetDwordsPerLd());
    GenerateClangDefsym(options, "k_mult", pcfg->GetKMult());
    GenerateClangDefsym(options, "c_mult", pcfg->GetCMult());
    GenerateClangDefsym(options, "n_mult", pcfg->GetNMult());
    GenerateClangDefsym(options, "w_mult", pcfg->GetWMult());
    GenerateClangDefsym(options, "h_mult", pcfg->GetHMult());
    GenerateClangDefsym(options, "h_per_chunk", pcfg->GetHPerChunk());
    GenerateClangDefsym(options, "waves_k_in_group", pcfg->GetWavesKInGroup());
    GenerateClangDefsym(options, "waves_c_in_group", pcfg->GetWavesCInGroup());

    KernelInfo kinfo;
    kinfo.comp_options = options.str();

    const int waves_in_group = pcfg->GetWavesCInGroup() * pcfg->GetWavesKInGroup();
    kinfo.l_wk.clear(); // workgroupsize
    kinfo.l_wk.push_back(64ULL * waves_in_group);
    kinfo.l_wk.push_back(1);
    kinfo.l_wk.push_back(1);

    kinfo.g_wk.clear(); // gridsize

    const int n_per_wave = pcfg->GetNMult() * pcfg->GetNPerGpr();
    const int k_per_wave = pcfg->GetKMult() * pcfg->GetWavesKInGroup();

    kinfo.g_wk.push_back(kinfo.l_wk[0] *
                         divide_round_plus_inf(uv_lj.in_strided_w, uv_lj.w_per_wave) *
                         divide_round_plus_inf(uv_lj.in_strided_h, uv_lj.h_per_wave));

    kinfo.g_wk.push_back(divide_round_plus_inf(problem.GetOutChannels(), k_per_wave));

    kinfo.g_wk.push_back(divide_round_plus_inf(problem.GetBatchSize(), n_per_wave));

    kinfo.kernel_file = "conv1x1u_stride2.s";
    kinfo.kernel_name = "miopenGcnAsmConv1x1U_stride2";

    result.construction_params.push_back(kinfo);

    {
        int N, C, H, W, K, n_groups;
        GetCompiledInParameters(ctx, problem, &N, &C, &H, &W, &K, &n_groups);
        result.invoker_factory =
            miopen::conv::MakeGcnAsm1x1UInvokerFactory(N, C, H, W, K, n_groups);
    }

    return result;
}

PerformanceConfigConvAsm1x1UV2 ConvAsm1x1UV2::Search(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem,
                                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

} // namespace conv
} // namespace solver
} // namespace miopen
