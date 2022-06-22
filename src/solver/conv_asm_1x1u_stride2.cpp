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
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2)

namespace miopen {
namespace solver {

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

enum class MemLayout : int
{
    NCHW = 0,
    CNHW = 1,
};

struct config_helper
{
    config_helper(const ConvolutionContext& conf, const PerformanceConfigConvAsm1x1UV2& perf)
    {
        if(conf.direction.IsForward())
        {
            stride_w   = conf.kernel_stride_w;
            stride_h   = conf.kernel_stride_h;
            dilation_w = 1;
            dilation_h = 1;
        }
        else
        {
            stride_w   = 1;
            stride_h   = 1;
            dilation_w = conf.kernel_stride_w;
            dilation_h = conf.kernel_stride_h;
        }

        in_strided_w = divide_round_plus_inf(conf.in_width, stride_w);
        in_strided_h = divide_round_plus_inf(conf.in_height, stride_h);

        w_per_wave  = static_cast<int>(divide_round_plus_inf(perf.dwords_per_ld, stride_w) *
                                      perf.w_mult * (perf.chunk_size / perf.h_per_chunk));
        h_per_wave  = perf.h_per_chunk * perf.h_mult;
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
        int count       = nk * c_hi * h * w * vec_c;
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

bool PerformanceConfigConvAsm1x1UV2::SetNextValue(const ConvolutionContext& /*config*/)
{
    // Increment with wrap-around:
    do
    {
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_SEARCH_OPTIMIZED{}))
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
    if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_SEARCH_OPTIMIZED{}))
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

bool PerformanceConfigConvAsm1x1UV2::IsValid(const ConvolutionContext& config) const
{
    const auto elements_in_dword = 4 / GetTypeSize(config.in_data_type);

    if(!IsValidValue())
        return false;
    if(!(waves_c_in_group * waves_k_in_group <= 16))
        return false;
    if(!(waves_c_in_group <= config.n_inputs))
        return false;
    if(!(h_per_chunk <= chunk_size))
        return false;
    if(!(k_mult * waves_k_in_group <= config.n_outputs))
        return false;

    // cppcheck-suppress unreadVariable
    const config_helper uv_lj(config, *this);

    const auto elements_per_ld = (dwords_per_ld * elements_in_dword);
    const auto active_elements = divide_round_plus_inf(elements_per_ld, uv_lj.stride_w);

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
    const auto total_n_blocks = (config.batch_sz + GetNPerGpr() - 1) / GetNPerGpr();
    if(!(n_mult <= total_n_blocks))
        return false;

    const auto c_per_wave      = (config.n_inputs + waves_c_in_group - 1) / waves_c_in_group;
    const auto c_per_last_wave = config.n_inputs - (c_per_wave * (waves_c_in_group - 1));

    if(config.direction.IsBackwardData() && !(config.n_outputs % k_mult == 0))
        return false;

    {
        // cppcheck-suppress unreadVariable
        buff_info ibuf(MemLayout::NCHW,
                       config.batch_sz,
                       config.n_inputs,
                       config.in_height,
                       config.in_width,
                       1,
                       GetTypeSize(config.in_data_type));
        // cppcheck-suppress unreadVariable
        buff_info obuf(MemLayout::NCHW,
                       config.batch_sz,
                       config.n_outputs,
                       config.out_height,
                       config.out_width,
                       1,
                       GetTypeSize(config.out_data_type));
        int n_miss = n_mult * GetNPerGpr() - 1;
        if((static_cast<long>(config.n_inputs) + n_miss) * ibuf.byte_stride.nk >= (1LL << 31) ||
           (static_cast<long>(config.n_outputs) + n_miss) * obuf.byte_stride.nk >= (1LL << 31))
            return false;
    }
    return (c_per_wave % c_mult == 0) && (c_per_last_wave % c_mult == 0);
}

void PerformanceConfigConvAsm1x1UV2::HeuristicInit(const ConvolutionContext& config)
{
    int c_check   = config.direction.IsForward() ? config.n_inputs : 0;
    int k_check   = config.direction.IsForward() ? 0 : config.n_inputs;
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

    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        h_per_chunk = chunk_size;
    }
    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        c_mult = 1;
        k_mult = 1;
    }
    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        h_per_chunk = 1;
    }
    if(!IsValid(config))
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

std::string PerformanceConfigConvAsm1x1UV2::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvAsm1x1UV2
ConvAsm1x1UV2::GetDefaultPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvAsm1x1UV2 pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsm1x1UV2::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                             const PerformanceConfigConvAsm1x1UV2& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

bool ConvAsm1x1UV2::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.Is2d())
        return false;
    if(!(params.direction.IsForward() || params.direction.IsBackwardData()))
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!params.rmv.IsV2orV3())
        return false;
    if(!params.IsFp32())
        return false;

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos && name.find("gfx9") == std::string::npos)
    {
        return false;
    }
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    if(name == "gfx90a" && params.conv_problem.IsGfx90aFp16altRequired())
        return false;

    const auto elements_in_dword = 4 / GetTypeSize(params.in_data_type);
    // clang-format off
    const auto img_hw = params.out_height * params.out_width;
    bool ok = (params.pad_w == 0
        && params.pad_h == 0
        && params.kernel_size_w == 1
        && params.kernel_size_h == 1
        && params.kernel_stride_w <= 2
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0
        && params.in_layout == "NCHW"
        && params.group_counts == 1
        && img_hw >= elements_in_dword);

    if(params.kernel_stride_w == 1){
        // save search time for non-strided convolutions
        // now this kernel much slower than conv1x1u.s
        ok = false;
    }
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }

    // Check limits:
    auto h_w = static_cast<long>(params.in_height) * params.in_width;
    const auto r_s     = static_cast<long>(params.kernel_size_h) * params.kernel_size_w;
    const auto c_h_w   = static_cast<long>(params.n_inputs) * h_w;    // C*H*W
    const auto k_h_w   = static_cast<long>(params.n_outputs) * h_w;   // K*H*W
    const auto n_c_h_w = static_cast<long>(params.batch_sz) * c_h_w;  // N*C*H*W
    const auto n_k_h_w = static_cast<long>(params.batch_sz) * k_h_w;  // N*K*H*W
    const auto c_k_r_s = static_cast<long>(params.n_inputs) * params.n_outputs * r_s; // C*K*R*S
    ok = params.batch_sz < std::pow(2, 16)      // -n   N batch_size
         && params.n_inputs < std::pow(2, 16)   // -c   C input_channels
         && params.n_outputs < std::pow(2, 16)  // -k   K output_channels
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
        const auto& config = params; // alias
        // cppcheck-suppress unreadVariable
        buff_info ibuf(MemLayout::NCHW,
                       config.batch_sz,
                       config.n_inputs,
                       config.in_height,
                       config.in_width,
                       1,
                       GetTypeSize(config.in_data_type));
        // cppcheck-suppress unreadVariable
        buff_info obuf(MemLayout::NCHW,
                       config.batch_sz,
                       config.n_outputs,
                       config.out_height,
                       config.out_width,
                       1,
                       GetTypeSize(config.out_data_type));

        const int eurictic_init_min_n_mult     = 1;
        const int eurictic_init_max_chunk_size = 16;
        const int n_miss = eurictic_init_min_n_mult * (64 / eurictic_init_max_chunk_size) - 1;

        if((static_cast<long>(config.n_inputs) + n_miss) * ibuf.byte_stride.nk >= (1LL << 31) ||
           (static_cast<long>(config.n_outputs) + n_miss) * obuf.byte_stride.nk >= (1LL << 31))
            ok = false;
    }
    return ok;
}

ConvSolution ConvAsm1x1UV2::GetSolution(const ConvolutionContext& params,
                                        const PerformanceConfigConvAsm1x1UV2& config) const
{
    ConvSolution result;
    std::ostringstream options;

    result.workspace_sz = 0;

    int data_len                               = GetTypeSize(params.out_data_type);
    const PerformanceConfigConvAsm1x1UV2* pcfg = &config;

    PerformanceConfigConvAsm1x1UV2 fromEnv;
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
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
    }

    // cppcheck-suppress unreadVariable
    const config_helper uv_lj(params, *pcfg);

    GenerateClangDefsym(options, "stride_h", uv_lj.stride_h);
    GenerateClangDefsym(options, "stride_w", uv_lj.stride_w);

    GenerateClangDefsym(options, "idilation_h", uv_lj.dilation_h);
    GenerateClangDefsym(options, "idilation_w", uv_lj.dilation_w);

    GenerateClangDefsym(options, "img_h", params.in_height); // H
    GenerateClangDefsym(options, "img_w", params.in_width);  // W

    GenerateClangDefsym(options, "out_h", params.out_height); // H
    GenerateClangDefsym(options, "out_w", params.out_width);  // W

    // Note that params.n_outputs and params.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "batch_size", params.batch_sz);       // N
    GenerateClangDefsym(options, "input_channels", params.n_inputs);   // C
    GenerateClangDefsym(options, "output_channels", params.n_outputs); // K
    GenerateClangDefsym(options, "wei_h", params.kernel_size_h);       // R
    GenerateClangDefsym(options, "wei_w", params.kernel_size_w);       // S
    GenerateClangDefsym(options, "pad_h", params.pad_h);
    GenerateClangDefsym(options, "pad_w", params.pad_w);
    GenerateClangDefsym(options, "weights_layout", params.direction.IsForward() ? 0 : 1);

    GenerateClangDefsym(options, "vec_c_in", 1);
    GenerateClangDefsym(options, "vec_k_out", 1);
    GenerateClangDefsym(options, "vec_c_filter", 1);

    GenerateClangDefsym(options, "acc_type", 1);
    GenerateClangDefsym(options, "buf_type", (data_len == 2 ? 2 : 1));

    // cppcheck-suppress unreadVariable
    buff_info ibuf(MemLayout::NCHW,
                   params.batch_sz,
                   params.n_inputs,
                   params.in_height,
                   params.in_width,
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info obuf(MemLayout::NCHW,
                   params.batch_sz,
                   params.n_outputs,
                   params.out_height,
                   params.out_width,
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info fbuf(params.direction.IsForward() ? MemLayout::NCHW : MemLayout::CNHW,
                   params.n_outputs,
                   params.n_inputs,
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

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", params.rmv.UseV3() ? 5 : 4);

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
    kinfo.l_wk.push_back(64 * waves_in_group);
    kinfo.l_wk.push_back(1);
    kinfo.l_wk.push_back(1);

    kinfo.g_wk.clear(); // gridsize

    const int n_per_wave = pcfg->GetNMult() * pcfg->GetNPerGpr();
    const int k_per_wave = pcfg->GetKMult() * pcfg->GetWavesKInGroup();

    kinfo.g_wk.push_back(kinfo.l_wk[0] *
                         divide_round_plus_inf(uv_lj.in_strided_w, uv_lj.w_per_wave) *
                         divide_round_plus_inf(uv_lj.in_strided_h, uv_lj.h_per_wave));

    kinfo.g_wk.push_back(divide_round_plus_inf(params.n_outputs, k_per_wave));

    kinfo.g_wk.push_back(divide_round_plus_inf(params.batch_sz, n_per_wave));

    kinfo.kernel_file = "conv1x1u_stride2.s";
    kinfo.kernel_name = "miopenGcnAsmConv1x1U_stride2";

    result.construction_params.push_back(kinfo);

    {
        int N, C, H, W, K, n_groups;
        GetCompiledInParameters(params, &N, &C, &H, &W, &K, &n_groups);
        result.invoker_factory = conv::MakeGcnAsm1x1UInvokerFactory(N, C, H, W, K, n_groups);
    }

    return result;
}

PerformanceConfigConvAsm1x1UV2 ConvAsm1x1UV2::Search(const ConvolutionContext& context,
                                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

} // namespace solver
} // namespace miopen
