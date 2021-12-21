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
#include <miopen/solver.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U)

namespace miopen {
namespace solver {

static inline bool UseSubsample(const ConvolutionContext& c)
{
    return (c.kernel_stride_w > 1 || c.kernel_stride_h > 1) && c.direction.IsForward();
}

static inline bool UseUpsample(const ConvolutionContext& c)
{
    return (c.kernel_stride_w > 1 || c.kernel_stride_h > 1) && c.direction.IsBackwardData();
}

/// After 2x subsampling kernel, image size on asm kernel input becomes 4x (2*2) smaller.
/// As padding = 0, we can simply re-use output image size (no computations required).
/// \note For backward convolutions input image size is held in
/// out_height/out_width and vice versa.
static inline int AsmImgHeight(const ConvolutionContext& c)
{
    return UseSubsample(c) ? c.out_height : c.in_height;
}

static inline int AsmImgWidth(const ConvolutionContext& c)
{
    return UseSubsample(c) ? c.out_width : c.in_width;
}

/// \todo move to separate header and use in other solvers.
template <int L, int H>
inline static bool IsTwoPower(const int v)
{
    static_assert(L <= H, "L <= H");
    if(((v - 1) & v) != 0)
        return false;
    return L <= v && v <= H;
}

template <int L, int H>
inline static bool NextTwoPower(int& v)
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
inline static bool IsLinear(const int v)
{
    static_assert(L <= H, "L <= H");
    return L <= v && v <= H;
}

template <int L, int H>
inline static bool NextLinear(int& v)
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
inline static bool Is_1_4_8_12_to_32(const int& v)
{
    return v == 1 || (v % 4 == 0 && IsLinear<1, 8>(v / 4));
}

inline static bool Next_1_4_8_12_to_32(int& v)
{
    assert(Is_1_4_8_12_to_32(v));
    int tmp        = v / 4;
    const bool ret = NextLinear<0, 8>(tmp);
    v              = ret ? 1 : tmp * 4;
    return ret;
}

#ifndef NDEBUG
inline static bool Is_1_4(const int& v) { return v == 1 || v == 4; }
#endif

inline static bool Next_1_4(int& v)
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

bool PerformanceConfigConvAsm1x1U::SetNextValue(const ConvolutionContext& /*config*/)
{
    // Increment with wrap-around:
    do
    {
        if(!NextLinear<1, 4>(read_size))
            break;
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED{}))
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
    if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED{}))
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

inline bool
PerformanceConfigConvAsm1x1U::operator==(const PerformanceConfigConvAsm1x1U& other) const
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

bool PerformanceConfigConvAsm1x1U::IsValidValue() const
{
    // clang-format off
    return IsLinear<1,4>(read_size)
        && Is_1_4_8_12_to_32(k_mult)
        && IsLinear<1,16>(chunks_per_wave)
        && IsTwoPower<1,64>(chunk_size)
        && IsLinear<1,8>(n_mult)
        && IsTwoPower<1,32>(c_mult)
        && IsLinear<1,8>(waves_c_in_group)
        && IsTwoPower<1,8>(waves_k_in_group); // clang-format on
}

bool PerformanceConfigConvAsm1x1U::IsValid(const ConvolutionContext& config) const
{
    const auto elements_in_dword = 4 / GetTypeSize(config.in_data_type);

    if(!IsValidValue())
        return false;
    if(!(read_size * elements_in_dword <= chunks_per_wave))
        return false;
    if(!(waves_c_in_group <= config.n_inputs))
        return false;
    if(!(k_mult * waves_k_in_group <= config.n_outputs))
        return false;
    if(!(waves_c_in_group * waves_k_in_group <= 16))
        return false;
    if((c_mult % elements_in_dword) != 0)
        return false;
    if((k_mult % elements_in_dword) != 0)
        return false;
    if(chunks_per_wave % elements_in_dword != 0)
        return false;
    const auto in_gprs =
        (chunks_per_wave * n_mult * c_mult + elements_in_dword - 1) / elements_in_dword;
    const auto acc_gprs = chunks_per_wave * n_mult * k_mult;
    const auto img_hw   = config.out_height * config.out_width;
    // TODO last vgpr only for old card.
    // ADD if(option.machine_version_major == 9)
    // vgprs  = 4 + 2 * in_gprs + acc_gprs + (img_hw % elements_in_dword != 0 ? 1: 0);
    // else
    const auto vgprs = 4 + 2 * in_gprs + acc_gprs + (img_hw % elements_in_dword != 0 ? 1 : 0) + 1;
    if(!(vgprs < 256))
        return false;
    const auto max_waves_per_CU = (256 / vgprs) * 4;
    if(!(max_waves_per_CU >= waves_c_in_group * waves_k_in_group))
        return false;
    const auto sgprs = 25 + 2 * k_mult * c_mult;
    if(!(sgprs < 102)) /// \todo This is valid for Gfx8 and Gfx9. Check for newer parts.
        return false;
    const int total_n_blocks = (config.batch_sz + GetNPerGpr() - 1) / GetNPerGpr();
    if(!(n_mult <= total_n_blocks))
        return false;

    const int total_chunks = (img_hw + chunk_size - 1) / chunk_size;
    if(!(chunks_per_wave <= total_chunks))
        return false;

    const int c_per_wave      = (config.n_inputs + waves_c_in_group - 1) / waves_c_in_group;
    const int c_per_last_wave = config.n_inputs - (c_per_wave * (waves_c_in_group - 1));

    if(config.direction.IsBackwardData() && !(config.n_outputs % k_mult == 0))
        return false;
    return (c_per_wave % c_mult == 0) && (c_per_last_wave % c_mult == 0);
}

void PerformanceConfigConvAsm1x1U::HeuristicInit(const ConvolutionContext& config)
{
    if(config.in_data_type == miopenDouble)
        MIOPEN_THROW("Double data type is not supported by ConvAsm1x1U");

    const auto elements_in_dword = 4 / GetTypeSize(config.in_data_type);
    read_size                    = 4;
    k_mult                       = 16;
    chunks_per_wave              = static_cast<int>(read_size * elements_in_dword);
    chunk_size                   = 16;
    n_mult                       = 2;
    c_mult                       = elements_in_dword;
    waves_c_in_group             = 1;
    waves_k_in_group             = 1;

    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        read_size  = 1;
        k_mult     = 4;
        chunk_size = 1;
        n_mult     = 1;
    }
    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        k_mult = (elements_in_dword == 1) ? 1 : 4;
        c_mult = 2;
    }
    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        chunks_per_wave = 2;
        c_mult          = elements_in_dword;
    }
    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        chunks_per_wave = elements_in_dword;
    }
    if(!IsValid(config))
    {
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
    }
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceConfigConvAsm1x1U::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvAsm1x1U
ConvAsm1x1U::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvAsm1x1U pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsm1x1U::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                           const PerformanceConfigConvAsm1x1U& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

bool ConvAsm1x1U::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U{}))
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
    if(!(params.IsFp32() || params.IsFp16()))
        return false;

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx9") == std::string::npos)
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
    const int img_hw = params.out_height * params.out_width;
    bool ok = (params.pad_w == 0         // -q  pad_w
        && params.pad_h == 0             // -p  pad_h
        && params.kernel_stride_w <= 2   // -u  stride_w
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_size_w == 1     // -x  S wei_w
        && params.kernel_size_h == 1     // -y  R wei_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0
        && params.n_inputs % elements_in_dword == 0
        && params.n_outputs % elements_in_dword == 0
        && params.in_layout == "NCHW"
        && params.group_counts == 1
        && img_hw >= elements_in_dword
        && (elements_in_dword == 1 || params.n_outputs >= 4));
    if(params.direction.IsBackwardData() && elements_in_dword != 1)
        ok = ok && (params.n_outputs % 4 == 0);
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    /// \todo Ilya: The checks below look adequate but needs to be double-checked.
    {
        const long input_line_size = 4 * params.in_width;
        const long input_feature_map_size = input_line_size * params.in_height;
        const long input_stack_size = input_feature_map_size * params.n_inputs;
        if (! (input_stack_size < (1U << 24)))
            return false;
    }
    {
        const long output_line_size = 4 * params.out_width;
        const long output_feature_map_size = output_line_size * params.out_height;
        const long output_stack_size = output_feature_map_size * params.n_outputs;
        if (! (output_stack_size < (1U << 24)))
            return false;
    }
    // Check limits:
    auto h_w = static_cast<long>(AsmImgHeight(params)) * AsmImgWidth(params);
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
    return ok;
}

size_t ConvAsm1x1U::GetWorkspaceSize(const ConvolutionContext& params) const
{
    if(UseSubsample(params) || UseUpsample(params))
    {
        int in_batch_stride = AsmImgWidth(params) * AsmImgHeight(params) *
                              (UseSubsample(params) ? params.n_inputs : params.n_outputs);
        int data_len = GetTypeSize(params.out_data_type);
        return in_batch_stride * params.batch_sz * data_len;
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

ConvSolution ConvAsm1x1U::GetSolution(const ConvolutionContext& params,
                                      const PerformanceConfigConvAsm1x1U& config,
                                      const bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;

    std::ostringstream options;

    KernelInfo ss_us_kernel;
    int data_len = GetTypeSize(params.out_data_type);
    if(UseSubsample(params) || UseUpsample(params))
    {
        // subsampled input, in_height equals to image size after downsampling
        int in_batch_stride = AsmImgWidth(params) * AsmImgHeight(params) *
                              (UseSubsample(params) ? params.n_inputs : params.n_outputs);
        int write_unit =
            (AsmImgWidth(params) % 4 == 0)
                ? 4
                : (AsmImgWidth(params) % 3 == 0) ? 3 : (AsmImgWidth(params) % 2 == 0) ? 2 : 1;

        int n_grp0_size0 = 256;

        const auto subsample_kernel_compilation_options =
            std::string(" -DDATA_TYPE=") +
            (params.in_data_type == miopenHalf ? "ushort" : "float") +
            std::string(" -DMLO_GRP0_SZ0=") + std::to_string(n_grp0_size0) +
            std::string(" -DMLO_GRP0_SZ1=1 ") + std::string(" -DMLO_GRP0_SZ2=1 ") +
            std::string(" -DMLO_FILTER0_STRIDE0=") + std::to_string(params.kernel_stride_w) +
            std::string(" -DMLO_FILTER0_STRIDE1=") + std::to_string(params.kernel_stride_h) +
            std::string(" -DMLO_WRITE_UNIT=") + std::to_string(write_unit) +
            std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(params.out_channel_stride) +
            std::string(" -DMLO_OUT_STRIDE=") + std::to_string(params.out_stride) +
            std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(in_batch_stride) +
            std::string(" -DMLO_IN0_BATCH_STRIDE=") +
            std::to_string(params.direction.IsForward() ? params.in_batch_stride
                                                        : params.out_batch_stride) +
            std::string(" -DMLO_IN0_CHANNEL_STRIDE=") + std::to_string(params.in_channel_stride) +
            std::string(" -DMLO_IN0_STRIDE=") + std::to_string(params.in_stride) +
            params.general_compile_options;

        ss_us_kernel.l_wk.push_back(n_grp0_size0);
        ss_us_kernel.l_wk.push_back(1);
        ss_us_kernel.l_wk.push_back(1);
        // output is number of subsampled input maps
        size_t gbl_wk0 = (in_batch_stride / write_unit);
        size_t gbl_wk1 = params.batch_sz;
        size_t gbl_wk2 = 1;

        ss_us_kernel.g_wk.push_back(gbl_wk0);
        ss_us_kernel.g_wk.push_back(gbl_wk1);
        ss_us_kernel.g_wk.push_back(gbl_wk2);

        ss_us_kernel.kernel_file = "MIOpenUtilKernels3.cl";

        if(UseSubsample(params))
            ss_us_kernel.kernel_name = "SubSample";
        else
            ss_us_kernel.kernel_name = "UpSample";

        ss_us_kernel.comp_options = subsample_kernel_compilation_options;
    }
    result.workspce_sz = GetWorkspaceSize(params);

    GenerateClangDefsym(options, "stride_h", 1);
    GenerateClangDefsym(options, "stride_w", 1);
    GenerateClangDefsym(options, "img_h", AsmImgHeight(params)); // H
    GenerateClangDefsym(options, "img_w", AsmImgWidth(params));  // W

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

    // cppcheck-suppress unreadVariable
    buff_info ibuf(MemLayout::NCHW,
                   params.batch_sz,
                   params.n_inputs,
                   AsmImgHeight(params),
                   AsmImgWidth(params),
                   1,
                   data_len);
    // cppcheck-suppress unreadVariable
    buff_info obuf(MemLayout::NCHW,
                   params.batch_sz,
                   params.n_outputs,
                   AsmImgHeight(params),
                   AsmImgWidth(params),
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

    const PerformanceConfigConvAsm1x1U* pcfg = &config;
    PerformanceConfigConvAsm1x1U fromEnv;
    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
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
    main_kernel.l_wk.push_back(64 * waves_in_group);
    main_kernel.l_wk.push_back(1);
    main_kernel.l_wk.push_back(1);

    main_kernel.g_wk.clear(); // gridsize
    const int hw_per_wave = pcfg->GetChunksPerWave() * pcfg->GetChunkSize();

    main_kernel.g_wk.push_back(
        main_kernel.l_wk[0] *
        divide_round_plus_inf(AsmImgHeight(params) * AsmImgWidth(params), hw_per_wave));

    main_kernel.g_wk.push_back(
        divide_round_plus_inf(params.n_outputs, pcfg->GetKMult() * pcfg->GetWavesKInGroup()));
    const int n_images_per_wave = pcfg->GetNMult() * pcfg->GetNPerGpr();
    main_kernel.g_wk.push_back(divide_round_plus_inf(params.batch_sz, n_images_per_wave));

    main_kernel.kernel_file = "conv1x1u.s";
    main_kernel.kernel_name = "miopenGcnAsmConv1x1U";

    if(UseSubsample(params))
        result.construction_params.push_back(ss_us_kernel);

    result.construction_params.push_back(main_kernel);

    if(UseUpsample(params))
        result.construction_params.push_back(ss_us_kernel);

    if(UseSubsample(params))
    {
        int N, C, H, W, K, n_groups, out_H, out_W;
        GetCompiledInParameters(params, &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W);
        result.invoker_factory = conv::MakeGcnAsm1x1USSInvokerFactory(
            N, C, K, n_groups, out_H, out_W, result.workspce_sz);
    }
    else if(UseUpsample(params))
    {
        int N, C, H, W, K, n_groups;
        GetCompiledInParameters(params, &N, &C, &H, &W, &K, &n_groups);
        result.invoker_factory =
            conv::MakeGcnAsm1x1UUSInvokerFactory(N, C, K, n_groups, H, W, result.workspce_sz);
    }
    else
    {
        int N, C, H, W, K, n_groups;
        GetCompiledInParameters(params, &N, &C, &H, &W, &K, &n_groups);
        result.invoker_factory = conv::MakeGcnAsm1x1UInvokerFactory(N, C, H, W, K, n_groups);
    }

    return result;
}

PerformanceConfigConvAsm1x1U ConvAsm1x1U::Search(const ConvolutionContext& context,
                                                 const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

} // namespace solver
} // namespace miopen
