/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <tuple>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_ULTRA_RXS_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_ULTRA_RXS_F2X3_PERF_VALS)

#define MAX_CU_LIMIT 512

namespace miopen {
namespace solver {

namespace {
// clang-format off
    auto PerfFieldRules()
    {
        return seq::MakeRuleSet(
            std::make_tuple(seq::Span<int, 1, MAX_CU_LIMIT>{}, &PerformanceConfigConvBinWinogradUltraRxSf2x3::n_groups)
        );
    }
// clang-format on

constexpr unsigned group_size = 64;
constexpr unsigned o_tile_W   = 2;
constexpr unsigned o_tile_H   = 2;
constexpr unsigned d_tile_W   = 4;
constexpr unsigned d_tile_H   = 4;

// step is alwas based on the output tile size
constexpr unsigned o_tile_step_W = o_tile_W;
constexpr unsigned o_tile_step_H = o_tile_H;
constexpr unsigned d_tile_step_W = o_tile_W;
constexpr unsigned d_tile_step_H = o_tile_H;

//
// Number of tile lanes (QWORDs for packed clip bits)
//
constexpr unsigned d_clip_tiles_QW = group_size * d_tile_W / (sizeof(uint64_t) * CHAR_BIT);
constexpr unsigned o_clip_tiles_QW = group_size * o_tile_W / (sizeof(uint64_t) * CHAR_BIT);

struct work_info
{
    int64_t d_load_offset_addr;
    int64_t o_store_offset_addr;
    uint64_t step_1_pos;
    uint64_t step_2_pos;
    uint64_t d_clip[d_clip_tiles_QW][d_tile_H];
    uint64_t o_clip[o_clip_tiles_QW][o_tile_H];
};

inline void WU_control_make_3x3_w_info(unsigned N,
                                       unsigned H,
                                       unsigned W,
                                       unsigned o_H,
                                       unsigned o_W,
                                       int pad_H,
                                       int pad_W,
                                       unsigned d_stride_N,
                                       unsigned d_stride_H,
                                       unsigned d_stride_W,
                                       unsigned o_stride_N,
                                       unsigned o_stride_H,
                                       unsigned o_stride_W,
                                       std::vector<work_info>& w_info)
{
    //
    // We assume the filter position is controlled by the LEFT pads and output sizes only here
    //
    // If the output size needed to be conputed based on the input size, filter size and left/right
    // pads, it is supposed to be done somewhere outside
    //

    int64_t o_cur_w = 0;
    int64_t o_cur_h = 0;
    int64_t cur_n   = 0;
    int64_t n       = 0;

    while((o_cur_w < o_W) && (o_cur_h < o_H) && (cur_n < N))
    {

        work_info cur_w_i;
        int64_t d_cur_w = o_cur_w - pad_W;
        int64_t d_cur_h = o_cur_h - pad_H;

        cur_w_i.d_load_offset_addr =
            d_cur_w * d_stride_W + d_cur_h * d_stride_H + cur_n * d_stride_N;
        cur_w_i.o_store_offset_addr =
            o_cur_w * o_stride_W + o_cur_h * o_stride_H + cur_n * o_stride_N;

        for(unsigned n_tile = 0; n_tile < group_size; n_tile++)
        {
            for(unsigned i = 0; i < d_tile_W; i++)
            {
                for(unsigned j = 0; j < d_tile_H; j++)
                {
                    unsigned k = n_tile * d_tile_W / (sizeof(uint64_t) * CHAR_BIT);

                    cur_w_i.d_clip[k][j] <<= 1;
                    cur_w_i.d_clip[k][j] |= (d_cur_w + i < 0) || (W <= d_cur_w + i) ||
                                            (d_cur_h + j < 0) || (H <= d_cur_h + j) ||
                                            (cur_n < 0) || (N <= cur_n);
                }
            }
            for(unsigned i = 0; i < o_tile_W; i++)
            {
                for(unsigned j = 0; j < o_tile_H; j++)
                {
                    unsigned k = n_tile * o_tile_W / (sizeof(uint64_t) * CHAR_BIT);

                    cur_w_i.o_clip[k][j] <<= 1;
                    cur_w_i.o_clip[k][j] |= (o_cur_w + i < 0) || (o_W <= o_cur_w + i) ||
                                            (o_cur_h + j < 0) || (o_H <= o_cur_h + j) ||
                                            (cur_n < 0) || (N <= cur_n);
                }
            }

            d_cur_w += d_tile_step_W;
            o_cur_w += o_tile_step_W;
            cur_w_i.step_1_pos <<= 1;
            cur_w_i.step_2_pos <<= 1;

            if(o_W <= o_cur_w)
            {
                cur_w_i.step_1_pos |= 1;

                o_cur_w = 0;
                d_cur_w = o_cur_w - pad_W;

                o_cur_h += o_tile_step_H;
                d_cur_h += d_tile_step_H;
            }
            if(o_H <= o_cur_h)
            {
                cur_w_i.step_2_pos |= 1;

                o_cur_h = 0;
                d_cur_h = o_cur_h - pad_H;

                cur_n += 1;
            }
        }
        w_info.push_back(cur_w_i);
        n++;
    }
}

inline void WU_control_w_info_bit_encode(std::vector<work_info>& w_info,
                                         std::vector<uint32_t>& gpu_control)
{
    for(auto i = 0; i < w_info.size(); i++)
    {
        std::array<uint32_t, 64> block = {0};
        work_info w_i                  = w_info[i];

        for(auto j = 0; j < 32; j++)
        {
            uint64_t qword;
            bool bit_reverse;

            if(j == 0)
            {
                qword       = w_i.d_load_offset_addr;
                bit_reverse = 0;
            }
            else if(j == 1)
            {
                qword       = w_i.o_store_offset_addr;
                bit_reverse = 0;
            }
            else if(j == 2)
            {
                qword       = w_i.step_1_pos;
                bit_reverse = 1;
            }
            else if(j == 3)
            {
                qword       = w_i.step_2_pos;
                bit_reverse = 1;
            }
            else if(j >= 4 && j < 4 + d_clip_tiles_QW * d_tile_H)
            {
                unsigned k  = j - 4;
                qword       = w_i.d_clip[k / d_tile_H][k % d_tile_H];
                bit_reverse = 1;
            }
            else if(j >= 4 + d_clip_tiles_QW * d_tile_H &&
                    j < 4 + d_clip_tiles_QW * d_tile_H + o_clip_tiles_QW * o_tile_H)
            {
                unsigned k  = j - 4 - d_clip_tiles_QW * d_tile_H;
                qword       = w_i.o_clip[k / o_tile_H][k % o_tile_H];
                bit_reverse = 1;
            }
            else if(j == 24)
            {
                qword       = i;
                bit_reverse = 0;
            }
            else
            {
                qword       = 0;
                bit_reverse = 0;
            }

            for(auto k = 0; k < 64; k++)
            {
                auto idx = bit_reverse ? 63 - k : k;
                block[idx] <<= 1;
                block[idx] |= (qword & 1);
                qword >>= 1;
            }
        }

        gpu_control.insert(gpu_control.end(), block.begin(), block.end());
    }
}

inline void WU_control_make_3x3(unsigned N,
                                unsigned H,
                                unsigned W,
                                unsigned o_H,
                                unsigned o_W,
                                unsigned pad_H,
                                unsigned pad_W,
                                unsigned d_stride_N,
                                unsigned d_stride_H,
                                unsigned d_stride_W,
                                unsigned o_stride_N,
                                unsigned o_stride_H,
                                unsigned o_stride_W,
                                std::vector<uint32_t>& gpu_control,
                                unsigned n_groups,
                                unsigned intl_factor)
{
    std::vector<work_info> w_info;
    WU_control_make_3x3_w_info(N,
                               H,
                               W,
                               o_H,
                               o_W,
                               pad_H,
                               pad_W,
                               d_stride_N,
                               d_stride_H,
                               d_stride_W,
                               o_stride_N,
                               o_stride_H,
                               o_stride_W,
                               w_info);

    std::vector<work_info> w_info_intl;
    for(int i = 0; i < w_info.size(); i += intl_factor * n_groups)
        for(int k = 0; k < intl_factor; k++)
            for(int j = k; j < intl_factor * n_groups && i + j < w_info.size(); j += intl_factor)
                w_info_intl.push_back(w_info[i + j]);

    WU_control_w_info_bit_encode(w_info_intl, gpu_control);
}

#if MIOPEN_BACKEND_HIP
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
    if(params.direction.IsBackwardData())
    {
        if(!(0 <= params.GetBackwardPadW() && params.GetBackwardPadW() < std::pow(2, 16)))
            return false;
        if(!(0 <= params.GetBackwardPadH() && params.GetBackwardPadH() < std::pow(2, 16)))
            return false;
    }
    const auto grid_workgroup_count_x = params.GetStream().GetMaxHardwareComputeUnits();
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    constexpr auto ELEM_SZ    = static_cast<int64_t>(sizeof(half_float::half));
    constexpr auto D_W_PITCH  = ELEM_SZ * 1;
    constexpr auto O_W_PITCH  = ELEM_SZ * 1;
    const auto D_H_PITCH      = D_W_PITCH * W;
    const auto O_H_PITCH      = O_W_PITCH * OW;
    const auto D_C_PITCH      = D_H_PITCH * H;
    const auto O_K_PITCH      = O_H_PITCH * OH;
    const auto D_N_PITCH      = D_C_PITCH * C;
    const auto O_N_PITCH      = O_K_PITCH * K;
    const auto TILES_N_ROW    = (OW + o_tile_step_W - 1) / o_tile_step_W;
    const auto TILES_N_COLUMN = (OH + o_tile_step_H - 1) / o_tile_step_H;

    const auto D_STEP_1_PITCH = d_tile_step_H * D_H_PITCH - TILES_N_ROW * d_tile_step_W * D_W_PITCH;
    const auto O_STEP_1_PITCH = o_tile_step_H * O_H_PITCH - TILES_N_ROW * o_tile_step_W * O_W_PITCH;
    const auto D_STEP_2_PITCH = D_N_PITCH - TILES_N_COLUMN * d_tile_step_H * D_H_PITCH;
    const auto O_STEP_2_PITCH = O_N_PITCH - TILES_N_COLUMN * o_tile_step_H * O_H_PITCH;

    // clang-format off
    return N == 1
        && C <= 240
        && K <= 16
        && S <= 3
        && R <= 3
        && D_H_PITCH < std::pow(2, 16)
        && O_H_PITCH < std::pow(2, 16)
        && D_C_PITCH < std::pow(2, 30)
        && O_K_PITCH < std::pow(2, 30)
        && D_STEP_1_PITCH < std::pow(2, 18)
        && O_STEP_1_PITCH < std::pow(2, 18)
        && D_STEP_2_PITCH < std::pow(2, 30)
        && O_STEP_2_PITCH < std::pow(2, 30)
        && grid_workgroup_count_x < std::pow(2, 16);
    // clang-format on
}
#endif

} // namespace

PerformanceConfigConvBinWinogradUltraRxSf2x3::PerformanceConfigConvBinWinogradUltraRxSf2x3(
    int n_groups_, int intl_factor_)
    : n_groups(n_groups_), intl_factor(intl_factor_)
{
}

void PerformanceConfigConvBinWinogradUltraRxSf2x3::HeuristicInit(const ConvolutionContext& config)
{
    // TODO: Find best heuristics
    n_groups    = config.GetStream().GetMaxHardwareComputeUnits();
    intl_factor = 1;
}

bool PerformanceConfigConvBinWinogradUltraRxSf2x3::SetNextValue(
    const ConvolutionContext& /*config*/)
{
    return !PerfFieldRules().Next(*this);
}

bool PerformanceConfigConvBinWinogradUltraRxSf2x3::IsValidValue() const
{
    return PerfFieldRules().IsIn(*this);
}

bool PerformanceConfigConvBinWinogradUltraRxSf2x3::IsValid(const ConvolutionContext& config) const
{
    if(config.GetStream().GetMaxHardwareComputeUnits() < n_groups)
        return false;

    if(!IsValidValue())
        return false;
    return true;
}

inline bool PerformanceConfigConvBinWinogradUltraRxSf2x3::operator==(
    const PerformanceConfigConvBinWinogradUltraRxSf2x3& other) const
{
    return n_groups == other.n_groups;
}

std::string PerformanceConfigConvBinWinogradUltraRxSf2x3::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvBinWinogradUltraRxSf2x3
ConvBinWinogradUltraRxSf2x3::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvBinWinogradUltraRxSf2x3 pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvBinWinogradUltraRxSf2x3::IsValidPerformanceConfig(
    const ConvolutionContext& problem, const PerformanceConfigConvBinWinogradUltraRxSf2x3& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

PerformanceConfigConvBinWinogradUltraRxSf2x3
ConvBinWinogradUltraRxSf2x3::Search(const ConvolutionContext& context,
                                    const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

class ShaderModel : public UnifiedDescriptionConv2d
{
    const size_t DATATYPE_BITS;    // S
    const size_t n_groups;         // BQ ~compute units
    const bool out_of_model_scope; // Shader model produces unreliable results.

    public:
    ShaderModel(const ConvolutionContext& ctx)
        : UnifiedDescriptionConv2d(ctx),
          DATATYPE_BITS(ctx.IsFp16() ? 16 : 32),
          n_groups(ctx.GetStream()
                       .GetMaxHardwareComputeUnits()), /// \todo Take n_groups from PerfConfig.
          out_of_model_scope(true)
    {
    }

    double ComputeWti() const
    {
        if(out_of_model_scope)
            return -1.0; // Shader model produces unreliable results.

        // TODO: Define performance model
        const auto WTI_predicted = -1.0;
        return WTI_predicted;
    }
};

static float GetWtiBase(const ConvolutionContext& params)
{
    constexpr auto WTI_UNKNOWN = -2.0;
    const auto rv              = ShaderModel(params).ComputeWti();
    return rv < 0 ? WTI_UNKNOWN : rv;
}

float ConvBinWinogradUltraRxSf2x3::GetWti(const ConvolutionContext& params) const
{
    return GetWtiBase(params);
}

static bool IsApplicableBase(const ConvolutionContext& params)
{
#if MIOPEN_BACKEND_HIP
    if(!params.Is2d())
        return false;
    if(!params.IsFp16())
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;

    const auto name = params.GetStream().GetDeviceName();
    if(!StartsWith(name, "gfx10"))
        return false;

    // clang-format off
    if (!( params.kernel_stride_w == 1
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0
        && params.group_counts == 1
        && params.in_layout == "NCHW"))
        return false;
    // clang-format on

    const auto n_inputs_per_group  = params.n_inputs / params.group_counts,
               n_outputs_per_group = params.n_outputs / params.group_counts;

    if(!params.direction.IsBackwardWrW())
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
    else
    {
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
#else
    (void)params;
    return false;
#endif
}

bool ConvBinWinogradUltraRxSf2x3::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_ULTRA_RXS_F2X3{}))
        return false;
    return IsApplicableBase(params);
}

ConvSolution
ConvBinWinogradUltraRxSf2x3::GetSolution(const ConvolutionContext& params,
                                         const PerformanceConfigConvBinWinogradUltraRxSf2x3& config,
                                         const bool disableConfigOverrideFromEnv) const
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool IsWarned;
    if(!IsWarned)
    {
        if(params.GetStream().GetMaxHardwareComputeUnits() > MAX_CU_LIMIT)
            MIOPEN_LOG_WE(SolverDbId(*this)
                          << ": GPU has " << params.GetStream().GetMaxHardwareComputeUnits()
                          << "CUs, but this solver supports max " << MAX_CU_LIMIT
                          << "and thus may show sub-optimal performance.");
        IsWarned = true;
    }

    const PerformanceConfigConvBinWinogradUltraRxSf2x3* pcfg = &config;
    PerformanceConfigConvBinWinogradUltraRxSf2x3 fromEnv;

    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz =
            miopen::GetStringEnv(MIOPEN_DEBUG_AMD_WINOGRAD_ULTRA_RXS_F2X3_PERF_VALS{});
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

    const auto n_groups    = pcfg->GetNGroups();
    const auto intl_factor = pcfg->GetInterleaveFactor();
    const auto group_cnt   = params.group_counts;

    int N, C, H, W, K, out_H, out_W, R, S, pad_H, pad_W, flags;
    BuffInfo d_buf, o_buf, f_buf;

    int unused               = 0;
    uint64_t reserved_offset = 0;
    int* reserved_ptr        = nullptr;
    float relu_alpha         = 1.0;

    if(!params.direction.IsBackwardWrW())
    {
        const auto is_forward = params.direction.IsForward();

        constexpr unsigned F_REVERSE_R = 1 << 0;
        constexpr unsigned F_REVERSE_S = 1 << 1;
        constexpr unsigned F_FLIP_K_C  = 1 << 2;

        GetCompiledInParameters(
            params, &N, &C, &H, &W, &K, &unused, &out_H, &out_W, &R, &S, &pad_H, &pad_W);

        flags = is_forward ? 0 : F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
        C     = C / group_cnt;
        K     = K / group_cnt;

        // cppcheck-suppress unreadVariable
        d_buf = BuffInfo(GetGroupConvLayout(GetMemLayout_t(params.in_layout), true),
                         N,
                         C,
                         H,
                         W,
                         group_cnt,
                         GetTypeSize(params.in_data_type));
        // cppcheck-suppress unreadVariable
        o_buf = BuffInfo(GetGroupConvLayout(GetMemLayout_t(params.out_layout), true),
                         N,
                         K,
                         out_H,
                         out_W,
                         group_cnt,
                         GetTypeSize(params.out_data_type));
        // cppcheck-suppress unreadVariable
        f_buf = BuffInfo(GetGroupConvLayout(is_forward ? (MemLayout_t::NCHW)
                                                       : GetSwappedNCLayout(MemLayout_t::NCHW),
                                            false),
                         K,
                         C,
                         R,
                         S,
                         group_cnt,
                         GetTypeSize(params.weights_data_type));
    }
    else
    {
        GetCompiledInParameters(
            params, &C, &K, &R, &S, &N, &unused, &H, &W, &out_H, &out_W, &unused, &unused);

        flags = 0;
        N     = N / group_cnt;
        K     = K / group_cnt;
        pad_H = params.conv_problem.GetConv().GetConvPads()[0];
        pad_W = params.conv_problem.GetConv().GetConvPads()[1];

        d_buf =
            BuffInfo(GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(params.in_layout)), true),
                     N,
                     C,
                     H,
                     W,
                     group_cnt,
                     GetTypeSize(params.in_data_type));
        o_buf = BuffInfo(
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(params.out_layout)), false),
            N,
            K,
            out_H,
            out_W,
            group_cnt,
            GetTypeSize(params.out_data_type));
        f_buf = BuffInfo(GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true),
                         K,
                         C,
                         R,
                         S,
                         group_cnt,
                         GetTypeSize(params.weights_data_type));
    }

    const unsigned tiles_n_row    = (out_W + o_tile_step_W - 1) / o_tile_step_W;
    const unsigned tiles_n_column = (out_H + o_tile_step_H - 1) / o_tile_step_H;

    const unsigned d_N_pitch = d_buf.byte_stride.nk;
    const unsigned d_C_pitch = d_buf.byte_stride.c;
    const unsigned d_H_pitch = d_buf.byte_stride.h;
    const unsigned d_W_pitch = d_buf.byte_stride.w;

    const int d_step_1_pitch = d_tile_step_H * d_H_pitch - tiles_n_row * d_tile_step_W * d_W_pitch;
    const int d_step_2_pitch = d_N_pitch - tiles_n_column * d_tile_step_H * d_H_pitch;

    const unsigned o_N_pitch = o_buf.byte_stride.nk;
    const unsigned o_K_pitch = o_buf.byte_stride.c;
    const unsigned o_H_pitch = o_buf.byte_stride.h;
    const unsigned o_W_pitch = o_buf.byte_stride.w;

    const int o_step_1_pitch = o_tile_step_H * o_H_pitch - tiles_n_row * o_tile_step_W * o_W_pitch;
    const int o_step_2_pitch = o_N_pitch - tiles_n_column * o_tile_step_H * o_H_pitch;

    std::vector<uint32_t> gpu_control;
    WU_control_make_3x3(N,
                        H,
                        W,
                        out_H,
                        out_W,
                        pad_H,
                        pad_W,
                        d_N_pitch,
                        d_H_pitch,
                        d_W_pitch,
                        o_N_pitch,
                        o_H_pitch,
                        o_W_pitch,
                        gpu_control,
                        n_groups,
                        intl_factor);

    const unsigned n_works      = gpu_control.size() / 64;
    const size_t control_buf_sz = gpu_control.size() * sizeof(decltype(gpu_control)::value_type);

    const size_t wg_size = 256;

    KernelInfo kernel;

    kernel.g_wk.push_back(wg_size * n_groups * params.group_counts);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    std::string kernel_name    = "miopenSp3AsmConv_Ultra_v1_1_3_gfx10";
    std::string kernel_file    = "Conv_Winograd_Ultra_v1_1_3";
    std::string kernel_postfix = "_fp16_pk_stride1";

    kernel.kernel_name = kernel_name + kernel_postfix;
    kernel.kernel_file = kernel_file + kernel_postfix + ".s";

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
        {"control_buf_sz", control_buf_sz},

        // Control buffer is part of kernel and depends on the following parameters.
        // The following unused compile options ensure the uniqueness of the control buffer
        // and the correct program caching.
        {"hash_N", N},
        {"hash_H", H},
        {"hash_W", W},
        {"hash_out_H", out_H},
        {"hash_out_W", out_W},
        {"hash_pad_H", pad_H},
        {"hash_pad_W", pad_W},
        {"hash_n_groups", n_groups},
        {"hash_intl_factor", intl_factor},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
    kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");

    ConvSolution solution;

    solution.construction_params.push_back(kernel);
    solution.invoker_factory = [=](std::vector<Kernel> kernels) {
        void* control_buf_ptr = nullptr;

#if MIOPEN_BACKEND_HIP
        const auto kern = kernels.front();
        const auto mod  = kern.program.GetModule();

        size_t dev_buf_sz;
        hipDeviceptr_t dev_buf_ptr;
        auto status = hipModuleGetGlobal(&dev_buf_ptr, &dev_buf_sz, mod, "control_buf");
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "Failed hip to get control_buf info");

        if(dev_buf_sz < control_buf_sz)
            MIOPEN_THROW("Invalid buffer size");

        params.GetStream().Copy(
            static_cast<const void*>(gpu_control.data()), dev_buf_ptr, control_buf_sz);

        control_buf_ptr = dev_buf_ptr;
#endif

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto k = handle.Run(kernels.front());
            ConstData_t in, wei, out;

            if(!params.direction.IsBackwardWrW())
            {
                const auto& invoke_params = primitive_params.CastTo<conv::DataInvokeParams>();
                const auto& tensors       = invoke_params.tensors;
                in                        = tensors.in;
                wei                       = tensors.w;
                out                       = tensors.out;
            }
            else
            {
                const auto& invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();
                const auto& tensors       = invoke_params.tensors;
                in                        = tensors.x;
                wei                       = tensors.dy;
                out                       = tensors.dw;
            }

            k(C,
              K,
              n_groups,
              n_works,
              d_C_pitch,
              d_H_pitch,
              d_step_1_pitch,
              d_step_2_pitch,
              o_K_pitch,
              o_H_pitch,
              o_step_1_pitch,
              o_step_2_pitch,
              in,
              out,
              control_buf_ptr,
              wei,
              reserved_ptr, // Unused bias_addr.
              relu_alpha,
              flags,
              R,
              S,
              reserved_offset,
              reserved_offset,
              reserved_offset,
              reserved_offset,
              reserved_offset);
        };
    };

    return solution;
}

} // namespace solver
} // namespace miopen
