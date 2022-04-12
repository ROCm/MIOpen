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

#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/sequences.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>

#include <cstdint>
#include <sstream>
#include <limits>
#include <cassert>
#include <tuple>

#define WORKAROUND_ISSUE_1146 1 // check asm solver applicability for gfx90a

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U)

namespace miopen {
namespace solver {

namespace {
// clang-format off
auto PerfFieldRules()
{
    return seq::MakeRuleSet(
        std::make_tuple(seq::Span<int, 0, 9>{}, &PerformanceConfigConvAsm3x3U::limit_wave_cnt),
        std::make_tuple(seq::Span<int, 1, 8>{}, &PerformanceConfigConvAsm3x3U::filters_per_wave),
        std::make_tuple(seq::Span<int, 1, 8>{}, &PerformanceConfigConvAsm3x3U::output_lines_per_wave)
    );
}
// clang-format on

} // namespace

bool PerformanceConfigConvAsm3x3U::SetNextValue(const ConvolutionContext& /*config*/)
{
    return !PerfFieldRules().Next(*this);
}

PerformanceConfigConvAsm3x3U::PerformanceConfigConvAsm3x3U(int lwc, int fpw, int olpw)
    : limit_wave_cnt(lwc), filters_per_wave(fpw), output_lines_per_wave(olpw)
{
}

inline bool
PerformanceConfigConvAsm3x3U::operator==(const PerformanceConfigConvAsm3x3U& other) const
{
    return PerfFieldRules().Compare(*this, other);
}

bool PerformanceConfigConvAsm3x3U::IsValidValue() const { return PerfFieldRules().IsIn(*this); }

bool PerformanceConfigConvAsm3x3U::IsValid(const ConvolutionContext& config) const
{
    if(!IsValidValue())
        return false;
    // to-do add support of uneven_outputs into grouped conv
    bool uneven_outputs = (config.n_outputs % filters_per_wave) != 0;
    auto num_wavefronts = config.n_outputs / filters_per_wave;
    if(config.group_counts > 1 && (uneven_outputs || (num_wavefronts % config.group_counts != 0)))
        return false;

    // Count the number of VGPRs required.
    const auto& img_width  = config.in_width;
    const auto& img_height = config.in_height;
    int n                  = 0;

    const bool enable_zero_line_padding_on_read = (img_height != output_lines_per_wave);
    if(enable_zero_line_padding_on_read)
        ++n;

    const int img_x_blocks = img_width;
    const int w64_chunks   = (img_x_blocks + 63) / 64;
    assert(w64_chunks != 0);
    if(w64_chunks == 0)
        return false;
    const int active_lanes = (img_x_blocks + w64_chunks - 1) / w64_chunks;
    assert(active_lanes != 0);
    if(active_lanes == 0)
        return false;
    const bool uneven_line_read_mode = (img_x_blocks % active_lanes != 0);
    if(uneven_line_read_mode)
        ++n;

    const int block_size_x        = 1;
    const int gprs_per_input_line = (img_x_blocks * block_size_x + active_lanes - 1) / active_lanes;
    const int input_lines_per_wave =
        (img_height == output_lines_per_wave) ? output_lines_per_wave : (output_lines_per_wave + 2);

    const int k_group_size                  = config.n_outputs / config.group_counts;
    const bool k_group_size_is_power_of_two = ((k_group_size & (k_group_size - 1)) == 0);
    n += (k_group_size_is_power_of_two || gprs_per_input_line * input_lines_per_wave >= 4)
             ? (gprs_per_input_line * input_lines_per_wave)
             : 4; // linesA
    n += (k_group_size_is_power_of_two || gprs_per_input_line * input_lines_per_wave >= 3)
             ? (gprs_per_input_line * input_lines_per_wave)
             : 3; // linesB

    // const bool enable_dpp_zero_column_padding = true;
    // if(enable_dpp_zero_column_padding)
    n += 2;

    const int acc_lines_per_wave = output_lines_per_wave;
    n += (gprs_per_input_line * filters_per_wave * acc_lines_per_wave);

    const int available_vgprs = 256;
    return n < available_vgprs;
}

void PerformanceConfigConvAsm3x3U::HeuristicInit(const ConvolutionContext& config)
{
    limit_wave_cnt        = 0;
    filters_per_wave      = 2;
    output_lines_per_wave = 2;

    if(config.n_outputs % (filters_per_wave * config.group_counts) != 0)
    {
        filters_per_wave = 1;
    }

    MIOPEN_LOG_I(ToString());
}

std::string PerformanceConfigConvAsm3x3U::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigConvAsm3x3U
ConvAsm3x3U::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvAsm3x3U pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsm3x3U::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                           const PerformanceConfigConvAsm3x3U& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

bool ConvAsm3x3U::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.Is2d())
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!(params.direction.IsForward() || params.direction.IsBackwardData()))
        return false;
    if(!params.rmv.IsV2orV3())
        return false;

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx8") || StartsWith(name, "gfx9")))
        return false;
#if WORKAROUND_ISSUE_1146
    if(name == "gfx90a")
        return false;
#endif
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    constexpr auto GIB                         = static_cast<int64_t>(1024) * 1024 * 1024;
    constexpr auto TIB                         = GIB * 1024;
    constexpr auto ELEM_SZ                     = static_cast<int64_t>(sizeof(float));
    constexpr int64_t SHADER_FEATURE_INDEX_MAX = static_cast<uint32_t>(-1);
    const auto IN_FEATURE_COUNT  = static_cast<int64_t>(params.batch_sz) * params.n_inputs;
    const auto OUT_FEATURE_COUNT = static_cast<int64_t>(params.batch_sz) * params.n_outputs;
    const auto IN_IMG_SZ         = ELEM_SZ * params.in_height * params.in_width;
    const auto OUT_IMG_SZ        = ELEM_SZ * params.out_height * params.out_width;
    const auto IN_BUF_SZ         = IN_IMG_SZ * IN_FEATURE_COUNT;
    const auto OUT_BUF_SZ        = OUT_IMG_SZ * OUT_FEATURE_COUNT;
    const auto WEI_BUF_SZ =
        ELEM_SZ * params.n_inputs * params.n_outputs * params.kernel_size_h * params.kernel_size_w;
    // clang-format off
    return params.pad_w == 1
        && params.pad_h == 1
        && params.kernel_stride_w == 1
        && params.kernel_stride_h == 1
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.kernel_size_w == 3
        && params.kernel_size_h == 3
        && params.n_inputs > 0
        && (params.n_inputs / params.group_counts) % 4 == 0 /// \todo: remove restriction that (n_inputs/group_counts) must be multiple of 4
        && params.in_width > 3
        && params.in_width <= 1000
        && IN_IMG_SZ  <= GIB
        && OUT_IMG_SZ <= 4 * GIB
        && IN_FEATURE_COUNT  - 1 <= SHADER_FEATURE_INDEX_MAX
        && OUT_FEATURE_COUNT - 1 <= SHADER_FEATURE_INDEX_MAX
        && IN_BUF_SZ  <= 256 * TIB
        && OUT_BUF_SZ <= 256 * TIB
        && WEI_BUF_SZ <= 4 * GIB
        && params.IsFp32()
        && params.in_layout == "NCHW";
        // && (params.forward ? params.weights_layout == "KCHW" : params.weights_layout == "CKHW" )
    // clang-format on
}

ConvSolution ConvAsm3x3U::GetSolution(const ConvolutionContext& params,
                                      const PerformanceConfigConvAsm3x3U& config,
                                      const bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;
    // Perf tune:
    const PerformanceConfigConvAsm3x3U* pcfg = &config;
    PerformanceConfigConvAsm3x3U fromEnv;
    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(params))
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U_PERF_VALS: "
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

    const int k_group_size                  = params.n_outputs / params.group_counts;
    const bool k_group_size_is_power_of_two = ((k_group_size & (k_group_size - 1)) == 0);

    const auto w64_chunks   = (params.in_width + 63) / 64;
    const auto active_lanes = (params.in_width + w64_chunks - 1) / w64_chunks;

    KernelBuildParameters options{
        {"batch_size", params.batch_sz},
        {"img_width", params.in_width},
        {"img_height", params.in_height},
        {"input_channels", params.n_inputs},
        {"output_channels", params.n_outputs},
        {"weights_layout", params.direction.IsForward() ? 0 : 1},
        {"reverse_weights", params.direction.IsForward() ? 0 : 1},
        {"ROCM_METADATA_VERSION", params.rmv.UseV3() ? 5 : 4},
        {"limit_wave_cnt", pcfg->limit_wave_cnt},
        {"filters_per_wave", pcfg->filters_per_wave},
        {"output_lines_per_wave", pcfg->output_lines_per_wave},
        // Debugging:
        {"group_counts", params.group_counts},
        {"k_group_size_is_power_of_two", k_group_size_is_power_of_two},
        {"workgroup_size_x", active_lanes},
    };

    KernelInfo construction_params;
    construction_params.comp_options = options.GenerateFor(kbp::GcnAsm{});

    construction_params.l_wk.push_back(active_lanes);
    construction_params.l_wk.push_back(1);
    construction_params.l_wk.push_back(1);

    construction_params.g_wk.push_back(
        active_lanes * ((params.n_outputs + pcfg->filters_per_wave - 1) / pcfg->filters_per_wave));
    construction_params.g_wk.push_back((params.in_height + pcfg->output_lines_per_wave - 1) /
                                       pcfg->output_lines_per_wave);
    construction_params.g_wk.push_back(params.batch_sz);

    construction_params.kernel_file = "conv3x3.s";
    construction_params.kernel_name = "miopenGcnAsmConv3x3U";

    result.construction_params.push_back(construction_params);
    result.invoker_factory = &conv::MakeGenericXWYPadInvoker;

    return result;
}

PerformanceConfigConvAsm3x3U ConvAsm3x3U::Search(const ConvolutionContext& context,
                                                 const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

} // namespace solver
} // namespace miopen
