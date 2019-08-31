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

#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/kernel_build_params.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3U_PERF_VALS)

namespace miopen {
namespace solver {

bool PerformanceConfigConvAsm3x3U::SetNextValue()
{
    // Increment with wrap-around:
    do
    {
        if(++limit_wave_cnt <= 9) // [0..9]
            break;
        limit_wave_cnt = 0;
        if(++filters_per_wave <= 8) // [1..8]
            break;
        filters_per_wave = 1;
        if(++output_lines_per_wave <= 8) // [1..8]
            break;
        // All the fields (components) of performance confic have wrapped around.
        return false;
    } while(false);
    return true;
}

PerformanceConfigConvAsm3x3U::PerformanceConfigConvAsm3x3U(int lwc, int fpw, int olpw)
    : limit_wave_cnt(lwc), filters_per_wave(fpw), output_lines_per_wave(olpw)
{
}

inline bool PerformanceConfigConvAsm3x3U::
operator==(const PerformanceConfigConvAsm3x3U& other) const
{
    // clang-format off
    return limit_wave_cnt == other.limit_wave_cnt
        && filters_per_wave == other.filters_per_wave
        && output_lines_per_wave == other.output_lines_per_wave; // clang-format on
}

bool PerformanceConfigConvAsm3x3U::IsValidValue() const
{
    // clang-format off
    return (0 <= limit_wave_cnt && limit_wave_cnt <= 9)
        && (1 <= filters_per_wave && filters_per_wave <= 8)
        && (1 <= output_lines_per_wave && output_lines_per_wave <= 8); // clang-format on
}

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

void PerformanceConfigConvAsm3x3U::EuristicInit(const ConvolutionContext& config)
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
    pp.EuristicInit(params);
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
    if(!params.use_asm_kernels)
        return false;
    if(!params.Is2d())
        return false;
    if(params.rmv != rocm_meta_version::AMDHSA_1_0)
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx9") == std::string::npos)
    {
        return false;
    }
    assert(params.weights_layout.length() == 0); // FIXME _weights_layout is not supported yet.
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
        && params.IsFp32()
        && params.in_layout == "NCHW";
        // && (params.forward ? params.weights_layout == "KCHW" : params.weights_layout == "CKHW" )
    // clang-format on
}

bool ConvAsm3x3U::IsFast(const ConvolutionContext& params) const { return params.in_width >= 50; }

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
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3U_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(params))
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3U_PERF_VALS: "
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
        {"no_params_file", 1},
        {"ROCM_METADATA_VERSION", 4},
        {"limit_wave_cnt", pcfg->limit_wave_cnt},
        {"filters_per_wave", pcfg->filters_per_wave},
        {"output_lines_per_wave", pcfg->output_lines_per_wave},
        // Debugging:
        {"enable_debug_output", 0},
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
    construction_params.kernel_name = "gcnAsmConv3x3U";

    result.construction_params.push_back(construction_params);
    return result;
}

template <typename B, typename T>
int ConvAsm3x3U::RunAndMeasureSolution(miopen::Handle& profile_h,
                                       B bot_ocl_buf,
                                       T top_ocl_buf,
                                       ConstData_t wei_ocl_buf,
                                       ConstData_t bias_ocl_buf,
                                       const ConvolutionContext&,
                                       const ConvSolution& solution,
                                       float& elapsed_time) const
{
    assert(bias_ocl_buf == nullptr);
    (void)bias_ocl_buf;
    const KernelInfo k_info = solution.construction_params.back();
#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        // ConvolutionContext::general_compile_options is for OpenCL kernels
        // and thus not applicable for assembly.
        auto kernel = profile_h.AddKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);
        int padding_val = 0;
        kernel(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_val);
        elapsed_time = profile_h.GetKernelTime();
    }
#ifdef NDEBUG
    catch(miopen::Exception&)
    {
        return -1;
    }
#endif
    return 0;
}

PerformanceConfigConvAsm3x3U ConvAsm3x3U::Search(const ConvolutionContext& context) const
{
    if(context.direction.IsForward())
        return GenericSearchFwd(*this, context);
    else
        return GenericSearchBwd(*this, context);
}

} // namespace solver
} // namespace miopen
