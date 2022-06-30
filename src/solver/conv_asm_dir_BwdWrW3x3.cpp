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
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>

#define WORKAROUND_ISSUE_532 1 // ConvAsmBwdWrW3x3 has precision issues with some PerformanceConfigs
#define MIOPEN_GCN_ASM_DIRECT_3X3WRW_SEARCH_LWC_FIXED 0
#define WORKAROUND_SWDEV_330460 1 // ConvAsmBwdWrw3x3 has precision issues on MI200

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3_PERF_VALS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3)

namespace miopen {
namespace solver {

inline static bool Inc_1_2_4_8(int& v)
{
    assert(v == 1 || v == 2 || v == 4 || v == 8);
    if(v == 8)
    {
        v = 1;
        return true;
    }
    v = v * 2;
    return false;
}

inline static bool Is_1_2_4_8(const int& v) { return v == 1 || v == 2 || v == 4 || v == 8; }

bool PerformanceConfigAsmDirect3x3WrW::SetNextValue(const ConvolutionContext& /*config*/)
{
    // Increment with wrap-around:
    do
    {
#if MIOPEN_GCN_ASM_DIRECT_3X3WRW_SEARCH_LWC_FIXED == 0
        if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3_SEARCH_OPTIMIZED{}))
        {
            // (0 <= limit_wave_cnt && limit_wave_cnt <= 9)
            if(++limit_wave_cnt <= 9)
                break;
        }
#endif
        limit_wave_cnt = 0;
        // (0 <= reverse_inout && reverse_inout <= 1)
        if(++reverse_inout <= 1)
            break;
        reverse_inout = 0;
        // (8 == chunk_size || 16 == chunk_size)
        if((chunk_size += 8) <= 16)
            break;
        chunk_size = 8;
        // (1 == k_per_wave || 2 == k_per_wave || 4 == k_per_wave || 8 == k_per_wave)
        if(!Inc_1_2_4_8(k_per_wave))
            break;
        // (1 <= pipe_lines_depth && pipe_lines_depth <= 16)
        if(++pipe_lines_depth <= 16)
            break;
        pipe_lines_depth = 1;
        // (1 <= n_per_group && n_per_group <= 8);
        if(++n_per_group <= 8)
            break;
        n_per_group = 1;
        // All the fields (components) of performance confic have wrapped around.
        return false;
    } while(false);
    return true;
}

PerformanceConfigAsmDirect3x3WrW::PerformanceConfigAsmDirect3x3WrW(
    int lwc, int rio, int csz, int kpw, int pld, int npg)
    : limit_wave_cnt(lwc),
      reverse_inout(rio),
      chunk_size(csz),
      k_per_wave(kpw),
      pipe_lines_depth(pld),
      n_per_group(npg)
{
}

inline bool
PerformanceConfigAsmDirect3x3WrW::operator==(const PerformanceConfigAsmDirect3x3WrW& other) const
{
    // clang-format off
    return limit_wave_cnt == other.limit_wave_cnt
        && reverse_inout == other.reverse_inout
        && chunk_size == other.chunk_size
        && k_per_wave == other.k_per_wave
        && pipe_lines_depth == other.pipe_lines_depth
        && n_per_group == other.n_per_group; // clang-format on
}

bool PerformanceConfigAsmDirect3x3WrW::IsValidValue() const
{
    // clang-format off
    return (0 <= limit_wave_cnt && limit_wave_cnt <= 9)
        && (0 <= reverse_inout && reverse_inout <= 1)
        && (8 == chunk_size || 16 == chunk_size)
        && Is_1_2_4_8(k_per_wave)
        && (1 <= pipe_lines_depth && pipe_lines_depth <= 16)
        && (1 <= n_per_group && n_per_group <= 8); // clang-format on
}

static bool IsReverseInOutAllowed(const ConvolutionContext& config)
{
    return config.kernel_stride_w == 1 && config.kernel_stride_h == 1;
}

inline int elements_in_dword(const ConvolutionContext& config) { return config.IsFp16() ? 2 : 1; }

bool PerformanceConfigAsmDirect3x3WrW::IsValid(const ConvolutionContext& config) const
{
    if(!IsValidValue())
        return false;
    assert(chunk_size != 0);
    if(reverse_inout == 0)
    {
        if((config.n_outputs % (GetCPerWave() * config.group_counts) != 0) ||
           (config.n_inputs % (GetKPerWave() * config.group_counts) != 0))
            return false;
    }
    else
    {
        if((config.n_outputs % (GetKPerWave() * config.group_counts) != 0) ||
           (config.n_inputs % (GetCPerWave() * config.group_counts) != 0))
            return false;
    }
    if((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        return false;
    if((reverse_inout != 0 ? config.n_inputs : config.n_outputs) % GetCPerWave() != 0)
        return false;
    if(!(chunk_size * k_per_wave <= 64))
        return false;
    if((reverse_inout != 0 ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        return false;
    if(!(n_per_group <= config.batch_sz))
        return false;
    if(!(1 <= pipe_lines_depth && pipe_lines_depth <= std::min(config.out_height, 16)))
        return false;
    if((reverse_inout != 0) && !IsReverseInOutAllowed(config))
        return false;
    {
        const int accums_cnt = (config.kernel_size_w * config.kernel_size_h * GetCPerWave() *
                                k_per_wave * chunk_size) /
                               64;
        assert(chunk_size);
        const int out_w_vec =
            (config.out_width + elements_in_dword(config) - 1) / elements_in_dword(config);
        int gprs_per_line_in = (out_w_vec + chunk_size - 1) / chunk_size;
        if(chunk_size != 16)
        {
            assert(chunk_size - config.pad_w);
            gprs_per_line_in =
                (out_w_vec + chunk_size - config.pad_w - 1) / (chunk_size - config.pad_w);
        }
        assert(config.kernel_stride_w);
        gprs_per_line_in += gprs_per_line_in % config.kernel_stride_w;
        const int gprs_per_line_out =
            (gprs_per_line_in > 1) ? gprs_per_line_in / config.kernel_stride_w : 1;

        const int lines_in           = pipe_lines_depth + config.kernel_size_h - 1;
        const int vgprs_for_lines_in = lines_in * elements_in_dword(config) * gprs_per_line_in;
        assert(config.kernel_stride_h);
        const int lines_out =
            (pipe_lines_depth + config.kernel_stride_h - 1) / config.kernel_stride_h;
        const int vgprs_for_lines_out = lines_out * elements_in_dword(config) * gprs_per_line_out;
        const int vgprs_for_division =
            (vgprs_for_lines_in >= 4 ? 0 : 4) + (vgprs_for_lines_out >= 3 ? 0 : 3);

        const int k_group_size = config.n_inputs /
                                 (reverse_inout != 0 ? GetCPerWave() : GetKPerWave()) /
                                 config.group_counts;
        const bool k_group_size_is_power_of_two = ((k_group_size & (k_group_size - 1)) == 0);
        const int vgprs = accums_cnt + vgprs_for_lines_in + vgprs_for_lines_out +
                          (k_group_size_is_power_of_two ? 0 : vgprs_for_division) + 6 +
                          (elements_in_dword(config) - 1);
        if(!(vgprs <= 256))
            return false;
        if(n_per_group > 4)
            if(!(vgprs <= 128))
                return false;
        if(limit_wave_cnt != 0 && limit_wave_cnt * 4 < n_per_group)
            return false;
        const auto lds_size = (n_per_group - 1) * 64 /*wavesize*/ * sizeof(float) * accums_cnt;
        if(!(lds_size <= 65536))
            return false;

        const int unroll_factor = pipe_lines_depth * (pipe_lines_depth + 2);
        const int steps         = std::max(0, config.out_height - 1 - pipe_lines_depth);
        assert(unroll_factor);
        const int loops        = pipe_lines_depth + unroll_factor + steps % unroll_factor + 1;
        const int m_instr      = 3 + (gprs_per_line_in + 3) / 4;
        const std::string name = config.GetStream().GetDeviceName();
        /// \todo parsing "gfx[0-9]+" and finding major/minor/stepping from handle. using this
        /// information here and in all similar places across other Solvers.
        const bool dot2_inst_avail = (name == "gfx906" || name == "gfx908");
        const bool dot2_emulate    = (!dot2_inst_avail) && (elements_in_dword(config) == 2);
        const int v_instr          = (k_per_wave * config.kernel_size_h * gprs_per_line_out *
                             config.kernel_size_w * 4 * (dot2_emulate ? 2 : 1)) /
                            3 * elements_in_dword(config);
        const int exch_instr = elements_in_dword(config) == 2 ? 3 * m_instr : 0;
        const int total =
            loops * (m_instr + v_instr + exch_instr) * elements_in_dword(config); // instructions
        if(total >= 32000) // Estimation, a bit smaller than 32K.
            return false;
    }
    return true;
}

void PerformanceConfigAsmDirect3x3WrW::HeuristicInit(const ConvolutionContext& config)
{
    limit_wave_cnt = 0;

    chunk_size = (config.out_width < 48) ? 8 : 16;
    if((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        chunk_size = 16; // Fixup for correctness

    reverse_inout = 0;
    if(IsReverseInOutAllowed(config) && ((config.n_outputs % 4 != 0) || (config.out_width < 8)))
        reverse_inout = 1;

    const auto c_k = config.n_outputs * config.n_inputs / config.group_counts; // C*K
    if(c_k < 256)
        k_per_wave = 1;
    else if(c_k < 16384)
        k_per_wave = 2;
    else // C*K >= 16k
        k_per_wave = ((chunk_size == 8) ? 2 : 4);
    while((reverse_inout != 0 ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        k_per_wave /= 2; // Fixup for correctness

    if(c_k <= 512)
        n_per_group = 8;
    else if(c_k <= 4096)
        n_per_group = 4;
    else if(c_k <= 8192)
        n_per_group = 2;
    else
        n_per_group = 1;
    if(n_per_group > config.batch_sz)
        n_per_group = config.batch_sz; // n_per_group should never be > batch size.
    if(config.out_width >= 256 &&
       n_per_group > 4) // when width >= 256, n_per_group should not be > 4.
        n_per_group = 4;

    pipe_lines_depth = (config.out_height <= 1) ? 1 : 2;
    if((config.out_height < 8) && (config.out_width < 64))
    {
        pipe_lines_depth = config.out_height; // Special case.
    }

    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        limit_wave_cnt   = 0;
        reverse_inout    = 0;
        chunk_size       = 16; // CPerWave() = 4;
        k_per_wave       = 1;
        pipe_lines_depth = 2;
        n_per_group      = 1;
        if(config.n_outputs % (4 * config.group_counts) != 0)
        {
            /// (1) If reverse is Off, then both (C % c_per_wave) and (K % k_per_wave) must be 0.
            /// Toggling reverse swaps C and K in the condition above.
            /// (2) From the other hand, IsApplicable() ensures that either C or K is evenly
            /// divisable by 4.
            /// (3) We just set k_per_wave=1, c_per_wave=4. Therefore, (1) always can be satisfied
            /// here. If (C % c_per_wave) is not zero, just push reverse button so K and C will
            /// swap.
            ///
            /// \note C (input channels) resides in n_outputs, K (output channels) - in n_inputs,
            /// because that's how reverse convolutions are handled in MIOpen.
            reverse_inout = 1;
        }
        if(!IsValid(config))
        {
            MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init 2...");
            pipe_lines_depth = 1;
        }
        assert(IsValid(config));
    }
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceConfigAsmDirect3x3WrW::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConfigAsmDirect3x3WrW
ConvAsmBwdWrW3x3::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigAsmDirect3x3WrW pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsmBwdWrW3x3::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                                const PerformanceConfigAsmDirect3x3WrW& c) const
{
    return c.IsValidValue() && c.IsValid(problem);
}

bool ConvAsmBwdWrW3x3::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.Is2d())
        return false;
    if(!params.direction.IsBackwardWrW())
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!params.rmv.IsV2orV3())
        return false;

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx8") || StartsWith(name, "gfx9")))
        return false;
    if(!params.IsLayoutDefault())
    {
        return false;
    }
#if WORKAROUND_ISSUE_532
    if(StartsWith(name, "gfx9") && (params.kernel_stride_w > 1 || params.kernel_stride_h > 1))
        return false;
#endif

    if(name == "gfx90a" && params.conv_problem.IsGfx90aFp16altRequired())
        return false;

#if WORKAROUND_SWDEV_330460
    if(name == "gfx90a" && params.IsFp32())
        return false;
#endif

    // clang-format off
    bool ok = params.pad_w == 1           // -q  pad_w
        && params.pad_h == 1              // -p  pad_h
        && params.kernel_stride_w <= 2      // -v  stride_w
        && params.kernel_stride_h <= 2      // -u  stride_h
        && params.kernel_size_w == 3      // -x  S wei_w
        && params.kernel_size_h == 3      // -y  R wei_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0
        && (params.IsFp32() || params.IsFp16())
        && params.in_layout == "NCHW";
    if(!ok)
        return false; // Early exit to speed up the check.

    if(params.IsFp16()
          && (StartsWith(name, "gfx8") // Not supported.
             || params.batch_sz % 2 != 0)) /// \todo Initial version.
       return false;

    // Check limits:
    const auto h_w     = static_cast<long>(params.out_height) * params.out_width;
    const auto r_s     = static_cast<long>(params.kernel_size_h) * params.kernel_size_w;
    const auto c_h_w   = static_cast<long>(params.n_outputs) * h_w;   // C*H*W
    const auto k_h_w   = static_cast<long>(params.n_inputs) * h_w;    // K*H*W
    const auto c_r_s   = static_cast<long>(params.n_outputs) * r_s;   // C*R*S
    const auto k_r_s   = static_cast<long>(params.n_inputs) * r_s;    // K*R*S
    const auto n_c_h_w = static_cast<long>(params.batch_sz) * c_h_w;  // N*C*H*W
    const auto n_k_h_w = static_cast<long>(params.batch_sz) * k_h_w;  // N*K*H*W
    const auto c_k_r_s = static_cast<long>(params.n_outputs) * k_r_s; // C*K*R*S
    ok = params.out_width > 0
         && params.out_width <= 512
         && (IsReverseInOutAllowed(params)
                ? ((params.n_outputs % (4 * params.group_counts) == 0) || (params.n_inputs % (4 * params.group_counts) == 0))
                : (params.n_outputs % (4 * params.group_counts) == 0))
         && params.out_height < std::pow(2, 16) // -H   H img_h
         && params.batch_sz < std::pow(2, 16)   // -n   N batch_size
         && params.n_outputs < std::pow(2, 16)  // -c   C input_channels
         && params.n_inputs < std::pow(2, 16)   // -k   K output_channels
         && c_h_w < std::pow(2, 22)
         && k_h_w < std::pow(2, 22)
         && c_r_s < std::pow(2, 22)
         && k_r_s < std::pow(2, 22)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29); // clang-format on
    return ok;
}

ConvSolution ConvAsmBwdWrW3x3::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfigAsmDirect3x3WrW& config,
                                           const bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;
    std::ostringstream options;
    GenerateClangDefsym(options, "elements_in_dword", (params.IsFp16()) ? 2 : 1);
    GenerateClangDefsym(options, "batch_size", params.batch_sz); // N
    GenerateClangDefsym(options, "img_h", params.out_height);    // H
    GenerateClangDefsym(options, "img_w", params.out_width);     // W
    // Note that params.n_outputs and params.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "input_channels", params.n_outputs); // C
    GenerateClangDefsym(options, "output_channels", params.n_inputs); // K
    GenerateClangDefsym(options, "wei_h", params.kernel_size_h);      // R
    GenerateClangDefsym(options, "wei_w", params.kernel_size_w);      // S
    GenerateClangDefsym(options, "pad_h", params.pad_h);
    GenerateClangDefsym(options, "pad_w", params.pad_w);
    GenerateClangDefsym(options, "stride_h", params.kernel_stride_h);
    GenerateClangDefsym(options, "stride_w", params.kernel_stride_w);
    GenerateClangDefsym(options, "weights_layout", 0);
    GenerateClangDefsym(options, "reverse_weights", 0);
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", params.rmv.UseV3() ? 5 : 4);
    // Perf tune:
    const PerformanceConfigAsmDirect3x3WrW* pcfg = &config;
    PerformanceConfigAsmDirect3x3WrW fromEnv;
    if(!disableConfigOverrideFromEnv)
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(params))
                {
                    MIOPEN_LOG_E("MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3_PERF_VALS: "
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
    GenerateClangDefsym(options, "limit_wave_cnt", pcfg->GetLimitWaveCnt());
    GenerateClangDefsym(options, "chunk_size", pcfg->GetChunkSize());
    GenerateClangDefsym(options, "c_per_wave", pcfg->GetCPerWave());
    GenerateClangDefsym(options, "k_per_wave", pcfg->GetKPerWave());
    GenerateClangDefsym(options, "n_per_group", pcfg->GetNPerGroup());
    GenerateClangDefsym(options, "pipe_lines_depth", pcfg->GetPipeLinesDepth());
    GenerateClangDefsym(options, "reverse_inout", pcfg->GetReverseInout());
    // Debugging:
    GenerateClangDefsym(options, "enable_debug_output", 0);
    GenerateClangDefsym(options, "group_counts", params.group_counts);

    const int k_group_size =
        params.n_inputs / (pcfg->reverse_inout != 0 ? pcfg->GetCPerWave() : pcfg->GetKPerWave()) /
        params.group_counts;
    const bool k_group_size_is_power_of_two = ((k_group_size & (k_group_size - 1)) == 0);
    GenerateClangDefsym(options, "k_group_size_is_power_of_two", k_group_size_is_power_of_two);

    KernelInfo kernel;

    kernel.comp_options = options.str();

    kernel.l_wk.clear(); // workgroupsize
    kernel.l_wk.push_back(64 * pcfg->GetNPerGroup());
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.clear(); // gridsize
    kernel.g_wk.push_back(64 * pcfg->GetNPerGroup());

    if(pcfg->GetReverseInout() == 0)
    {
        kernel.g_wk.push_back(params.n_outputs / pcfg->GetCPerWave() / params.group_counts);
        kernel.g_wk.push_back(params.n_inputs / pcfg->GetKPerWave());
    }
    else
    {
        kernel.g_wk.push_back(params.n_outputs / pcfg->GetKPerWave() / params.group_counts);
        kernel.g_wk.push_back(params.n_inputs / pcfg->GetCPerWave());
    }

    kernel.kernel_file = "conv3x3wrw.s";
    kernel.kernel_name = "miopenGcnAsmConv3x3WrW";

    result.construction_params.push_back(kernel);
    result.workspace_sz = 0;

    int N, C, H, W, K, n_groups;
    GetCompiledInParameters(params, &N, &C, &H, &W, &K, &n_groups);

    result.invoker_factory = [N, C, H, W, K, n_groups](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto k              = handle.Run(kernels[0]);
            const auto& invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();
            int unused                = 0;
            int* return_addr          = nullptr;
            const auto& x             = invoke_params.tensors.x;
            const auto& dy            = invoke_params.tensors.dy;
            const auto& dw            = invoke_params.tensors.dw;
            k(N, C, H, W, K, n_groups, unused, unused, x, dw, dy, return_addr);
        };
    };

    return result;
}

PerformanceConfigAsmDirect3x3WrW ConvAsmBwdWrW3x3::Search(const ConvolutionContext& context,
                                                          const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

} // namespace solver
} // namespace miopen
