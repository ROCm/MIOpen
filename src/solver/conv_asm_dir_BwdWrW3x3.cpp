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

bool PerformanceConfigAsmDirect3x3WrW::SetNextValue(const ProblemDescription&)
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
        // NOLINTNEXTLINE (bugprone-assignment-in-if-condition)
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

bool PerformanceConfigAsmDirect3x3WrW::operator==(
    const PerformanceConfigAsmDirect3x3WrW& other) const
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

static bool IsReverseInOutAllowed(const ProblemDescription& problem)
{
    return problem.GetKernelStrideW() == 1 && problem.GetKernelStrideH() == 1;
}

inline int elements_in_dword(const ProblemDescription& problem) { return problem.IsFp16() ? 2 : 1; }

bool PerformanceConfigAsmDirect3x3WrW::IsValid(const ConvolutionContext& ctx,
                                               const ProblemDescription& problem) const
{
    if(!IsValidValue())
        return false;
    assert(chunk_size != 0);
    if(reverse_inout == 0)
    {
        if((problem.GetOutChannels_() % (GetCPerWave() * problem.GetGroupCount()) != 0) ||
           (problem.GetInChannels_() % (GetKPerWave() * problem.GetGroupCount()) != 0))
            return false;
    }
    else
    {
        if((problem.GetOutChannels_() % (GetKPerWave() * problem.GetGroupCount()) != 0) ||
           (problem.GetInChannels_() % (GetCPerWave() * problem.GetGroupCount()) != 0))
            return false;
    }
    if((problem.GetOutChannels_() % (64 / chunk_size) != 0) &&
       (problem.GetInChannels_() % (64 / chunk_size) != 0))
        return false;
    if((reverse_inout != 0 ? problem.GetInChannels_() : problem.GetOutChannels_()) %
           GetCPerWave() !=
       0)
        return false;
    if(!(chunk_size * k_per_wave <= 64))
        return false;
    if((reverse_inout != 0 ? problem.GetOutChannels_() : problem.GetInChannels_()) % k_per_wave !=
       0)
        return false;
    if(!(n_per_group <= problem.GetBatchSize_()))
        return false;
    if(!(1 <= pipe_lines_depth && pipe_lines_depth <= std::min(problem.GetOutHeight_(), 16U)))
        return false;
    if((reverse_inout != 0) && !IsReverseInOutAllowed(problem))
        return false;
    {
        const int accums_cnt = (problem.GetWeightsWidth_() * problem.GetWeightsHeight_() *
                                GetCPerWave() * k_per_wave * chunk_size) /
                               64;
        assert(chunk_size);
        const int out_w_vec =
            (problem.GetOutWidth_() + elements_in_dword(problem) - 1) / elements_in_dword(problem);
        int gprs_per_line_in = (out_w_vec + chunk_size - 1) / chunk_size;
        if(chunk_size != 16)
        {
            assert(chunk_size - problem.GetPadW());
            gprs_per_line_in =
                (out_w_vec + chunk_size - problem.GetPadW() - 1) / (chunk_size - problem.GetPadW());
        }
        assert(problem.GetKernelStrideW());
        gprs_per_line_in += gprs_per_line_in % problem.GetKernelStrideW();
        const int gprs_per_line_out =
            (gprs_per_line_in > 1) ? gprs_per_line_in / problem.GetKernelStrideW() : 1;

        const int lines_in           = pipe_lines_depth + problem.GetWeightsHeight_() - 1;
        const int vgprs_for_lines_in = lines_in * elements_in_dword(problem) * gprs_per_line_in;
        assert(problem.GetKernelStrideH());
        const int lines_out =
            (pipe_lines_depth + problem.GetKernelStrideH() - 1) / problem.GetKernelStrideH();
        const int vgprs_for_lines_out = lines_out * elements_in_dword(problem) * gprs_per_line_out;
        const int vgprs_for_division =
            (vgprs_for_lines_in >= 4 ? 0 : 4) + (vgprs_for_lines_out >= 3 ? 0 : 3);

        const int k_group_size = problem.GetInChannels_() /
                                 (reverse_inout != 0 ? GetCPerWave() : GetKPerWave()) /
                                 problem.GetGroupCount();
        const bool k_group_size_is_power_of_two = ((k_group_size & (k_group_size - 1)) == 0);
        const int vgprs = accums_cnt + vgprs_for_lines_in + vgprs_for_lines_out +
                          (k_group_size_is_power_of_two ? 0 : vgprs_for_division) + 6 +
                          (elements_in_dword(problem) - 1);
        if(!(vgprs <= 256))
            return false;
        if(n_per_group > 4)
            if(!(vgprs <= 128))
                return false;
        if(limit_wave_cnt != 0 && limit_wave_cnt * 4 < n_per_group)
            return false;
        const auto lds_size = static_cast<std::size_t>(n_per_group - 1) * solver::wave_size *
                              sizeof(float) * accums_cnt;
        if(!(lds_size <= 65536))
            return false;

        const int unroll_factor = pipe_lines_depth * (pipe_lines_depth + 2);
        const int steps =
            std::max(0, static_cast<int>(problem.GetOutHeight_()) - 1 - pipe_lines_depth);
        assert(unroll_factor);
        const int loops        = pipe_lines_depth + unroll_factor + steps % unroll_factor + 1;
        const int m_instr      = 3 + (gprs_per_line_in + 3) / 4;
        const std::string name = ctx.GetStream().GetDeviceName();
        /// \todo parsing "gfx[0-9]+" and finding major/minor/stepping from handle. using this
        /// information here and in all similar places across other Solvers.
        const bool dot2_inst_avail = (name == "gfx906" || name == "gfx908");
        const bool dot2_emulate    = (!dot2_inst_avail) && (elements_in_dword(problem) == 2);
        const int v_instr =
            (k_per_wave * static_cast<int>(problem.GetWeightsHeight_()) * gprs_per_line_out *
             static_cast<int>(problem.GetWeightsWidth_()) * 4 * (dot2_emulate ? 2 : 1)) /
            3 * elements_in_dword(problem);
        const int exch_instr = elements_in_dword(problem) == 2 ? 3 * m_instr : 0;
        const int total =
            loops * (m_instr + v_instr + exch_instr) * elements_in_dword(problem); // instructions
        if(total >= 32000) // Estimation, a bit smaller than 32K.
            return false;
    }
    return true;
}

void PerformanceConfigAsmDirect3x3WrW::HeuristicInit(const ConvolutionContext& ctx,
                                                     const ProblemDescription& problem)
{
    limit_wave_cnt = 0;

    chunk_size = (problem.GetOutWidth_() < 48) ? 8 : 16;
    if((problem.GetOutChannels_() % (64 / chunk_size) != 0) &&
       (problem.GetInChannels_() % (64 / chunk_size) != 0))
        chunk_size = 16; // Fixup for correctness

    reverse_inout = 0;
    if(IsReverseInOutAllowed(problem) &&
       ((problem.GetOutChannels_() % 4 != 0) || (problem.GetOutWidth_() < 8)))
        reverse_inout = 1;

    const auto c_k =
        problem.GetOutChannels_() * problem.GetInChannels_() / problem.GetGroupCount(); // C*K
    if(c_k < 256)
        k_per_wave = 1;
    else if(c_k < 16384)
        k_per_wave = 2;
    else // C*K >= 16k
        k_per_wave = ((chunk_size == 8) ? 2 : 4);
    while((reverse_inout != 0 ? problem.GetOutChannels_() : problem.GetInChannels_()) %
              k_per_wave !=
          0)
        k_per_wave /= 2; // Fixup for correctness

    if(c_k <= 512)
        n_per_group = 8;
    else if(c_k <= 4096)
        n_per_group = 4;
    else if(c_k <= 8192)
        n_per_group = 2;
    else
        n_per_group = 1;
    if(n_per_group > problem.GetBatchSize_())
        n_per_group = problem.GetBatchSize_(); // n_per_group should never be > batch size.
    if(problem.GetOutWidth_() >= 256 &&
       n_per_group > 4) // when width >= 256, n_per_group should not be > 4.
        n_per_group = 4;

    pipe_lines_depth = (problem.GetOutHeight_() <= 1) ? 1 : 2;
    if((problem.GetOutHeight_() < 8) && (problem.GetOutWidth_() < 64))
    {
        pipe_lines_depth = problem.GetOutHeight_(); // Special case.
    }

    if(!IsValid(ctx, problem))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        limit_wave_cnt   = 0;
        reverse_inout    = 0;
        chunk_size       = 16; // CPerWave() = 4;
        k_per_wave       = 1;
        pipe_lines_depth = 2;
        n_per_group      = 1;
        if(problem.GetOutChannels_() % (4 * problem.GetGroupCount()) != 0)
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
        if(!IsValid(ctx, problem))
        {
            MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init 2...");
            pipe_lines_depth = 1;
        }
        assert(IsValid(ctx, problem));
    }
    MIOPEN_LOG_I(ToString());
}

PerformanceConfigAsmDirect3x3WrW
ConvAsmBwdWrW3x3::GetDefaultPerformanceConfig(const ConvolutionContext& ctx,
                                              const ProblemDescription& problem) const
{
    PerformanceConfigAsmDirect3x3WrW pp;
    pp.HeuristicInit(ctx, problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsmBwdWrW3x3::IsValidPerformanceConfig(
    const ConvolutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigAsmDirect3x3WrW& config) const
{
    return config.IsValidValue() && config.IsValid(ctx, problem);
}

bool ConvAsmBwdWrW3x3::IsApplicable(const ConvolutionContext& ctx,
                                    const ProblemDescription& problem) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3{}))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.direction.IsBackwardWrW())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!ctx.rmv.IsV2orV3())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx8") || StartsWith(name, "gfx9")))
        return false;
    if(!problem.IsLayoutDefault())
    {
        return false;
    }

    if(problem.IsTensorsCasted())
        return false;
#if WORKAROUND_ISSUE_532
    if(StartsWith(name, "gfx9") &&
       (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1))
        return false;
#endif

    if(name == "gfx90a" && problem.IsGfx90aFp16altRequired())
        return false;

#if WORKAROUND_SWDEV_330460
    if(!miopen::IsEnabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3{}) && name == "gfx90a" &&
       problem.IsFp32())
        return false;
#endif

    // clang-format off
    bool ok = problem.GetPadW() == 1        // -q  pad_w
        && problem.GetPadH() == 1           // -p  pad_h
        && problem.GetKernelStrideW() <= 2  // -v  stride_w
        && problem.GetKernelStrideH() <= 2  // -u  stride_h
        && problem.GetWeightsWidth_() == 3   // -x  S wei_w
        && problem.GetWeightsHeight_() == 3  // -y  R wei_h
        && problem.GetDilationW() == 1
        && problem.GetDilationH() == 1
        && problem.GetBias() == 0
        && (problem.IsFp32() || problem.IsFp16())
        && problem.GetInLayout() == "NCHW";
    if(!ok)
        return false; // Early exit to speed up the check.

    if(problem.IsFp16()
          && (StartsWith(name, "gfx8") // Not supported.
             || problem.GetBatchSize_() % 2 != 0)) /// \todo Initial version.
       return false;

    // Check limits:
    const auto h_w     = static_cast<int64_t>(problem.GetOutHeight_()) * problem.GetOutWidth_();
    const auto r_s     = static_cast<int64_t>(problem.GetWeightsHeight_()) * problem.GetWeightsWidth_();
    const auto c_h_w   = static_cast<int64_t>(problem.GetOutChannels_()) * h_w;     // C*H*W
    const auto k_h_w   = static_cast<int64_t>(problem.GetInChannels_()) * h_w;      // K*H*W
    const auto c_r_s   = static_cast<int64_t>(problem.GetOutChannels_()) * r_s;     // C*R*S
    const auto k_r_s   = static_cast<int64_t>(problem.GetInChannels_()) * r_s;      // K*R*S
    const auto n_c_h_w = static_cast<int64_t>(problem.GetBatchSize_()) * c_h_w;     // N*C*H*W
    const auto n_k_h_w = static_cast<int64_t>(problem.GetBatchSize_()) * k_h_w;     // N*K*H*W
    const auto c_k_r_s = static_cast<int64_t>(problem.GetOutChannels_()) * k_r_s;   // C*K*R*S
    ok = problem.GetOutWidth_() > 0
         && problem.GetOutWidth_() <= 512
         && (IsReverseInOutAllowed(problem)
                ? ((problem.GetOutChannels_() % (4 * problem.GetGroupCount()) == 0) || (problem.GetInChannels_() % (4 * problem.GetGroupCount()) == 0))
                : (problem.GetOutChannels_() % (4 * problem.GetGroupCount()) == 0))
         && problem.GetOutHeight_() < std::pow(2, 16)    // -H   H img_h
         && problem.GetBatchSize_() < std::pow(2, 16)    // -n   N batch_size
         && problem.GetOutChannels_() < std::pow(2, 16)  // -c   C input_channels
         && problem.GetInChannels_() < std::pow(2, 16)   // -k   K output_channels
         && c_h_w < std::pow(2, 22)
         && k_h_w < std::pow(2, 22)
         && c_r_s < std::pow(2, 22)
         && k_r_s < std::pow(2, 22)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29); // clang-format on
    return ok;
}

ConvSolution ConvAsmBwdWrW3x3::GetSolution(const ConvolutionContext& ctx,
                                           const ProblemDescription& problem,
                                           const PerformanceConfigAsmDirect3x3WrW& config) const
{
    ConvSolution result;
    std::ostringstream options;
    GenerateClangDefsym(options, "elements_in_dword", (problem.IsFp16()) ? 2 : 1);
    GenerateClangDefsym(options, "batch_size", problem.GetBatchSize_()); // N
    GenerateClangDefsym(options, "img_h", problem.GetOutHeight_());      // H
    GenerateClangDefsym(options, "img_w", problem.GetOutWidth_());       // W
    // Note that problem.n_outputs and problem.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "input_channels", problem.GetOutChannels_()); // C
    GenerateClangDefsym(options, "output_channels", problem.GetInChannels_()); // K
    GenerateClangDefsym(options, "wei_h", problem.GetWeightsHeight_());        // R
    GenerateClangDefsym(options, "wei_w", problem.GetWeightsWidth_());         // S
    GenerateClangDefsym(options, "pad_h", problem.GetPadH());
    GenerateClangDefsym(options, "pad_w", problem.GetPadW());
    GenerateClangDefsym(options, "stride_h", problem.GetKernelStrideH());
    GenerateClangDefsym(options, "stride_w", problem.GetKernelStrideW());
    GenerateClangDefsym(options, "weights_layout", 0);
    GenerateClangDefsym(options, "reverse_weights", 0);
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    // Perf tune:
    const PerformanceConfigAsmDirect3x3WrW* pcfg = &config;

    PerformanceConfigAsmDirect3x3WrW fromEnv;
    {
        std::string s;
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3_PERF_VALS{});
        if(p_asciz != nullptr)
        {
            s = std::string(p_asciz);
            if(!s.empty()) // else nothing to parse.
            {
                if(!fromEnv.Deserialize(s) || !fromEnv.IsValid(ctx, problem))
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
    GenerateClangDefsym(options, "group_counts", problem.GetGroupCount());

    const int k_group_size =
        problem.GetInChannels_() /
        (pcfg->reverse_inout != 0 ? pcfg->GetCPerWave() : pcfg->GetKPerWave()) /
        problem.GetGroupCount();
    const bool k_group_size_is_power_of_two = ((k_group_size & (k_group_size - 1)) == 0);
    GenerateClangDefsym(options, "k_group_size_is_power_of_two", k_group_size_is_power_of_two);

    KernelInfo kernel;

    kernel.comp_options = options.str();

    kernel.l_wk.clear(); // workgroupsize
    kernel.l_wk.push_back(static_cast<std::size_t>(64) * pcfg->GetNPerGroup());
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.clear(); // gridsize
    kernel.g_wk.push_back(static_cast<std::size_t>(64) * pcfg->GetNPerGroup());

    if(pcfg->GetReverseInout() == 0)
    {
        kernel.g_wk.push_back(problem.GetOutChannels_() / pcfg->GetCPerWave() /
                              problem.GetGroupCount());
        kernel.g_wk.push_back(problem.GetInChannels_() / pcfg->GetKPerWave());
    }
    else
    {
        kernel.g_wk.push_back(problem.GetOutChannels_() / pcfg->GetKPerWave() /
                              problem.GetGroupCount());
        kernel.g_wk.push_back(problem.GetInChannels_() / pcfg->GetCPerWave());
    }

    kernel.kernel_file = "conv3x3wrw.s";
    kernel.kernel_name = "miopenGcnAsmConv3x3WrW";

    result.construction_params.push_back(kernel);
    result.workspace_sz = 0;

    int N, C, H, W, K, n_groups;
    GetCompiledInParameters(ctx, problem, &N, &C, &H, &W, &K, &n_groups);

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

PerformanceConfigAsmDirect3x3WrW ConvAsmBwdWrW3x3::Search(const ConvolutionContext& ctx,
                                                          const ProblemDescription& problem,
                                                          const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

} // namespace solver
} // namespace miopen
