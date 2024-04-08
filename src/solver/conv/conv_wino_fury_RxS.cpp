/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/conv/invokers/gcn_asm_wino.hpp>
#include <miopen/conv/kernel_interface/winograd_kernel_interface.hpp>
#include <miopen/env.hpp>
#if !MIOPEN_USE_COMGR
#include <miopen/kernel_build_params.hpp>
#endif
#include <miopen/stringutils.hpp>

#define WORKAROUND_SWDEV_453577 1

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2)

#define IS2X3 (Winodata == 2 && Winofilter == 3)
#define IS3X2 (Winodata == 3 && Winofilter == 2)

constexpr std::size_t sync_buffer_size = 2048; // 2K

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription            = miopen::conv::ProblemDescription;
using WinoShaderArgsV40             = miopen::conv::WinoShaderArgsV40;
using WinoShaderActivationModeV40_t = miopen::conv::WinoShaderActivationModeV40_t;
using WinoShaderFlagsV40            = miopen::conv::WinoShaderFlagsV40;

namespace {

// Template is used to catch -Wshift-count-overflow
template <uint32_t exp>
constexpr uint32_t PowOf2()
{
    return 1U << exp;
}

// Divide two non-negative integers and return ceil of the quotient
constexpr uint64_t DivCeil(uint64_t numer, uint64_t denom) { return (numer + denom - 1) / denom; }

constexpr uint64_t RoundUpToMultiple(uint64_t val, uint64_t mul) { return DivCeil(val, mul) * mul; }

// Number of thread groups
uint32_t GetNGroups(uint64_t cu_count)
{
    // Current limitations:
    // n_groups < 2^8
    constexpr uint64_t max_n_groups = PowOf2<8>() - 1;

    return std::min(cu_count, max_n_groups);
}

bool IsShaderConstraintsMetV2(const WinoShaderArgsV40& args, uint32_t n_groups)
{
    // Current limitations:
    // clang-format off
    return args.N < PowOf2<16>()
        && args.C < PowOf2<16>()
        && args.H < PowOf2<16>()
        && args.W < PowOf2<16>()
        && args.pad_h >= std::numeric_limits<int16_t>::min() && args.pad_h <= std::numeric_limits<int16_t>::max()
        && args.pad_w >= std::numeric_limits<int16_t>::min() && args.pad_w <= std::numeric_limits<int16_t>::max()
        && args.out_h < PowOf2<16>()
        && args.out_w < PowOf2<16>() - 3
        && args.R <= 3
        && args.S <= 3
        && (static_cast<uint64_t>(args.N - 1) * args.C + 1) * args.H * args.W < PowOf2<31>()
        && (static_cast<uint64_t>(args.N - 1) * args.K + 1) * args.out_h * args.out_w < PowOf2<31>()
        && DivCeil(args.K, 16) <= n_groups
        && args.G == 1;
    // clang-format on
}

bool IsShaderConstraintsMet(const WinoShaderArgsV40& args, uint32_t n_groups)
{
    return IsShaderConstraintsMetV2(args, n_groups);
}

bool GpuHasReducedVGPRMem(const std::string& dev_name)
{
    if(dev_name == "gfx1100" || dev_name == "gfx1101" || dev_name == "gfx1151")
        return false;
    return true;
}

class ShaderModel
{
    const uint64_t N, C, K, R, S, oH, oW, G;
    const uint64_t n_groups;
    const uint32_t cu_count;
    const bool reduced_vgpr;

    struct PerfModelInfo
    {
        uint64_t predicted_clk;
        float granularity_loss;
    };

public:
    ShaderModel(const ExecutionContext& ctx,
                const WinoShaderArgsV40& args,
                uint32_t cu_cnt,
                uint32_t n_grp,
                bool reduced_vgpr_mem)
        : N(args.N),
          C(args.C),
          K(args.K),
          R(args.R),
          S(args.S),
          oH(args.out_h),
          oW(args.out_w),
          G(args.G),
          n_groups(n_grp),
          cu_count(cu_cnt),
          reduced_vgpr(reduced_vgpr_mem)
    {
        std::ignore = ctx;
    }

    bool IsC32ModePreferable() const
    {
        PerfModelInfo perf_model_c16, perf_model_c32;
        perf_model_c16 = PerfPrediction(false);
        perf_model_c32 = PerfPrediction(true);
        return perf_model_c32.predicted_clk <= perf_model_c16.predicted_clk;
    }

private:
    PerfModelInfo PerfPrediction(bool c32_mode) const
    {
        constexpr uint64_t t_R  = 3;
        constexpr uint64_t t_S  = 3;
        constexpr uint64_t t_oH = 2;
        constexpr uint64_t t_oW = 2;

        constexpr uint64_t nhw_factor   = 62;
        constexpr uint64_t k_factor     = 16;
        const uint64_t c_factor         = c32_mode ? 32 : 16;
        constexpr uint64_t nhw_factor_g = RoundUpToMultiple(nhw_factor, 32);

        const uint64_t Rg  = RoundUpToMultiple(R, t_R);
        const uint64_t Sg  = RoundUpToMultiple(S, t_S);
        const uint64_t Cg  = RoundUpToMultiple(C, c_factor);
        const uint64_t Kg  = RoundUpToMultiple(K, k_factor);
        const uint64_t oHg = RoundUpToMultiple(oH, t_oH);
        const uint64_t oWg = RoundUpToMultiple(oW, t_oW) + t_oW;

        const uint64_t c_loops = Cg / c_factor;
        const uint64_t k_ways  = Kg / k_factor;

        const uint64_t nkhw_per_work = k_factor * nhw_factor_g * t_oH * t_oW;

        const uint64_t nhw_tiles  = N * DivCeil(oHg, t_oH) * DivCeil(oWg, t_oW);
        const uint64_t n_groups_e = k_ways * (n_groups / k_ways);
        const uint64_t n_works    = k_ways * DivCeil(nhw_tiles, nhw_factor);
        const uint64_t n_works_per_cu =
            DivCeil(n_works, n_groups_e) * DivCeil(n_groups_e, cu_count);

        const uint64_t macsg = n_works_per_cu * cu_count * nkhw_per_work * Cg * Rg * Sg;
        const uint64_t macs  = N * G * K * C * oH * R * oW * S;

        PerfModelInfo out;
        out.granularity_loss = static_cast<float>(macsg - macs) / macsg;

        const uint64_t n_works_per_filter = reduced_vgpr ? 5 : 10;
        const uint64_t f_relaods = c_loops == 1 ? 1 : DivCeil(n_works_per_cu, n_works_per_filter);

        const uint64_t ph_start  = c32_mode ? 4 : 6;
        const uint64_t ph_accum  = n_works_per_cu * (c_loops - 1);
        const uint64_t ph_activ  = n_works_per_cu;
        const uint64_t ph_filter = f_relaods * c_loops;

        // Constant parameters of the model valid for gfx1100. Values for other ASICs may be
        // different, however as an approximate heuristic for choosing between C16 and C32
        // modes it would be enough.
        const uint64_t clk_start  = c32_mode ? 2600 : 1450;
        const uint64_t clk_accum  = c32_mode ? 2938 : 1645;
        const uint64_t clk_activ  = c32_mode ? 2989 : 1696;
        const uint64_t clk_filter = c32_mode ? 2600 : 1450;

        out.predicted_clk = ph_start * clk_start + ph_accum * clk_accum + ph_activ * clk_activ +
                            ph_filter * clk_filter;

        return out;
    }
};

} // namespace

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoFuryRxS<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    if constexpr(IS2X3)
    {
        if(miopen::IsDisabled(ENV(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3)))
            return false;
    }
    if constexpr(IS3X2)
    {
        if(miopen::IsDisabled(ENV(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2)))
            return false;
    }

    if(!ctx.use_asm_kernels)
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsFp16())
        return false;
    if(problem.HasNonPackedTensors())
        return false;

    const auto dev_name = ctx.GetStream().GetDeviceName();
    // All gfx11 ASICs are supported
    if(!StartsWith(dev_name, "gfx11"))
        return false;

    if(problem.GetInLayout() != "NCHW")
        return false;
    if(!(problem.GetKernelStrideH() == 1 && problem.GetKernelStrideW() == 1))
        return false;
    if(!(problem.GetDilationH() == 1 && problem.GetDilationW() == 1))
        return false;

    WinoShaderArgsV40 args;
    if(!args.SetConvParams(problem))
        return false;

    const auto cu_count = ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto n_groups = GetNGroups(cu_count);

    return IsShaderConstraintsMet(args, n_groups);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoFuryRxS<Winodata, Winofilter>::GetWti(const ExecutionContext& ctx,
                                                    const ProblemDescription& problem) const
{
    std::ignore = ctx;
    std::ignore = problem;

    return -2.0; // Unknown WTI
}

template <uint32_t Winodata, uint32_t Winofilter>
size_t
ConvWinoFuryRxS<Winodata, Winofilter>::GetWorkspaceSize(const ExecutionContext& ctx,
                                                        const ProblemDescription& problem) const
{
    std::ignore = problem;

    const bool coop_launch = ctx.GetStream().CooperativeLaunchSupported();
    return coop_launch ? sync_buffer_size : 0; // 2KB buffer for global sync
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoFuryRxS<Winodata, Winofilter>::GetSolution(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    const auto dev_name         = ctx.GetStream().GetDeviceName();
    const auto cu_count         = ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto n_groups         = GetNGroups(cu_count);
    const bool reduced_vgpr_mem = GpuHasReducedVGPRMem(dev_name);
#if WORKAROUND_SWDEV_453577
    const bool coop_launch = false;
#else
    const bool coop_launch = ctx.GetStream().CooperativeLaunchSupported();
#endif

    constexpr size_t wg_size = 384;

    WinoShaderArgsV40 args;
    // Main convolution parameters
    if(!args.SetConvParams(problem))
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    const auto shader_model = ShaderModel(ctx, args, cu_count, n_groups, reduced_vgpr_mem);
    // For ASICs with redused VGPR memory we have only c16 kernel
    const bool c32_mode = reduced_vgpr_mem ? false : shader_model.IsC32ModePreferable();

    // Warning
    static bool IsWarned = false;
    if(!IsWarned)
    {
        if(cu_count != n_groups)
        {
            MIOPEN_LOG_WE(SolverDbId()
                          << ": GPU has " << cu_count << " CUs, but this solver supports max "
                          << n_groups << " and thus may show sub-optimal performance.");
        }
        IsWarned = true;
    }

    // Kernel name & file
    const std::string kernel_version = "_v2_4_1";
    std::string kernel_name          = "miopenSp3AsmConvFury" + kernel_version;
    std::string kernel_file          = "Conv_Winograd_Fury" + kernel_version;

    if(StartsWith(dev_name, "gfx11"))
    {
        kernel_name += "_gfx11";
        kernel_name += reduced_vgpr_mem ? "_1024vgprs" : "_1536vgprs";
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    std::string kernel_postfix;

    if(problem.IsFp16())
    {
        kernel_postfix += "_fp16_fp16acc";
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    kernel_postfix += IS2X3 ? "_f2x3" : "_f3x2";
    kernel_postfix += c32_mode ? "_c32" : "_c16";
    kernel_postfix += "_stride1";

    kernel_name += kernel_postfix;
    kernel_file += kernel_postfix + ".s";

    // KernelInfo
    KernelInfo kernel;

#if !MIOPEN_USE_COMGR
    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5}, // For AmdgcnAssemble(...)
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
#endif
    kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.push_back(wg_size * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.kernel_file = kernel_file;
    kernel.kernel_name = kernel_name;

    // Data layout related parameters
    args.SetStrides(problem);

    // Fused activation parameters
    args.SetActivParams(WinoShaderActivationModeV40_t::IDENTITY, 0.0f, 0.0f);

    // Other shader parameters
    auto flags = WinoShaderFlagsV40::F_NKCHR_STRIDES | WinoShaderFlagsV40::F_TENSOR_OFFSETS |
                 WinoShaderFlagsV40::F_USE_ACTIVATION_MODE |
                 WinoShaderFlagsV40::F_USE_EXTENDED_FLAGS_64;
    if(problem.IsDirectionBackwardData())
        flags |= WinoShaderFlagsV40::F_REVERSE_R | WinoShaderFlagsV40::F_REVERSE_S;

    uint8_t sync_limit  = 0;
    uint8_t sync_period = 0;
    if(coop_launch)
    {
        sync_limit  = 255;
        sync_period = c32_mode ? 3 : 4;
    }
    args.SetShaderParams(n_groups, flags, sync_limit, sync_period);

    // Solution
    ConvSolution result;
    result.construction_params.push_back(kernel);
    result.invoker_factory = miopen::conv::MakeGcnAsmWinoV40InvokerFactory(
        args, problem.GetDirection(), coop_launch ? sync_buffer_size : 0);
    result.workspace_sz = GetWorkspaceSize(ctx, problem);

    return result;
}

template struct ConvWinoFuryRxS<2, 3>;
// template struct ConvWinoFuryRxS<3, 2>;

} // namespace conv
} // namespace solver
} // namespace miopen
