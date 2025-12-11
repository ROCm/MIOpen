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

#define CONV_WINO_FURY_RXS_CPP

#include <miopen/conv/solvers.hpp>
#include <miopen/fusion/solvers.hpp>

#include <miopen/conv/invokers/gcn_asm_wino.hpp>
#include <miopen/conv/kernel_interface/winograd_kernel_interface.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/fusion/utils.hpp>

#define WORKAROUND_SWDEV_453577 1

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2)

#define IS2X3 (Winodata == 2 && Winofilter == 3)
#define IS3X2 (Winodata == 3 && Winofilter == 2)

namespace miopen {
namespace solver {

using ProblemDescription           = miopen::conv::ProblemDescription;
using WinoShaderArgsV2             = miopen::WinoShaderArgsV2;
using WinoShaderActivationModeV2_t = miopen::WinoShaderActivationModeV2_t;
using WinoShaderFlagsV2            = miopen::WinoShaderFlagsV2;

namespace {

constexpr std::size_t sync_buffer_size = 2048; // 2K

// Template is used to catch -Wshift-count-overflow
template <uint32_t exp>
constexpr uint32_t PowOf2()
{
    return 1U << exp;
}

// Number of thread groups
uint32_t GetNGroups(uint64_t cu_count)
{
    // Current limitations:
    // n_groups < 2^8
    constexpr uint64_t max_n_groups = PowOf2<8>() - 1;

    return std::min(cu_count, max_n_groups);
}

bool GpuHasReducedVGPRMem(const std::string& dev_name)
{
    static constexpr std::array<std::string_view, 5> kFullVgprMemDevices{
        "gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201"};
    const std::string_view name{dev_name};
    return std::find(kFullVgprMemDevices.begin(), kFullVgprMemDevices.end(), name) ==
           kFullVgprMemDevices.end();
}

struct PerfModelInfo
{
    uint64_t predicted_clk;
    float granularity_loss;
};

struct PerfModelCost
{
    uint64_t start_cost;
    uint64_t accum_cost;
    uint64_t activ_cost;
    uint64_t filter_cost;
};

// Base class for shader performance models
class ShaderModel
{
protected:
    const uint64_t N, C, K, R, S, oH, oW, G, H, W;
    const int32_t pad_h, pad_w;
    const uint64_t n_groups;
    const uint32_t cu_count;
    const bool reduced_vgpr;
    std::array<PerfModelCost, 2> model_params;

public:
    static constexpr float default_wti = -2.0f;
    ShaderModel(const WinoShaderArgsV2& args,
                uint32_t cu_cnt,
                uint32_t n_grp,
                bool reduced_vgpr_mem)
        : N(args.N),
          C(args.Cg),
          K(args.Kg),
          R(args.R),
          S(args.S),
          oH(args.out_h),
          oW(args.out_w),
          G(args.G),
          H(args.H),
          W(args.W),
          pad_h(args.pad_h),
          pad_w(args.pad_w),
          n_groups(n_grp),
          cu_count(cu_cnt),
          reduced_vgpr(reduced_vgpr_mem)
    {
    }

    virtual ~ShaderModel() = default;

    bool IsC32ModePreferable(uint64_t& out_predicted_clk) const
    {
        bool result = false;
        PerfModelInfo perf_model_c16, perf_model_c32;
        perf_model_c16 = PerfPrediction(false);
        perf_model_c32 = PerfPrediction(true);
        if(perf_model_c32.predicted_clk <= perf_model_c16.predicted_clk)
        {
            result            = true;
            out_predicted_clk = perf_model_c32.predicted_clk;
        }
        else
        {
            out_predicted_clk = perf_model_c16.predicted_clk;
        }
        return result;
    }

    virtual bool IsShaderConstraintsMet() const = 0;
    virtual float ComputeWti() const            = 0;

protected:
    // Divide two non-negative integers and return ceil of the quotient
    uint64_t DivCeil(uint64_t numer, uint64_t denom) const { return (numer + denom - 1) / denom; }
    uint64_t RoundUpToMultiple(uint64_t val, uint64_t mul) const { return DivCeil(val, mul) * mul; }
    PerfModelInfo PerfPrediction(bool c32_mode) const
    {
        constexpr uint64_t t_R  = 3;
        constexpr uint64_t t_S  = 3;
        constexpr uint64_t t_oH = 2;
        constexpr uint64_t t_oW = 2;

        constexpr uint64_t nhw_factor = 62;
        constexpr uint64_t k_factor   = 16;
        const uint64_t c_factor       = c32_mode ? 32 : 16;
        const uint64_t nhw_factor_g   = RoundUpToMultiple(nhw_factor, 32);

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

        const uint64_t start_cost  = model_params[c32_mode ? 1 : 0].start_cost;
        const uint64_t accum_cost  = model_params[c32_mode ? 1 : 0].accum_cost;
        const uint64_t activ_cost  = model_params[c32_mode ? 1 : 0].activ_cost;
        const uint64_t filter_cost = model_params[c32_mode ? 1 : 0].filter_cost;

        out.predicted_clk = ph_start * start_cost + ph_accum * accum_cost + ph_activ * activ_cost +
                            ph_filter * filter_cost;
        return out;
    }
};

class ShaderModelV2 : public ShaderModel
{
public:
    static constexpr std::array<PerfModelCost, 2> GFX11_ModelParams{{
        {1450, 1645, 1696, 1450}, // C16 mode
        {2600, 2938, 2989, 2600}  // C32 mode
    }};

    ShaderModelV2(const WinoShaderArgsV2& args,
                  uint32_t cu_cnt,
                  uint32_t n_grp,
                  bool reduced_vgpr_mem)
        : ShaderModel(args, cu_cnt, n_grp, reduced_vgpr_mem)
    {
        model_params = GFX11_ModelParams;
    }

    bool IsShaderConstraintsMet() const override
    {
        // Current limitations:
        // clang-format off
        return N < PowOf2<16>()
            && C < PowOf2<16>()
            && H < PowOf2<16>()
            && W < PowOf2<16>()
            && pad_h >= std::numeric_limits<int16_t>::min() && pad_h <= std::numeric_limits<int16_t>::max()
            && pad_w >= std::numeric_limits<int16_t>::min() && pad_w <= std::numeric_limits<int16_t>::max()
            && oH < PowOf2<16>()
            && oW < PowOf2<16>() - 3
            && R <= 3
            && S <= 3
            && (static_cast<uint64_t>(N - 1) * C + 1) * H * W < PowOf2<31>()
            && (static_cast<uint64_t>(N - 1) * K + 1) * oH * oW < PowOf2<31>()
            && DivCeil(K, 16) <= n_groups
            && G == 1;
        // clang-format on
    }

    float ComputeWti() const override { return default_wti; }
};

class ShaderModelV4 : public ShaderModel
{
public:
    static constexpr uint64_t GFX12_MACRate = 512; // Float16 MAC operations per CU per clock
    static constexpr std::array<PerfModelCost, 2> GFX12_ModelParams{{
        {1010, 1343, 1495, 1010}, // C16 mode
        {1746, 2287, 2443, 1746}  // C32 mode
    }};

    ShaderModelV4(const WinoShaderArgsV2& args,
                  uint32_t cu_cnt,
                  uint32_t n_grp,
                  bool reduced_vgpr_mem)
        : ShaderModel(args, cu_cnt, n_grp, reduced_vgpr_mem)
    {
        model_params = GFX12_ModelParams;
    }

    bool IsShaderConstraintsMet() const override
    {
        // clang-format off
        return N < PowOf2<16>()
            && G < PowOf2<16>()
            && C < PowOf2<16>()
            && K < PowOf2<16>()
            && H < PowOf2<16>()
            && W < PowOf2<16>()
            && R <= 3
            && S <= 3
            && G * K < (1LL << 16)
            && G * C < (1LL << 16)
            && oH < PowOf2<16>()
            && oW < PowOf2<16>() - 3
            && ((G * K - 1) * C + 1) * R * S < (1LL << 28)
            && (static_cast<uint64_t>(N - 1) * G * C + 1) * H * W < PowOf2<31>()
            && (static_cast<uint64_t>(N - 1) * G * K + 1) * oH * oW < PowOf2<31>()
            && pad_h + static_cast<int64_t>(H) <= (1LL << 16)
            && pad_w + static_cast<int64_t>(W) <= (1LL << 16)
            && abs(pad_h) + oH + R             <= (1LL << 16)
            && abs(pad_w) + oW + S             <= (1LL << 16)
            && DivCeil(K, 16) <= n_groups;
        // clang-format on
    }

    float ComputeWti() const override
    {
        const uint64_t macs          = N * G * K * C * oH * R * oW * S;
        const float ideal_direct_clk = static_cast<float>(macs) / GFX12_MACRate / cu_count;
        uint64_t predicted_clk       = 0;
        bool is_c32_mode             = IsC32ModePreferable(predicted_clk);
        std::ignore                  = is_c32_mode;
        return predicted_clk != 0 ? ideal_direct_clk / predicted_clk : default_wti;
    }
};

// Factory class for creating appropriate shader model based on device architecture
class ShaderModelFactory
{
public:
    static std::unique_ptr<ShaderModel> Create(const std::string& dev_name,
                                               const WinoShaderArgsV2& args,
                                               uint32_t cu_count,
                                               uint32_t n_groups,
                                               bool reduced_vgpr_mem)
    {
        if(StartsWith(dev_name, "gfx11"))
        {
            return std::make_unique<ShaderModelV2>(args, cu_count, n_groups, reduced_vgpr_mem);
        }
        else if(StartsWith(dev_name, "gfx12"))
        {
            return std::make_unique<ShaderModelV4>(args, cu_count, n_groups, reduced_vgpr_mem);
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError, "Unsupported device architecture: " + dev_name);
        }
    }
};

bool IsShaderConstraintsMet(const WinoShaderArgsV2& args,
                            uint32_t n_groups,
                            const std::string& dev_name)
{
    const bool reduced_vgpr_mem = GpuHasReducedVGPRMem(dev_name);
    // The cu_count is not required for shader constraint checks.
    // It is simply assigned the value of n_groups for reference.
    const uint32_t cu_count = n_groups;

    auto shader_model =
        ShaderModelFactory::Create(dev_name, args, cu_count, n_groups, reduced_vgpr_mem);
    return shader_model->IsShaderConstraintsMet();
}

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoFuryRxSCommon
{
    static bool IsApplicable(const ExecutionContext&, const ProblemDescription&);
    static float GetWti(const ExecutionContext&, const ProblemDescription&);
    static size_t GetWorkspaceSize(const ExecutionContext&, bool fused = false);
    static ConvSolution GetSolution(const ExecutionContext&,
                                    const ProblemDescription&,
                                    bool fused                        = false,
                                    bool do_bias                      = false,
                                    miopenActivationMode_t activ_mode = miopenActivationPASTHRU);
};

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoFuryRxSCommon<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                               const ProblemDescription& problem)
{
    if constexpr(IS2X3)
    {
        if(env::disabled(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3))
            return false;
    }
    if constexpr(IS3X2)
    {
        if(env::disabled(MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2))
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
    // All gfx11/gfx12 ASICs are supported
    if(!(StartsWith(dev_name, "gfx11") || StartsWith(dev_name, "gfx12")))
        return false;

    if(!(problem.GetKernelStrideH() == 1 && problem.GetKernelStrideW() == 1))
        return false;
    if(!(problem.GetDilationH() == 1 && problem.GetDilationW() == 1))
        return false;

    WinoShaderArgsV2 args;
    if(!args.SetConvParams(problem))
        return false;

    const auto cu_count = ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto n_groups = GetNGroups(cu_count);

    return IsShaderConstraintsMet(args, n_groups, dev_name);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetWti(const ExecutionContext& ctx,
                                                          const ProblemDescription& problem)
{
    std::ignore = problem;

    const auto dev_name         = ctx.GetStream().GetDeviceName();
    const auto cu_count         = ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto n_groups         = GetNGroups(cu_count);
    const bool reduced_vgpr_mem = GpuHasReducedVGPRMem(dev_name);
    WinoShaderArgsV2 args;

    // Main convolution parameters
    if(!args.SetConvParams(problem))
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    auto shader_model =
        ShaderModelFactory::Create(dev_name, args, cu_count, n_groups, reduced_vgpr_mem);
    return shader_model->ComputeWti();
}

template <uint32_t Winodata, uint32_t Winofilter>
size_t ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetWorkspaceSize(const ExecutionContext& ctx,
                                                                     bool fused)
{
    // fusions do not support workspace
    if(fused)
        return 0;

    const bool coop_launch = ctx.GetStream().CooperativeLaunchSupported();
    return coop_launch ? sync_buffer_size : 0; // 2KB buffer for global sync
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetSolution(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem,
                                                         bool fused,
                                                         bool do_bias,
                                                         miopenActivationMode_t activ_mode)
{
    const auto dev_name         = ctx.GetStream().GetDeviceName();
    const auto cu_count         = ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto n_groups         = GetNGroups(cu_count);
    const bool reduced_vgpr_mem = GpuHasReducedVGPRMem(dev_name);
#if WORKAROUND_SWDEV_453577
    const bool coop_launch = false;
#else
    // fusions do not support workspace
    const bool coop_launch = !fused && ctx.GetStream().CooperativeLaunchSupported();
#endif

    constexpr size_t wg_size = 384;

    WinoShaderArgsV2 args;
    // Main convolution parameters
    if(!args.SetConvParams(problem))
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    if(!problem.IsFp16())
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    auto shader_model =
        ShaderModelFactory::Create(dev_name, args, cu_count, n_groups, reduced_vgpr_mem);
    // For ASICs with redused VGPR memory we have only c16 kernel
    uint64_t predicted_clk = 0;
    const bool c32_mode =
        reduced_vgpr_mem ? false : shader_model->IsC32ModePreferable(predicted_clk);
    std::ignore = predicted_clk;

    // Warning
    static bool IsWarned = false;
    if(!IsWarned)
    {
        if(cu_count != n_groups)
        {
            MIOPEN_LOG_WE("ConvWinoFuryRxSCommon"
                          << ": GPU has " << cu_count << " CUs, but this solver supports max "
                          << n_groups << " and thus may show sub-optimal performance.");
        }
        IsWarned = true;
    }

    // Build up kernel name & file
    std::string kernel_version = "_v2_4_1";
    std::string kernel_name    = "miopenSp3AsmConvFury";
    std::string kernel_file    = "Conv_Winograd_Fury";
    std::string kernel_postfix = "_fp16_fp16acc";
    std::string kernel_arch    = "_gfx11";

    const bool is_gfx11 = StartsWith(dev_name, "gfx11");
    const bool is_gfx12 = StartsWith(dev_name, "gfx12");

    if(!is_gfx11 && !is_gfx12)
        MIOPEN_THROW(miopenStatusInternalError);

    if(is_gfx12)
    {
        kernel_version = "_v4_6_0";
        kernel_arch    = "_gfx12";
        kernel_postfix = "_fp16_fp32acc";
    }

    kernel_postfix += IS2X3 ? "_f2x3" : "_f3x2";
    kernel_postfix += c32_mode ? "_c32" : "_c16";
    kernel_postfix += "_stride1";

    kernel_name += kernel_version;
    kernel_name += kernel_arch;
    kernel_name += reduced_vgpr_mem ? "_1024vgprs" : "_1536vgprs";
    kernel_name += kernel_postfix;

    kernel_file += kernel_version;
    kernel_file += kernel_postfix + ".s";

    // KernelInfo
    KernelInfo kernel;

    /// Kernel doesn't need ROCM_METADATA_VERSION, but AmdgcnAssemble()
    /// uses it to find out required CO version (hack).
    /// \todo Delete when COv2 support is removed.
    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
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
    args.SetActivParams(activ_mode);

    // Other shader parameters
    auto flags = WinoShaderFlagsV2::F_NKCHR_STRIDES | WinoShaderFlagsV2::F_TENSOR_OFFSETS |
                 WinoShaderFlagsV2::F_USE_ACTIVATION_MODE |
                 WinoShaderFlagsV2::F_USE_EXTENDED_FLAGS_64;
    if(problem.IsDirectionBackwardData())
        flags |= WinoShaderFlagsV2::F_REVERSE_R | WinoShaderFlagsV2::F_REVERSE_S;
    if(do_bias)
        flags |= WinoShaderFlagsV2::F_BIAS;

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
    result.invoker_factory = miopen::MakeGcnAsmWinoV2InvokerFactory(
        args, problem.GetDirection(), coop_launch ? sync_buffer_size : 0, fused);
    result.workspace_sz = GetWorkspaceSize(ctx, fused);

    return result;
}

} // namespace

namespace conv {

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoFuryRxS<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::IsApplicable(ctx, problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoFuryRxS<Winodata, Winofilter>::GetWti(const ExecutionContext& ctx,
                                                    const ProblemDescription& problem) const
{
    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetWti(ctx, problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
size_t
ConvWinoFuryRxS<Winodata, Winofilter>::GetWorkspaceSize(const ExecutionContext& ctx,
                                                        const ProblemDescription& problem) const
{
    std::ignore = problem;

    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetWorkspaceSize(ctx);
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoFuryRxS<Winodata, Winofilter>::GetSolution(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetSolution(ctx, problem);
}

template struct MIOPEN_INTERNALS_EXPORT ConvWinoFuryRxS<2, 3>;
// template struct MIOPEN_INTERNALS_EXPORT ConvWinoFuryRxS<3, 2>;

} // namespace conv

namespace fusion {

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoFuryRxSFused<Winodata, Winofilter>::IsApplicable(
    const FusionContext& ctx, const FusionDescription& problem) const
{
    const auto& desc = *problem.fusion_plan_desc;

    if(desc.op_map.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    if(desc.op_map.size() > 3)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map.size() == 2)
    {
        const auto prim = desc.op_map[1]->kind();
        if(!(prim == miopenFusionOpBiasForward || prim == miopenFusionOpActivForward))
            return false;
    }
    if(desc.op_map.size() == 3)
    {
        if(desc.op_map[1]->kind() != miopenFusionOpBiasForward)
            return false;
        if(desc.op_map[2]->kind() != miopenFusionOpActivForward)
            return false;
    }

    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);
    if(activ_idx != -1)
    {
        const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        switch(activ_op.activMode)
        {
        case miopenActivationPASTHRU:
        case miopenActivationLOGISTIC:
        case miopenActivationTANH:
        case miopenActivationLEAKYRELU: break;
        default: return false;
        }
    }

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::IsApplicable(ctx, conv_problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoFuryRxSFused<Winodata, Winofilter>::GetWti(const FusionContext& ctx,
                                                         const FusionDescription& problem) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetWti(ctx, conv_problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
size_t
ConvWinoFuryRxSFused<Winodata, Winofilter>::GetWorkspaceSize(const FusionContext& ctx,
                                                             const FusionDescription& problem) const
{
    std::ignore = problem;

    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetWorkspaceSize(ctx, true);
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoFuryRxSFused<Winodata, Winofilter>::GetSolution(const FusionContext& ctx,
                                                        const FusionDescription& problem) const
{
    const auto& desc    = *problem.fusion_plan_desc;
    const int bias_idx  = GetOpIdx(desc.op_map, miopenFusionOpBiasForward);
    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);

    const bool do_bias = (bias_idx != -1);
    auto activ_mode    = miopenActivationPASTHRU;
    if(activ_idx != -1)
    {
        const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
        activ_mode           = activ_op.activMode;
    }

    return ConvWinoFuryRxSCommon<Winodata, Winofilter>::GetSolution(
        ctx, conv_problem, true, do_bias, activ_mode);
}

template struct MIOPEN_INTERNALS_EXPORT ConvWinoFuryRxSFused<2, 3>;
// template struct MIOPEN_INTERNALS_EXPORT ConvWinoFuryRxSFused<3, 2>;

} // namespace fusion

} // namespace solver
} // namespace miopen
