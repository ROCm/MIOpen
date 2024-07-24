/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/utils.hpp>

#include <boost/any.hpp>
#include <boost/optional.hpp>

#include <tuple>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1)

#define IS3X2 (Winodata == 3 && Winofilter == 2)

static inline size_t Ceil(const size_t v, const size_t m)
{
    assert(m > 0);
    return (v + m - 1) / m;
}

namespace miopen {
namespace solver {

namespace {

// Winograd v21 is preferred on Vega10/Vega20 ASICs due to ~25% performance regression with Winograd
// v30. The exception is Winograd F(3,2) stride2 as this mode is unsupported in v21. Details:
// https://github.com/ROCm/MIOpen/pull/1927#issuecomment-1412741130
template <int Winodata, int Winofilter>
inline bool IsWinogradV21Preferred(const std::string& asic,
                                   const miopen::conv::ProblemDescription& problem)
{
    return (StartsWith(asic, "gfx900") || StartsWith(asic, "gfx906")) &&
           !(IS3X2 && problem.GetKernelStrideW() == 2);
}

inline bool IsShaderConstraintsMetV21(const miopen::conv::ProblemDescription& problem,
                                      const int R,
                                      const int S,
                                      const int C,
                                      const int K,
                                      const int H,
                                      const int W,
                                      const int OH,
                                      const int OW,
                                      const int N)
{
    uint64_t o_K_stride      = static_cast<uint64_t>(OH) * OW;
    uint64_t o_N_stride      = o_K_stride * K;
    uint64_t o_N_stride_OHOW = o_N_stride + o_K_stride;

    uint64_t d_C_stride    = static_cast<uint64_t>(H) * W;
    uint64_t d_N_stride    = d_C_stride * C;
    uint64_t d_N_stride_HW = d_N_stride + d_C_stride;

    auto num_tiles  = Ceil(OH, 2) * Ceil(OW, 2);
    auto stride_one = problem.GetKernelStrideH() == 1 && problem.GetKernelStrideW() == 1 &&
                      problem.GetDilationH() == 1 && problem.GetDilationW() == 1;

    // clang-format off
    // Check implementation limits.
    return N < std::pow(2, 16)
        && C < std::pow(2, 16)
        && H < std::pow(2, 16)
        && W < std::pow(2, 16)
        && K < std::pow(2, 16)
        && S < std::pow(2, 16)
        && R < std::pow(2, 16)
        && OH < std::pow(2, 16)
        && OW < std::pow(2, 16)
        && problem.GetPadW() < std::pow(2, 16)
        && problem.GetPadH() < std::pow(2, 16)
        && C * R * S < std::pow(2, 22)
        && K * R * S < std::pow(2, 28)
        && ((o_N_stride_OHOW < std::pow(2, 29) && d_N_stride_HW < std::pow(2, 29))
           || (stride_one && o_N_stride < std::pow(2, 30) && d_N_stride < std::pow(2, 30)
           && (N == 1 || num_tiles % 16 == 0)));
    // clang-format on
}

inline bool IsShaderConstraintsMetV30(const miopen::conv::ProblemDescription& problem,
                                      const int R,
                                      const int S,
                                      const int C,
                                      const int K,
                                      const int H,
                                      const int W,
                                      const int OH,
                                      const int OW,
                                      const int N)
{
    // clang-format off
    // Check implementation limits.
    return N < std::pow(2, 16)
        && C < std::pow(2, 16)
        && H < std::pow(2, 16)
        && W < std::pow(2, 16)
        && K < std::pow(2, 16)
        && S < std::pow(2, 16)
        && R < std::pow(2, 16)
        && OH < std::pow(2, 16)
        && OW < std::pow(2, 16)
        && problem.GetPadW() < std::pow(2, 16)
        && problem.GetPadH() < std::pow(2, 16)
        && H * W < std::pow(2, 29)
        && K * R * S < std::pow(2, 28)
        && (C + 1) * H * W < std::pow(2, 30)
        && (C + 1) * R * S < std::pow(2, 22)
        && (K + 1) * OH * OW < std::pow(2, 30);
    // clang-format on
}

} // namespace

namespace fusion {

bool ConvBinWinogradRxSf2x3g1Fused::IsApplicable(const FusionContext& context,
                                                 const FusionDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1))
        return false;
    if(!WinoCommonIsApplicable(context, problem))
        return false;

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);

    const std::string name = conv_ctx.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9") || StartsWith(name, "gfx10") || StartsWith(name, "gfx11")))
        return false;

    if(conv_problem.IsFp16() &&
       !(StartsWith(name, "gfx906") || StartsWith(name, "gfx908") || StartsWith(name, "gfx90a") ||
         StartsWith(name, "gfx94") || StartsWith(name, "gfx1011") || StartsWith(name, "gfx1012") ||
         StartsWith(name, "gfx103") || StartsWith(name, "gfx11")))
        return false;

    // clang-format off
    if (!((conv_problem.GetKernelStrideW() == 1 || conv_problem.GetKernelStrideW() == 2)
        && conv_problem.GetKernelStrideW() == conv_problem.GetKernelStrideH()
        && conv_problem.GetDilationW() == 1
        && conv_problem.GetDilationH() == 1))
        return false;
    // clang-format on

    if(conv_problem.IsTensorsCasted())
        return false;

    const auto group_count = conv_problem.GetGroupCount();
    if(group_count != 1)
        return false;

    const auto W  = conv_problem.GetInWidth();
    const auto H  = conv_problem.GetInHeight();
    const auto C  = conv_problem.GetInChannels();
    const auto N  = conv_problem.GetInBatchSize();
    const auto K  = conv_problem.GetOutChannels();
    const auto R  = conv_problem.GetWeightsHeight();
    const auto S  = conv_problem.GetWeightsWidth();
    const auto OH = conv_problem.GetOutHeight();
    const auto OW = conv_problem.GetOutWidth();

    return IsWinogradV21Preferred<2, 3>(name, conv_problem)
               ? IsShaderConstraintsMetV21(conv_problem, R, S, C, K, H, W, OH, OW, N)
               : IsShaderConstraintsMetV30(conv_problem, R, S, C, K, H, W, OH, OW, N);
}

ConvSolution ConvBinWinogradRxSf2x3g1Fused::GetSolution(const FusionContext& context,
                                                        const FusionDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;

    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto conv_ctx     = context.GetConvContext(conv_problem);

    const int n_groups  = conv_ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto name     = conv_ctx.GetStream().GetDeviceName();
    const auto is_gfx9  = StartsWith(name, "gfx9");
    const auto is_gfx10 = StartsWith(name, "gfx10");
    const auto is_v21   = IsWinogradV21Preferred<2, 3>(name, conv_problem);
    size_t wg_size      = is_gfx9 ? 512 : 256;
    kernel.g_wk.push_back(wg_size * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto force_cache_bypass = (name == "gfx940") || (name == "gfx941");

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
        {"FORCE_CACHE_BYPASS_ON_STORE", force_cache_bypass},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
    kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");

    const std::string kernel_version = is_v21 ? "_v21_1_3" : "_v30_3_1";
    kernel.kernel_file               = "Conv_Winograd" + kernel_version;
    kernel.kernel_name               = "miopenSp3AsmConv" + kernel_version;
    const auto kernel_postfix =
        "_fp32_f2x3_stride" + std::to_string(conv_problem.GetKernelStrideH());

    if(is_gfx9)
    {
        kernel.kernel_name += "_gfx9";
    }
    else if(is_gfx10)
    {
        kernel.kernel_name += "_gfx10";
    }
    else // if (is_gfx11)
    {
        kernel.kernel_name += "_gfx11";
    }

    kernel.kernel_name += kernel_postfix;
    kernel.kernel_file += kernel_postfix + ".s";
    result.construction_params.push_back(kernel);

    const auto& desc    = *problem.fusion_plan_desc;
    const int bias_idx  = GetOpIdx(desc.op_map, miopenFusionOpBiasForward);
    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);
    int N, C, H, W, K, unused, out_H, out_W, R, S, pad_H, pad_W;
    GetCompiledInParameters(
        context, conv_problem, &N, &C, &H, &W, &K, &unused, &out_H, &out_W, &R, &S, &pad_H, &pad_W);

    // todo: move into lambda when activation_mode will be fully supported
    constexpr int F_USE_ACTIVATION_MODE = 1 << 14;

    int flags = [&]() {
        constexpr int F_BIAS       = 1 << 7;
        constexpr int F_LEAKY_RELU = 1 << 8;

        // todo: use F_USE_ACTIVATION_MODE
        int flag = 0;

        if(bias_idx != -1)
            flag |= F_BIAS;
        if(activ_idx != -1) // unused in v30_3_1, has been kept for compatibility with v21
            flag |= F_LEAKY_RELU;

        return flag;
    }();

    constexpr uint8_t IDENTITY    = 0;
    constexpr uint8_t LEAKY_RELU  = 1;
    constexpr uint8_t SIGMOID     = 2;
    constexpr uint8_t SCALED_TANH = 3;

    uint8_t activation_mode_v30_3_1 = [&op_map = desc.op_map, activ_idx]() {
        if(activ_idx != -1)
        {
            const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*op_map[activ_idx]);
            switch(activ_op.activMode)
            {
            case miopenActivationLOGISTIC: return SIGMOID;
            case miopenActivationTANH: return SCALED_TANH;
            case miopenActivationLEAKYRELU: return LEAKY_RELU;
            case miopenActivationPASTHRU:
            case miopenActivationRELU:
            case miopenActivationSOFTRELU:
            case miopenActivationABS:
            case miopenActivationPOWER:
            case miopenActivationCLIPPEDRELU:
            case miopenActivationELU:
            default: return IDENTITY;
            };
        }
        return IDENTITY;
    }();

    const miopenActivationMode_t activation_mode_compat = [&]() {
        if(activ_idx != -1)
        {
            const auto& activ_op =
                dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[activ_idx]);
            return activ_op.activMode;
        }
        return miopenActivationPASTHRU;
    }();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& launch_kernel = handle.Run(kernels[0]);
            const auto& invoke_ctx =
                primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
            const auto& bot_buf = invoke_ctx.in;
            const auto& wei_buf = dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(
                                      *invoke_ctx.op_args.params[0])
                                      .weights;
            const auto& top_buf = invoke_ctx.out;
            const auto bias_ptr = [&]() {
                if(bias_idx != -1)
                {
                    return dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(
                               *invoke_ctx.op_args.params[1])
                        .bdata;
                }
                else
                    return static_cast<ConstData_t>(nullptr);
            }();

            float activ_alpha = [&]() {
                if(activ_idx != -1)
                {
                    const auto& activ_args = dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(
                        *invoke_ctx.op_args.params[activ_idx]);
                    if(activation_mode_compat == miopenActivationLEAKYRELU)
                        return static_cast<float>(activ_args.activAlpha);
                }
                return static_cast<float>(0.0);
            }();

            float activ_beta = [&]() {
                if(((flags & F_USE_ACTIVATION_MODE) != 0) &&
                   (activation_mode_v30_3_1 == SCALED_TANH))
                {
                    const auto& activ_args = dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(
                        *invoke_ctx.op_args.params[activ_idx]);
                    return static_cast<float>(activ_args.activBeta);
                }
                return static_cast<float>(0.0);
            }();

            auto zero_u64  = static_cast<uint64_t>(0);
            const int zero = 0;

            launch_kernel(N,
                          C,
                          H,
                          W,
                          K,
                          n_groups, // Not related to group convolutions
                          flags,    // flags
                          zero,     // reserved
                          bot_buf,
                          wei_buf,
                          top_buf,
                          static_cast<void*>(nullptr), // return_addr
                          R,
                          S,
                          pad_H,
                          pad_W,
                          out_H,
                          out_W,
                          bias_ptr,
                          activ_alpha, // leaky relu \ scaled tanh alpha
                          activ_beta,  // scaled tanh beta, always 0 for compatibility
                          zero_u64,    // d_offset", Other, zero_uint64),
                          zero_u64,    // f_offset", Other, zero_uint64),
                          zero_u64,    // o_offset", Other, zero_uint64),
                          zero_u64,    // b_offset", Other, zero_uint64),
                          zero,        // d_stride_nk", InputTensorDesc, zero_int),
                          zero,        // d_stride_c", InputTensorDesc, zero_int),
                          zero,        // d_stride_h", InputTensorDesc, zero_int),
                          zero,        // d_stride_w", InputTensorDesc, zero_int),
                          zero,        // f_stride_nk", OpAttr, zero_int),
                          zero,        // f_stride_c", OpAttr, zero_int),
                          zero,        // f_stride_h", OpAttr, zero_int),
                          zero,        // f_stride_w", OpAttr, zero_int),
                          zero,        // o_stride_nk", OutputTensorDesc, zero_int),
                          zero,        // o_stride_c", OutputTensorDesc, zero_int),
                          zero,        // o_stride_h", OutputTensorDesc, zero_int),
                          zero,        // o_stride_w", OutputTensorDesc, zero_int),
                          zero,        // group_count", OpAttr, zero_int),
                          zero,        // d_stride_g", Other, zero_int),
                          zero,        // f_stride_g", Other, zero_int),
                          zero         // o_stride_g", Other, zero_int),
            );                         // todo: turn on activation_mode
        };
    };
    return result;
}
float ConvBinWinogradRxSf2x3g1Fused::GetWti(const FusionContext& ctx,
                                            const FusionDescription& problem) const
{
    const auto conv_problem = problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    return conv::ConvBinWinogradRxSf2x3g1().GetWti(ctx, conv_problem);
}
} // namespace fusion
} // namespace solver
} // namespace miopen
