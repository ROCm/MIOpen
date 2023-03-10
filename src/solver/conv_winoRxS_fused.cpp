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
#include <iomanip>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_G1)

#define IS2X3 (Winodata == 2 && Winofilter == 3)
#define IS3X2 (Winodata == 3 && Winofilter == 2)

static inline size_t Ceil(const size_t v, const size_t m)
{
    assert(m > 0);
    return (v + m - 1) / m;
}

static inline size_t RoundUpToMultiple(size_t val, int factor)
{
    return Ceil(val, factor) * factor;
}

namespace miopen {
namespace solver {

namespace {

// Winograd v21 is preferred on Vega10/Vega20 ASICs due to ~25% performance regression with Winograd
// v30. The exception is Winograd F(3,2) stride2 as this mode is unsupported in v21. Details:
// https://github.com/ROCmSoftwarePlatform/MIOpen/pull/1927#issuecomment-1412741130
template <int Winodata, int Winofilter>
inline bool IsWinogradV21Preferred(const std::string& asic, const ProblemDescription& problem)
{
    return (StartsWith(asic, "gfx900") || StartsWith(asic, "gfx906")) &&
           !(IS3X2 && problem.kernel_stride_w == 2);
}

inline bool IsShaderConstraintsMetV21(const ProblemDescription& problem,
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
    const uint64_t o_K_stride      = static_cast<uint64_t>(OH) * OW;
    const uint64_t o_N_stride      = o_K_stride * K;
    const uint64_t o_N_stride_OHOW = o_N_stride + o_K_stride;

    const uint64_t d_C_stride    = static_cast<uint64_t>(H) * W;
    const uint64_t d_N_stride    = d_C_stride * C;
    const uint64_t d_N_stride_HW = d_N_stride + d_C_stride;

    const auto num_tiles  = Ceil(OH, 2) * Ceil(OW, 2);
    const auto stride_one = problem.kernel_stride_h == 1 && problem.kernel_stride_w == 1 &&
                            problem.kernel_dilation_h == 1 && problem.kernel_dilation_w == 1;

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
        && problem.pad_w < std::pow(2, 16)
        && problem.pad_h < std::pow(2, 16)
        && C * R * S < std::pow(2, 22)
        && K * R * S < std::pow(2, 28)
        && ((o_N_stride_OHOW < std::pow(2, 29) && d_N_stride_HW < std::pow(2, 29))
           || (stride_one && o_N_stride < std::pow(2, 30) && d_N_stride < std::pow(2, 30)
           && (N == 1 || num_tiles % 16 == 0)));
    // clang-format on
}

inline bool IsShaderConstraintsMetV30(const ProblemDescription& problem,
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
        && problem.pad_w < std::pow(2, 16)
        && problem.pad_h < std::pow(2, 16)
        && H * W < std::pow(2, 29)
        && K * R * S < std::pow(2, 28)
        && (C + 1) * H * W < std::pow(2, 30)
        && (C + 1) * R * S < std::pow(2, 22)
        && (K + 1) * OH * OW < std::pow(2, 30);
    // clang-format on
}

template <int Winodata, int Winofilter>
float GetGranularityLoss(const ProblemDescription& problem,
                         const int R,
                         const int S,
                         const int C,
                         const int K,
                         const int OH,
                         const int OW,
                         const int N,
                         const int G,
                         const int n_groups,
                         const int cu_count)
{
    const auto ostride_h = problem.kernel_stride_h;
    const auto ostride_w = problem.kernel_stride_w;
    const auto dstride_h = problem.kernel_dilation_h;
    const auto dstride_w = problem.kernel_dilation_w;

    // clang-format off
    const bool single_traverse_mode = ostride_h == 1 && dstride_h == 1 &&
                                      ostride_w == 1 && dstride_w == 1 && S <= Winofilter;

    const size_t granulated_S = single_traverse_mode
                                ? RoundUpToMultiple(S, Winofilter)
                                : RoundUpToMultiple(S, 2 * Winofilter);

    const size_t granulated_R = (ostride_h == 1 && dstride_h == 1) || (R % (2 * Winofilter) == 1)
                                ? RoundUpToMultiple(R, Winofilter)
                                : RoundUpToMultiple(R, 2 * Winofilter);

    const size_t granulated_C = single_traverse_mode
                                ? problem.IsFp16() ? RoundUpToMultiple(C, 4) : RoundUpToMultiple(C, 2)
                                : problem.IsFp16() ? RoundUpToMultiple(C, 2) : C;

    const auto granulated_OH = RoundUpToMultiple(OH, Winodata * dstride_h);
    const auto granulated_OW = RoundUpToMultiple(OW, Winodata * dstride_w);
    const auto NHW_tiles     = granulated_OH * granulated_OW * N / Winodata / Winodata;

    const size_t K_granularity         = dstride_h == 1 && dstride_w == 1 ? 32 : 16;
    const size_t NHW_tiles_granularity = dstride_h == 1 && dstride_w == 1 ? 32 : 64;
    const size_t NKHW_granularity      = K_granularity * NHW_tiles_granularity 
                                        * Winodata * Winodata
                                        / dstride_h / dstride_w;
    
    const auto n_works              = Ceil(K, K_granularity) * Ceil(NHW_tiles, NHW_tiles_granularity);
    const auto granulated_n_works   = Ceil(n_works, n_groups) * RoundUpToMultiple(static_cast<size_t>(G * n_groups), cu_count);

    const auto granulated_MACs  = granulated_n_works *  NKHW_granularity 
                                    * granulated_C * granulated_R * granulated_S;
    const auto direct_conv_MACs = static_cast<size_t>(N) * G * K * C 
                                    * Ceil(static_cast<size_t>(OH * R), dstride_h)
                                    * Ceil(static_cast<size_t>(OW * S), dstride_w);
    // clang-format on

    const float granularity_loss =
        1.0f - static_cast<float>(direct_conv_MACs) / static_cast<float>(granulated_MACs);

    MIOPEN_LOG_I2("granularity_loss=" << std::setprecision(2) << granularity_loss);
    if(granularity_loss < 0.0f || granularity_loss >= 1.0f)
        MIOPEN_LOG_E("Granularity loss must satisfy the interval [0,1)");

    return granularity_loss;
}

} // namespace

namespace fusion {

template <int Winodata, int Winofilter>
bool ConvBinWinogradRxSg1Fused<Winodata, Winofilter>::IsApplicable(
    const FusionContext& context, const FusionDescription& problem) const
{
    if(IS2X3 && miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1{}))
        return false;
    if(IS3X2 && miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2_G1{}))
        return false;

    if(!WinoCommonIsApplicable(context))
        return false;

    const miopen::ConvolutionContext conv_ctx =
        context.GetConvContext(0, miopen::conv::Direction::Forward, problem);

    if(!(conv_ctx.problem.IsFp32() || conv_ctx.problem.IsFp16()))
        return false;

    const std::string name = conv_ctx.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9") || StartsWith(name, "gfx10") || StartsWith(name, "gfx11")))
        return false;

    if(conv_ctx.problem.IsFp16() &&
       !(StartsWith(name, "gfx906") || StartsWith(name, "gfx908") || StartsWith(name, "gfx90a") ||
         StartsWith(name, "gfx1011") || StartsWith(name, "gfx1012") || StartsWith(name, "gfx103") ||
         StartsWith(name, "gfx11")))
        return false;

    // clang-format off
    if (!((conv_ctx.problem.kernel_stride_w == 1 || conv_ctx.problem.kernel_stride_w == 2)
        && conv_ctx.problem.kernel_stride_w == conv_ctx.problem.kernel_stride_h
        && conv_ctx.problem.kernel_dilation_w == 1
        && conv_ctx.problem.kernel_dilation_h == 1))
        return false;
    // clang-format on

    const auto group_count = conv_ctx.problem.conv_problem.GetGroupCount();
    if(group_count != 1)
        return false;

    const auto W  = conv_ctx.problem.conv_problem.GetInWidth();
    const auto H  = conv_ctx.problem.conv_problem.GetInHeight();
    const auto C  = conv_ctx.problem.conv_problem.GetInChannels();
    const auto N  = conv_ctx.problem.conv_problem.GetInBatchSize();
    const auto K  = conv_ctx.problem.conv_problem.GetOutChannels();
    const auto R  = conv_ctx.problem.conv_problem.GetWeightsHeight();
    const auto S  = conv_ctx.problem.conv_problem.GetWeightsWidth();
    const auto OH = conv_ctx.problem.conv_problem.GetOutHeight();
    const auto OW = conv_ctx.problem.conv_problem.GetOutWidth();

    return IsWinogradV21Preferred<Winodata, Winofilter>(name, conv_ctx.problem)
               ? IsShaderConstraintsMetV21(conv_ctx.problem, R, S, C, K, H, W, OH, OW, N)
               : IsShaderConstraintsMetV30(conv_ctx.problem, R, S, C, K, H, W, OH, OW, N);
}

template <int Winodata, int Winofilter>
ConvSolution
ConvBinWinogradRxSg1Fused<Winodata, Winofilter>::GetSolution(const FusionContext& context,
                                                             const FusionDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;

    const auto conv_ctx = context.GetConvContext(0, miopen::conv::Direction::Forward, problem);

    const int n_groups  = conv_ctx.GetStream().GetMaxHardwareComputeUnits();
    const auto name     = conv_ctx.GetStream().GetDeviceName();
    const auto is_gfx9  = StartsWith(name, "gfx9");
    const auto is_gfx10 = StartsWith(name, "gfx10");
    const auto is_v21   = IsWinogradV21Preferred<Winodata, Winofilter>(name, conv_ctx.problem);
    size_t wg_size      = is_gfx9 ? 512 : 256;
    kernel.g_wk.push_back(wg_size * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});
    kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");

    const std::string kernel_version = is_v21 ? "_v21_1_3" : "_v30_2_6";
    std::string kernel_file          = "Conv_Winograd" + kernel_version;
    std::string kernel_name          = "miopenSp3AsmConv" + kernel_version;
    std::string kernel_postfix;

    if(is_gfx9)
    {
        kernel_name += "_gfx9";
    }
    else if(is_gfx10)
    {
        kernel_name += "_gfx10";
    }
    else // if (is_gfx11)
    {
        kernel_name += "_gfx11";
    }

    if(conv_ctx.problem.IsFp32())
    {
        kernel_name += "_fp32";
        kernel_file += "_fp32";
    }
    else // if(conv_ctx.problem.IsFp16())
    {
        kernel_name += is_gfx9 ? "_fp16_dot2_edc" : "_fp16_dot2";
        kernel_file += "_fp16_dot2";
    }

    kernel_postfix += IS2X3 ? "_f2x3" : "_f3x2";
    kernel_postfix += "_stride" + std::to_string(conv_ctx.problem.kernel_stride_h);

    kernel.kernel_name += kernel_name + kernel_postfix;
    kernel.kernel_file += kernel_file + kernel_postfix + ".s";
    result.construction_params.push_back(kernel);

    const auto& desc    = *problem.fusion_plan_desc;
    const int bias_idx  = GetOpIdx(desc.op_map, miopenFusionOpBiasForward);
    const int activ_idx = GetOpIdx(desc.op_map, miopenFusionOpActivForward);
    int N, C, H, W, K, unused, out_H, out_W, R, S, pad_H, pad_W;
    GetCompiledInParameters(context,
                            conv_ctx.problem,
                            &N,
                            &C,
                            &H,
                            &W,
                            &K,
                            &unused,
                            &out_H,
                            &out_W,
                            &R,
                            &S,
                            &pad_H,
                            &pad_W);

    const auto granularity_loss =
        GetGranularityLoss<Winodata, Winofilter>(conv_ctx.problem,
                                                 R,
                                                 S,
                                                 C,
                                                 K,
                                                 out_H,
                                                 out_W,
                                                 N,
                                                 1, /* non-grouped convolution */
                                                 n_groups,
                                                 conv_ctx.GetStream().GetMaxHardwareComputeUnits());
    result.weight = (1 - granularity_loss) * 100.0f;

    const int zero = 0;
    int flags      = [&]() {
        constexpr int L_F_BIAS       = 1 << 7;
        constexpr int L_F_LEAKY_RELU = 1 << 8;
        int flag                     = 0;

        if(bias_idx != -1)
            flag |= L_F_BIAS;
        if(activ_idx != -1)
            flag |= L_F_LEAKY_RELU;

        return flag;
    }();

    const miopenActivationMode_t activ_mode = [&]() {
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
                    if(activ_mode == miopenActivationLEAKYRELU)
                        return (static_cast<float>(activ_args.activAlpha));
                }
                return static_cast<float>(0.0);
            }();

            auto zero_u64 = static_cast<uint64_t>(0);
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
                          activ_alpha, // leaky relu alpha
                          zero,        // reserved2", Other, zero_int),
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
            );
        };
    };
    return result;
}

template struct ConvBinWinogradRxSg1Fused<2, 3>;
template struct ConvBinWinogradRxSg1Fused<3, 2>;

} // namespace fusion
} // namespace solver
} // namespace miopen
