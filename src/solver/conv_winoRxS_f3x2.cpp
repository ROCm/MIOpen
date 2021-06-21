/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/env.hpp>
#include <miopen/kernel.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#include <boost/any.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2)

#define MAX_CU_LIMIT 512

namespace miopen {
namespace solver {

namespace {
/// \todo Consider re-using code from RxS_f2x3.
inline bool IsShaderContraintsMet(const int R,
                                  const int S,
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

    // clang-format off
    // Check implementation limits.
    return N < std::pow(2, 16)
        && C < std::pow(2, 16)
        && K < std::pow(2, 16)
        && H < std::pow(2, 16)
        && W < std::pow(2, 16)
        && OH < std::pow(2, 16)
        && OW < std::pow(2, 16)
        && params.pad_w < std::pow(2, 16)
        && params.pad_h < std::pow(2, 16)
        && S < std::pow(2, 16)
        && R < std::pow(2, 16)
        && grid_workgroup_count_x < std::pow(2, 16)
        && (C * H * W) <= std::pow(2, 28)
        && (OH * OW) <= std::pow(2, 23)
        && (K * OH * OW) <= std::pow(2, 28)
        && (K * R * S) <= std::pow(2, 28)
        && (C * R * S) <= std::pow(2, 28);
    // clang-format on
}

} // namespace

bool ConvBinWinogradRxSf3x2::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.Is2d())
        return false;
    if(!params.IsFp32())
        return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;
    if(!params.IsLayoutDefault())
        return false;

    const auto name = params.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9") || StartsWith(name, "gfx10")) || name == "gfx90a")
        return false;

    // clang-format off
    if (! (params.kernel_stride_w == 1
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0))
        return false;
    // clang-format on

    const auto n_inputs_per_group  = params.n_inputs / params.group_counts,
               n_outputs_per_group = params.n_outputs / params.group_counts;

    if(params.direction.IsBackwardWrW())
    {
        return IsShaderContraintsMet(params.in_height,
                                     params.in_width,
                                     params.batch_sz,    // N
                                     n_inputs_per_group, // K
                                     params.out_height,
                                     params.out_width,
                                     params.kernel_size_h,
                                     params.kernel_size_w,
                                     n_outputs_per_group, // C
                                     params);
    }
    else
    {
        return IsShaderContraintsMet(params.kernel_size_h, // RxS
                                     params.kernel_size_w,
                                     n_inputs_per_group,  // C
                                     n_outputs_per_group, // K
                                     params.in_height,    // HxW
                                     params.in_width,
                                     params.out_height, // OHxOW
                                     params.out_width,
                                     params.batch_sz, // N
                                     params);
    }
}

/// \todo Consider re-using code from RxS_f2x3.
ConvSolution ConvBinWinogradRxSf3x2::GetSolution(const ConvolutionContext& params) const
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool IsWarned;
    if(!IsWarned)
    {
        if(params.GetStream().GetMaxHardwareComputeUnits() > MAX_CU_LIMIT)
            MIOPEN_LOG_WE(SolverDbId(*this) << ": GPU has "
                                            << params.GetStream().GetMaxHardwareComputeUnits()
                                            << "CUs, but this solver supports max "
                                            << MAX_CU_LIMIT
                                            << "and thus may show sub-optimal performance.");
        IsWarned = true;
    }

    ConvSolution result;
    KernelInfo kernel;

    const int n_groups = params.GetStream().GetMaxHardwareComputeUnits();
    const auto name    = params.GetStream().GetDeviceName();
    const auto is_gfx9 = StartsWith(name, "gfx9");
    size_t wg_size     = is_gfx9 ? 512 : 256;

    kernel.g_wk.push_back(wg_size * n_groups * params.group_counts);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(wg_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    std::string kernel_name    = "miopenSp3AsmConv_v21_1_2";
    std::string kernel_file    = "Conv_Winograd_v21_1_2_f3x2";
    std::string kernel_postfix = params.IsFp32() ? "_fp32" : "_fp16_dot2_edc";

    if(is_gfx9)
    {
        kernel_name += "_gfx9";
    }
    else // if(StartsWith(name, "gfx10"))
    {
        kernel_name += "_gfx10";
        kernel.comp_options += std::string(" -mcumode -mwavefrontsize64");
    }

    if(params.kernel_stride_w == 1)
    {
        kernel_postfix += "_stride1";
    }

    kernel_postfix += "_group";
    kernel.kernel_name = kernel_name + "_f3x2" + kernel_postfix;
    kernel.kernel_file = kernel_file + kernel_postfix + ".s";

    result.construction_params.push_back(kernel);

    if(!params.direction.IsBackwardWrW())
    {
        const bool is_forward     = params.direction.IsForward();
        constexpr int F_REVERSE_R = 1 << 0;
        constexpr int F_REVERSE_S = 1 << 1;
        constexpr int F_FLIP_K_C  = 1 << 2;
        // These are not used yet. Nevertheless let's keep as a shader documentation.
        // constexpr int F_FLIP_DATA_N_C = 1 << 3; // Unsupported in f3x2.
        // constexpr int F_FLIP_OUT_N_K = 1 << 4; // Unsupported in f3x2.
        // constexpr int L_F_ADDR_INDIRECT  = 1 << 6;
        // constexpr int L_F_BIAS  = 1 << 7;
        // constexpr int L_F_LEAKY_RELU  = 1 << 8;
        constexpr int L_F_NKC_STRIDES   = 1 << 9;
        constexpr int L_F_GROUP_STRIDES = 1 << 10;
        // constexpr int L_F_FORCE_FILTER_TRAVERSE_MODE  = 1 << 11;
        // constexpr int L_F_FILTER_TRAVERSE_DUAL  = 1 << 12;
        // constexpr int L_F_TENSOR_OFFSETS  = 1 << 13;
        // constexpr int L_F_USE_EXTENDED_FLAGS_64  = 1 << 15;
        int reserved             = 0;
        uint64_t reserved_offset = 0;
        int* reserved_ptr        = nullptr;
        int ignore;

        int N, C, H, W, K, out_H, out_W, R, S, pad_H, pad_W;
        GetCompiledInParameters(
            params, &N, &C, &H, &W, &K, &ignore, &out_H, &out_W, &R, &S, &pad_H, &pad_W);
        const auto group_cnt = params.group_counts;
        C                    = C / group_cnt;
        K                    = K / group_cnt;
        int flags            = is_forward ? 0 : F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
        flags |= L_F_NKC_STRIDES + L_F_GROUP_STRIDES;

        // cppcheck-suppress unreadVariable
        BuffInfo d_buf(GetGroupConvLayout(GetMemLayout_t(params.in_layout), true),
                       N,
                       C,
                       H,
                       W,
                       group_cnt,
                       GetTypeSize(params.in_data_type)),
            // cppcheck-suppress unreadVariable
            o_buf(GetGroupConvLayout(GetMemLayout_t(params.out_layout), true),
                  N,
                  K,
                  out_H,
                  out_W,
                  group_cnt,
                  GetTypeSize(params.out_data_type)),
            // cppcheck-suppress unreadVariable
            f_buf(GetGroupConvLayout(is_forward ? (MemLayout_t::NCHW)
                                                : GetSwappedNCLayout(MemLayout_t::NCHW),
                                     false),
                  K,
                  C,
                  R,
                  S,
                  group_cnt,
                  GetTypeSize(params.weights_data_type));

        result.invoker_factory = [=](std::vector<Kernel> kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                const auto k         = handle.Run(kernels[0]);
                const auto& data_ctx = primitive_params.CastTo<conv::DataInvokeParams>();
                const auto& tensors  = data_ctx.tensors;

                // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                    << " d_buf.byte_stride.nk=" << d_buf.byte_stride.nk << " d_buf.byte_stride.c=" << d_buf.byte_stride.c
                    << " d_buf.byte_stride.h=" << d_buf.byte_stride.h << " d_buf.byte_stride.w=" << d_buf.byte_stride.w
                    << " f_buf.byte_stride.nk=" << f_buf.byte_stride.nk << " f_buf.byte_stride.c=" << f_buf.byte_stride.c
                    << " f_buf.byte_stride.h=" << f_buf.byte_stride.h << " f_buf.byte_stride.w=" << f_buf.byte_stride.w
                    << " o_buf.byte_stride.nk=" << o_buf.byte_stride.nk << " o_buf.byte_stride.c=" << o_buf.byte_stride.c
                    << " o_buf.byte_stride.h="  << o_buf.byte_stride.h <<  " o_buf.byte_stride.w=" << o_buf.byte_stride.w
                    << " d_buf.byte_stride.g=" << d_buf.byte_stride.g  << " o_buf.byte_stride.g="  << o_buf.byte_stride.g
                    << " f_buf.byte_stride.g=" << f_buf.byte_stride.g); // clang-format on

                k(N,
                  C,
                  H,
                  W,
                  K,
                  n_groups,
                  flags,
                  reserved,
                  tensors.in,
                  tensors.w,
                  tensors.out,
                  reserved_ptr, // Unused return_addr.
                  R,
                  S,
                  pad_H, // Like Fwd wino.
                  pad_W,
                  out_H,
                  out_W,
                  reserved_ptr,    // Unused bias_addr.
                  reserved,        // Unused relu_alpha.
                  reserved,        // Unused reserved2.
                  reserved_offset, // Unused d_offset.
                  reserved_offset, // Unused f_offset.
                  reserved_offset, // Unused o_offset.
                  reserved_offset, // Unused b_offset.
                  d_buf.byte_stride.nk,
                  d_buf.byte_stride.c,
                  d_buf.byte_stride.h,
                  d_buf.byte_stride.w,
                  f_buf.byte_stride.nk,
                  f_buf.byte_stride.c,
                  f_buf.byte_stride.h,
                  f_buf.byte_stride.w,
                  o_buf.byte_stride.nk,
                  o_buf.byte_stride.c,
                  o_buf.byte_stride.h,
                  o_buf.byte_stride.w,
                  group_cnt,
                  d_buf.byte_stride.g,
                  f_buf.byte_stride.g,
                  o_buf.byte_stride.g);
            };
        };
    }
    else
    {
        int unused = 0;
        int N, C, H, W, K, out_H, out_W, R, S;
        GetCompiledInParameters(
            params, &C, &K, &R, &S, &N, &unused, &H, &W, &out_H, &out_W, &unused, &unused);
        const auto group_cnt             = params.group_counts;
        static const int F_NKC_STRIDES   = 1 << 9;
        static const int F_GROUP_STRIDES = 1 << 10;
        int flags                        = F_NKC_STRIDES + F_GROUP_STRIDES;
        N                                = N / group_cnt;
        K                                = K / group_cnt;
        int pad_H                        = params.conv_problem.GetConv().GetConvPads()[0];
        int pad_W                        = params.conv_problem.GetConv().GetConvPads()[1];

        BuffInfo d_buf(
            GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(params.in_layout)), true),
            N,
            C,
            H,
            W,
            group_cnt,
            GetTypeSize(params.in_data_type)),
            o_buf(GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(params.out_layout)), false),
                  N,
                  K,
                  out_H,
                  out_W,
                  group_cnt,
                  GetTypeSize(params.out_data_type)),
            f_buf(GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true),
                  K,
                  C,
                  R,
                  S,
                  group_cnt,
                  GetTypeSize(params.weights_data_type));

        decltype(auto) batch_sz = params.batch_sz;
        decltype(auto) n_inputs = params.n_inputs;

        result.invoker_factory = [=](std::vector<Kernel> kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                decltype(auto) invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();
                const auto& tensors          = invoke_params.tensors;

                // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                    << " d_buf.byte_stride.nk=" << d_buf.byte_stride.nk << " d_buf.byte_stride.c=" << d_buf.byte_stride.c
                    << " d_buf.byte_stride.h=" << d_buf.byte_stride.h << " d_buf.byte_stride.w=" << d_buf.byte_stride.w
                    << " f_buf.byte_stride.nk=" << f_buf.byte_stride.nk << " f_buf.byte_stride.c=" << f_buf.byte_stride.c
                    << " f_buf.byte_stride.h=" << f_buf.byte_stride.h << " f_buf.byte_stride.w=" << f_buf.byte_stride.w
                    << " o_buf.byte_stride.nk=" << o_buf.byte_stride.nk << " o_buf.byte_stride.c=" << o_buf.byte_stride.c
                    << " o_buf.byte_stride.h="  << o_buf.byte_stride.h <<  " o_buf.byte_stride.w=" << o_buf.byte_stride.w
                    << " d_buf.byte_stride.g=" << d_buf.byte_stride.g  << " o_buf.byte_stride.g="  << o_buf.byte_stride.g
                    << " f_buf.byte_stride.g=" << f_buf.byte_stride.g); // clang-format on
                MIOPEN_LOG_I2(" ctx.batch_sz=" << batch_sz << "ctx.n_inputs=" << n_inputs);

                int reserved             = 0;
                uint64_t reserved_offset = 0;
                int* reserved_ptr        = nullptr;

                handle.Run(kernels[0])(N,
                                       C,
                                       H,
                                       W,
                                       K,
                                       n_groups,
                                       flags,
                                       reserved,
                                       tensors.x,
                                       tensors.dy,
                                       tensors.dw,
                                       reserved_ptr, // Unused return_addr.
                                       R,
                                       S,
                                       pad_H,
                                       pad_W,
                                       out_H,
                                       out_W,
                                       reserved_ptr,    // Unused bias_addr.
                                       reserved,        // Unused relu_alpha.
                                       reserved,        // Unused reserved2.
                                       reserved_offset, // Unused d_offset.
                                       reserved_offset, // Unused f_offset.
                                       reserved_offset, // Unused o_offset.
                                       reserved_offset, // Unused b_offset.
                                       d_buf.byte_stride.nk,
                                       d_buf.byte_stride.c,
                                       d_buf.byte_stride.h,
                                       d_buf.byte_stride.w,
                                       f_buf.byte_stride.nk,
                                       f_buf.byte_stride.c,
                                       f_buf.byte_stride.h,
                                       f_buf.byte_stride.w,
                                       o_buf.byte_stride.nk,
                                       o_buf.byte_stride.c,
                                       o_buf.byte_stride.h,
                                       o_buf.byte_stride.w,
                                       group_cnt,
                                       d_buf.byte_stride.g,
                                       f_buf.byte_stride.g,
                                       o_buf.byte_stride.g);
            };
        };
    }

    return result;
}

} // namespace solver
} // namespace miopen
