/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <limits>
#include <cassert>
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/tensor.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/miopen.h>
#include <miopen/generic_search.hpp>
#include <miopen/conv/invokers/impl_gemm.hpp>

#include <boost/any.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#if MIOPEN_BACKEND_HIP

#define WORKAROUND_SWDEV_203031 1 // See also issues #2075, #2067
#endif

namespace miopen {
namespace solver {
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F4X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F5X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F6X3)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X4)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X5)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X6)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F4X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F5X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F6X3)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X4)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X5)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X6)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_WORKSPACE_MAX)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING)

// Introduces a number of shader-specific aliases (names) in the current scope at zero cost.
// These names represent shader parameters, e.g. shader C is batch_size etc and useful for
// programming.
#define DEFINE_GETXFORMHWSIZE(params)                                                         \
    const auto                                                                                \
        wino_xform_h =                                                                        \
            solver::ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoXformHWSize(params, 1),                                          \
        wino_xform_w =                                                                        \
            solver::ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoXformHWSize(params, 0);

#define DEFINE_SHADER_ALIASES_2(params, is_wrw)                                                    \
    const auto& group_cnt = (params).group_counts;                                                 \
    const int N           = (is_wrw) ? ((params).n_outputs / group_cnt) : (params).batch_sz;       \
    const int K   = (is_wrw) ? ((params).n_inputs / group_cnt) : ((params).n_outputs / group_cnt); \
    const int C   = (is_wrw) ? (params).batch_sz : ((params).n_inputs / group_cnt);                \
    const auto& R = (is_wrw) ? (params).in_height : (params).kernel_size_h;                        \
    const auto& S = (is_wrw) ? (params).in_width : (params).kernel_size_w;                         \
    const auto& H = (is_wrw) ? (params).out_height : (params).in_height;                           \
    const auto& W = (is_wrw) ? (params).out_width : (params).in_width;                             \
    const auto& out_H = (is_wrw) ? (params).kernel_size_h : (params).out_height;                   \
    const auto& out_W = (is_wrw) ? (params).kernel_size_w : (params).out_width;

#define DEFINE_SHADER_ALIASES(params) \
    DEFINE_SHADER_ALIASES_2((params), (params).direction.IsBackwardWrW())

#if MIOPEN_BACKEND_HIP

#define DEFINE_GETDTILEHWSIZE(params)                                                         \
    const auto                                                                                \
        wino_dtile_h =                                                                        \
            solver::ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoDtileHWSize(params, 1),                                          \
        wino_dtile_w =                                                                        \
            solver::ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoDtileHWSize(params, 0);

#define DEFINE_SHADER_CONV_MOD_ALIASES_4(params, is_fwd, is_bwd, is_wrw) \
    const int stride_h     = (is_fwd) ? (params).kernel_stride_h : 1;    \
    const int stride_w     = (is_fwd) ? (params).kernel_stride_w : 1;    \
    const int f_dilation_h = (is_wrw) ? (params).kernel_stride_h : 1;    \
    const int f_dilation_w = (is_wrw) ? (params).kernel_stride_w : 1;    \
    const int d_dilation_h = (is_bwd) ? (params).kernel_stride_h : 1;    \
    const int d_dilation_w = (is_bwd) ? (params).kernel_stride_w : 1;

#define DEFINE_SHADER_CONV_MOD_ALIASES(params)                            \
    DEFINE_SHADER_CONV_MOD_ALIASES_4((params),                            \
                                     (params).direction.IsForward(),      \
                                     (params).direction.IsBackwardData(), \
                                     (params).direction.IsBackwardWrW())

#define GENERATE_MAIN_OPTIONS(options)                             \
    GenerateClangDefsym((options), "ROCM_METADATA_VERSION", 5);    \
    GenerateClangDefsym((options), "xformx_o_size", WinoDataW);    \
    GenerateClangDefsym((options), "xformy_o_size", WinoDataH);    \
    GenerateClangDefsym((options), "xformx_x_size", wino_xform_w); \
    GenerateClangDefsym((options), "xformy_x_size", wino_xform_h); \
    GenerateClangDefsym((options), "xformx_d_size", wino_dtile_w); \
    GenerateClangDefsym((options), "xformy_d_size", wino_dtile_h); \
    GenerateClangDefsym((options), "xformx_f_size", WinoFilterW);  \
    GenerateClangDefsym((options), "xformy_f_size", WinoFilterH);  \
    GenerateClangDefsym((options), "fdilation_w", f_dilation_w);   \
    GenerateClangDefsym((options), "fdilation_h", f_dilation_h);   \
    GenerateClangDefsym((options), "stride_h", stride_h);          \
    GenerateClangDefsym((options), "stride_w", stride_w);          \
    GenerateClangDefsym((options), "ddilation_h", d_dilation_h);   \
    GenerateClangDefsym((options), "ddilation_w", d_dilation_w);

struct WinoOffsets
{
    const size_t in, out, wei;
    WinoOffsets(size_t in_size, size_t out_size) : in(0), out(in_size), wei(in_size + out_size) {}
};

BuffInfo GetNormalBuffer(const ConvolutionContext& params, const ConvWinoBuffType buff_type)
{
    DEFINE_SHADER_ALIASES(params)
    const bool is_wrw = (params).direction.IsBackwardWrW();
    const bool is_fwd = (params).direction.IsForward();
    // clang-format off
    if(buff_type == ConvWinoBuffType::Input)
        return {
            GetGroupConvLayout(
                is_wrw 
                ? GetSwappedNCLayout(GetMemLayout_t(params.in_layout)) 
                : GetMemLayout_t(params.in_layout),
                true),
            N,
            C,
            H,
            W,
            group_cnt,
            static_cast<int>(GetTypeSize(params.in_data_type))};

    if(buff_type == ConvWinoBuffType::Output)
        // cppcheck-suppress unreadVariable
        return {
            is_wrw 
            ? GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), false)
            : GetGroupConvLayout(GetMemLayout_t(params.out_layout), true),
            N,
            K,
            out_H,
            out_W,
            group_cnt,
            static_cast<int>(GetTypeSize(params.out_data_type))};

    // ConvWinoBuffType::Weight
    // cppcheck-suppress unreadVariable
    return {
        (is_wrw 
            ? GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(params.out_layout)), true)
            : GetGroupConvLayout(
                is_fwd
                ? (MemLayout_t::NCHW)
                : GetSwappedNCLayout(MemLayout_t::NCHW),
                false)
            ),
        K,
        C,
        R,
        S,
        group_cnt,
        static_cast<int>(GetTypeSize(params.weights_data_type))};
    // clang-format on
}
#endif

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
GetWinoBuffer(const ConvolutionContext& params,
              const ConvWinoBuffType buff_type,
              const miopenDataType_t transform_data_type)
{
    DEFINE_GETXFORMHWSIZE(params)
    DEFINE_SHADER_ALIASES(params)

    WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW> Transform_info(
        N,
        K,
        C,
        group_cnt,
        out_H,
        out_W,
        R,
        S,
        (MemLayout_t::GCNHW),
        (ConvWinoXformType::N_GXhXw_C_Th_Tw),
        GetTypeSize(transform_data_type),
        buff_type,
        wino_xform_h,
        wino_xform_w);

    (void)H;
    (void)W;
    return Transform_info;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
inline bool IsApplicableGEMM(const ConvolutionContext& params)
{
#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)

    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;

    // int offset for Workspace buffers.
    return !(((GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                   params, ConvWinoBuffType::Input, transform_data_type))
                      .buff_info.total_byte_size /
                  GetTypeSize(transform_data_type) +
              (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                   params, ConvWinoBuffType::Output, transform_data_type))
                      .buff_info.total_byte_size /
                  GetTypeSize(transform_data_type)) >= (1LL << 31));
#else
    (void)params;
    return false;
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
inline bool IsApplicableTransform(const ConvolutionContext& params)
{
#if MIOPEN_BACKEND_HIP
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;
    if(!params.Is2d())
        return false;

    if(!(params.IsFp32() || params.IsFp16()))
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9")))
        return false;

    {
        std::size_t limit = miopen::Value(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_WORKSPACE_MAX{});
#if WORKAROUND_SWDEV_203031
        if(limit == 0)
        {
            if(name == "gfx900" ||
               (name == "gfx906" && params.GetStream().GetMaxComputeUnits() <= 60))
                limit = 2000000000ULL; // ~1.862 GiB
            else
                limit = std::numeric_limits<std::size_t>::max();
        }
#else
        if(limit == 0)
            limit = std::numeric_limits<std::size_t>::max();
#endif
        if(limit != std::numeric_limits<std::size_t>::max())
        {
            const auto required =
                ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}
                    .GetWorkspaceSize(params);
            MIOPEN_LOG_I2("Workspace required: " << required << ", limit: " << limit);
            if(required > limit)
                return false;
        }
    }

    if(!params.IsLayoutDefault())
    {
        return false;
    }

    DEFINE_GETXFORMHWSIZE(params)
    DEFINE_GETDTILEHWSIZE(params)
    DEFINE_SHADER_CONV_MOD_ALIASES(params)
    // cppcheck-suppress redundantCondition
    if(wino_xform_h > 8 || wino_xform_w > 8 || wino_dtile_h > 8 || wino_dtile_w > 8 ||
       WinoDataH > 8 || WinoFilterH > 8 || WinoDataW > 8 || WinoFilterW > 8)
    {
        return false;
    }

    if(params.direction.IsBackwardData() &&
       (WinoDataW % d_dilation_w != 0 || WinoDataH % d_dilation_h != 0))
    {
        return false;
    }

    {
        unsigned int const waves_in_group = 512 / wave_size;
        unsigned int const tiles_per_wave = 8;
        auto const tiles_per_group        = waves_in_group * tiles_per_wave / 2;
        auto const n_groups               = params.GetStream().GetMaxComputeUnits();
        auto const tiles_step             = tiles_per_group * n_groups;
        if(tiles_step >= std::pow(2, 16))
            return false;
    }

    BuffInfo in_buff  = GetNormalBuffer(params, ConvWinoBuffType::Input),
             out_buff = GetNormalBuffer(params, ConvWinoBuffType::Output),
             wei_buff = GetNormalBuffer(params, ConvWinoBuffType::Weight);
    {
        const miopenDataType_t transform_data_type =
            miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
                ? params.in_data_type
                : miopenFloat;

        auto wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            params, ConvWinoBuffType::Input, transform_data_type);
        auto wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            params, ConvWinoBuffType::Output, transform_data_type);
        auto wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            params, ConvWinoBuffType::Weight, transform_data_type);

        if(in_buff.total_byte_size > std::pow(2, 31) ||
           wei_buff.total_byte_size > std::pow(2, 31) ||
           out_buff.total_byte_size > std::pow(2, 31) ||
           wino_in.buff_info.total_byte_size > std::pow(2, 31) ||
           wino_out.buff_info.total_byte_size > std::pow(2, 31) ||
           wino_wei.buff_info.total_byte_size > std::pow(2, 31))
            return false;

        if(params.direction.IsBackwardWrW())
        {
            if(wino_out.buff_info.size.h != 1 || wino_out.buff_info.size.w != 1)
                return false;
        }
        else if(wino_wei.buff_info.size.h != 1 || wino_wei.buff_info.size.w != 1)
            return false;
    }

    // clang-format off
    bool ok = (
        stride_h == stride_w
        && f_dilation_h == f_dilation_w
        && d_dilation_h == d_dilation_w
        && (stride_h * f_dilation_h * d_dilation_h <= 2) //for future
        && (stride_w * f_dilation_w * d_dilation_w <= 2)
        && params.kernel_dilation_w == params.kernel_dilation_h
        && params.kernel_dilation_h == 1
        && in_buff.size.nk < std::pow(2, 16)
        && in_buff.size.c < std::pow(2, 16)
        && wei_buff.size.nk < std::pow(2, 16)
        && out_buff.size.h < std::pow(2, 16)
        && out_buff.size.w < std::pow(2, 16)
        && in_buff.size.g == 1
        && params.bias == 0
        && params.in_layout == "NCHW");
    // clang-format on
    return ok;
#else
    (void)params;
    return false;
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
bool ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(
    const ConvolutionContext& params) const
{
    // HIP backend required for sending ptr (buffer + offset)
    // ROCBLAS for GEMM step

    if(!IsApplicableGEMM<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params))
        return false;

    static const int wino_data_tile   = std::max(WinoDataH, WinoDataW);
    static const int wino_filter_tile = std::max(WinoFilterH, WinoFilterW);

    if(!params.direction.IsBackwardWrW())
    {
        if(wino_filter_tile != 3)
            return false;

        if(wino_data_tile == 6)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F6X3{}))
                return false;
        if(wino_data_tile == 5)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F5X3{}))
                return false;
        if(wino_data_tile == 4)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F4X3{}))
                return false;
        if(wino_data_tile == 3)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F3X3{}))
                return false;
        if(wino_data_tile == 2)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_F2X3{}))
                return false;
    }
    else
    {
        if(wino_data_tile != 3)
            return false;

        if(wino_filter_tile == 6)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X6{}))
                return false;
        if(wino_filter_tile == 5)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X5{}))
                return false;
        if(wino_filter_tile == 4)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X4{}))
                return false;
        if(wino_filter_tile == 3)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X3{}))
                return false;
        if(wino_filter_tile == 2)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_WINOGRAD_F3X2{}))
                return false;
    }

    return IsApplicableTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params);
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
size_t ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetWorkspaceSize(
    const ConvolutionContext& params) const
{
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;

    return (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                params, ConvWinoBuffType::Input, transform_data_type))
               .buff_info.total_byte_size +
           (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                params, ConvWinoBuffType::Output, transform_data_type))
               .buff_info.total_byte_size +
           (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                params, ConvWinoBuffType::Weight, transform_data_type))
               .buff_info.total_byte_size;
}

conv::DataInvokeParams GetFWDCtxFromAnyInvokeParams(const AnyInvokeParams& ctx, bool isWrW)
{
    if(isWrW)
    {
        const auto& invoke_params = ctx.CastTo<conv::WrWInvokeParams>();
        return conv::DataInvokeParams{{invoke_params.tensors.xDesc,
                                       invoke_params.tensors.x,
                                       invoke_params.tensors.dyDesc,
                                       invoke_params.tensors.dy,
                                       invoke_params.tensors.dwDesc,
                                       invoke_params.tensors.dw},
                                      invoke_params.workSpace,
                                      invoke_params.workSpaceSize};
    }
    else
    {
        return ctx.CastTo<conv::DataInvokeParams>();
    }
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
InvokerFactory MakeWinogradInvokerFactory(const ConvolutionContext& params,
                                          InvokerFactory xdlops_factory = InvokerFactory(),
                                          bool isXdlops                 = false)
{
#if MIOPEN_BACKEND_HIP
    const int pad_H = params.direction.IsBackwardData() ? params.GetBackwardPadH() : params.pad_h;
    const int pad_W = params.direction.IsBackwardData() ? params.GetBackwardPadW() : params.pad_w;
    const int n_groups           = params.GetStream().GetMaxComputeUnits();
    const int L_F_SWAP_NC        = 1;
    const int L_F_SINGLE_OTILE_W = 2;
    const int L_F_SINGLE_OTILE_H = 4;
    DEFINE_GETXFORMHWSIZE(params)

    bool is_wrw = (params).direction.IsBackwardWrW();

    BuffInfo in_buff      = GetNormalBuffer(params, ConvWinoBuffType::Input),
             out_buff     = GetNormalBuffer(params, ConvWinoBuffType::Output),
             weights_buff = GetNormalBuffer(params, ConvWinoBuffType::Weight);

    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;
    auto wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Input, transform_data_type);
    auto wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Output, transform_data_type);
    auto wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Weight, transform_data_type);

    const WinoOffsets transform_offset(wino_in.buff_info.total_byte_size,
                                       wino_out.buff_info.total_byte_size);

    InvokerFactory gemm_conv_factory;
    std::string gemm_conv_kernel_name;

    auto zeroDesc = TensorDescriptor();
    if(isXdlops)
    {
        gemm_conv_kernel_name = "XDLOPS_CONV: ";
        gemm_conv_factory     = xdlops_factory;
    }
    else
    {
        // GEMM
        gemm_conv_kernel_name = "WINO_GEMM: ";
        // clang-format off
        int m = weights_buff.size.nk,
            k = in_buff.size.c *
                 (is_wrw
                    ? wino_in.wino_info.wino_tiles_HW[0] * wino_in.wino_info.wino_tiles_HW[1]
                    : 1),
            n   = is_wrw
                    ? in_buff.size.nk
                    : (wino_in.buff_info.size.nk * wino_in.buff_info.size.w
                        * wino_in.buff_info.size.h);
        int lda = is_wrw ? k : m,
            ldb = is_wrw ? k : n,
            ldc = n;
        int batch_count       = wino_xform_h * wino_xform_w * in_buff.size.g;
        long long int strideA = m * k * 1LL, strideB = k * n * 1LL, strideC = m * n * 1LL;
        float alpha = 1., beta = 0.0;
        const bool isColMajor = false,
            transA = is_wrw ? false : true,
            transB = is_wrw ? true: false;

        GemmDescriptor wino_gemm_desc{isColMajor,transA,transB,m,n,k,
            lda,ldb,ldc,batch_count,strideA,strideB,
            strideC,alpha,beta,transform_data_type};
        // clang-format on

        gemm_conv_factory = [=](const std::vector<Kernel>&) {

            return [=](const Handle& handle, const AnyInvokeParams& ctx) {
#if MIOPEN_USE_ROCBLAS
                const auto& data_ctx = ctx.CastTo<conv::DataInvokeParams>();
                Data_t workSpace     = data_ctx.workSpace;
                CallGemmStridedBatched(
                    handle,
                    wino_gemm_desc,
                    workSpace,
                    static_cast<int>(transform_offset.wei / wino_wei.buff_info.element_size),
                    workSpace,
                    static_cast<int>(transform_offset.in / wino_in.buff_info.element_size),
                    workSpace,
                    static_cast<int>(transform_offset.out / wino_out.buff_info.element_size),
                    nullptr,
                    false,
                    GemmBackend_t::rocblas);
#else
                (void)handle;
                (void)ctx;
                MIOPEN_THROW(miopenStatusBadParm, "ConvMPAnydirectWinograd is not supported ");
#endif
            };

        };
    }
    const int main_flags =
        params.direction.IsBackwardWrW() ? (L_F_SINGLE_OTILE_W | L_F_SINGLE_OTILE_H) : L_F_SWAP_NC;

    return [=](const std::vector<Kernel>& kernels) {

        const std::vector<Kernel> transform_kernels =
            std::vector<Kernel>{kernels[0], kernels[1], kernels[2]};

        const std::vector<Kernel> conv_kernels =
            isXdlops ? std::vector<Kernel>{kernels[3]} : std::vector<Kernel>{};

        auto gemm_conv_invoker = gemm_conv_factory(conv_kernels);

        return [=](const Handle& handle, const AnyInvokeParams& ctx) {
            const conv::DataInvokeParams data_ctx =
                GetFWDCtxFromAnyInvokeParams(ctx, params.direction.IsBackwardWrW());
            const auto& tensors       = data_ctx.tensors;
            Data_t workSpace          = data_ctx.workSpace;
            const auto& workSpaceSize = data_ctx.workSpaceSize;
            float total_time          = 0;
            auto wino_in_ptr =
                static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.in);
            auto wino_w_ptr =
                static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.wei);
            auto wino_out_ptr =
                static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.out);

            for(int i = 0, cur = 0; i < 4; i++)
            {
                std::string kernel_name;
                int flags = main_flags;
                if(i == 2) // GEMM
                {
                    // rocblas_gemm use workSpace pointer and constant offset
                    // xdlops_conv use tensors.in, tensors.w, tensors.out
                    ConvDataTensors xdlops_tensor = ConvDataTensors(ConvFwdTensors{
                        zeroDesc, wino_in_ptr, zeroDesc, wino_w_ptr, zeroDesc, wino_out_ptr});
                    const auto invoke_params =
                        conv::DataInvokeParams{xdlops_tensor, workSpace, workSpaceSize};

                    gemm_conv_invoker(handle, invoke_params);
                    kernel_name = gemm_conv_kernel_name;
                }
                else
                {
                    const auto kernel     = handle.Run(transform_kernels[cur++]);
                    const BuffInfo* d_buf = nullptr;
                    const BuffInfo* o_buf = nullptr;
                    void* buff_out_addr   = nullptr;

                    auto const_buff_in_adr = tensors.in;
                    auto buff_in_adr       = wino_out_ptr;
                    bool const_input       = false;
                    kernel_name            = kernel.GetName();

                    if(i == 0) // Input
                    {          // Transform
                        d_buf             = &in_buff;
                        o_buf             = &(wino_in.buff_info);
                        const_buff_in_adr = tensors.in;
                        buff_out_addr     = wino_in_ptr;
                        const_input       = true;
                    }
                    else if(i == 1) // filter
                    {               // Transform
                        d_buf             = &weights_buff;
                        o_buf             = &(wino_wei.buff_info);
                        const_buff_in_adr = tensors.w;
                        buff_out_addr     = wino_w_ptr;
                        const_input       = true;
                        if(isXdlops)
                            flags &= ~L_F_SWAP_NC;
                    }
                    else if(i == 3)
                    { // Output
                        d_buf         = &(out_buff);
                        o_buf         = &(wino_out.buff_info);
                        buff_in_adr   = wino_out_ptr;
                        buff_out_addr = tensors.out;
                        const_input   = false;
                        if(is_wrw)
                            flags |= L_F_SWAP_NC;
                    }

                    const auto input_ptr =
                        static_cast<const void*>(const_input ? const_buff_in_adr : buff_in_adr);
                    const auto output_ptr = buff_out_addr;
                    // clang-format off
                    MIOPEN_LOG_I2(" G=" << d_buf->size.g << " M1=" << d_buf->size.nk << " M0=" << d_buf->size.c 
                        << " H=" << d_buf->size.h << " W=" << d_buf->size.w << " n_groups=" << n_groups << " flags=" << flags
                        << " pad_H=" << pad_H << " pad_W=" << pad_W << " tiles_h=" << o_buf->size.h << " tiles_w=" << o_buf->size.w
                        << " stride_G=" << d_buf->byte_stride.g << " stride_M1=" << d_buf->byte_stride.nk 
                        << " stride_M0=" << d_buf->byte_stride.c << " stride_H=" << d_buf->byte_stride.h 
                        << " stride_W=" << d_buf->byte_stride.w);
                    // clang-format on
                    kernel(input_ptr,
                           output_ptr,
                           d_buf->size.g,
                           d_buf->size.nk,
                           d_buf->size.c,
                           d_buf->size.h,
                           d_buf->size.w,
                           n_groups,
                           flags,
                           pad_H,
                           pad_W,
                           o_buf->size.h,
                           o_buf->size.w,
                           d_buf->byte_stride.g,
                           d_buf->byte_stride.nk,
                           d_buf->byte_stride.c,
                           d_buf->byte_stride.h,
                           d_buf->byte_stride.w);
                }
                if(handle.IsProfilingEnabled())
                {
                    float cur_time = handle.GetKernelTime();
                    MIOPEN_LOG_I2(kernel_name << ": " << cur_time);

                    if(i < 3)
                        total_time += cur_time;
                    else
                        handle.AccumKernelTime(total_time);
                }
            }
        };
    };
#else
    (void)params;
    (void)xdlops_factory;
    (void)isXdlops;
    MIOPEN_THROW(miopenStatusBadParm, "ConvMPAnydirectWinograd is not supported ");
    return nullptr;
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvSolution ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolution(
    const ConvolutionContext& params) const
{
    ConvSolution result;
    result.workspce_sz = GetWorkspaceSize(params);
#if MIOPEN_BACKEND_HIP

    const int n_groups = params.GetStream().GetMaxComputeUnits();
    DEFINE_GETXFORMHWSIZE(params)
    const std::vector<size_t> l_wk{512, 1, 1};
    const size_t g_wk_0 = n_groups * l_wk[0];
    const std::vector<size_t> g_wk{g_wk_0, 1, 1};
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;
    std::ostringstream options_in;
    DEFINE_SHADER_CONV_MOD_ALIASES(params)
    DEFINE_GETDTILEHWSIZE(params)
    GENERATE_MAIN_OPTIONS(options_in)
    GenerateClangDefsym(options_in, "xform_mirror", 0);
    GenerateClangDefsym(options_in, "src_type", (params.IsFp32() ? 1 : 2));
    GenerateClangDefsym(options_in, "dst_type", (transform_data_type == miopenFloat ? 1 : 2));

    std::ostringstream options_filter;
    GENERATE_MAIN_OPTIONS(options_filter)
    GenerateClangDefsym(options_filter, "xform_mirror", params.direction.IsBackwardData());
    GenerateClangDefsym(options_filter, "src_type", (params.IsFp32() ? 1 : 2));
    GenerateClangDefsym(options_filter, "dst_type", (transform_data_type == miopenFloat ? 1 : 2));

    std::ostringstream options_out;
    GENERATE_MAIN_OPTIONS(options_out)
    GenerateClangDefsym(options_out, "xform_mirror", 0);
    GenerateClangDefsym(options_out, "src_type", (transform_data_type == miopenFloat ? 1 : 2));
    GenerateClangDefsym(options_out, "dst_type", (params.IsFp32() ? 1 : 2));

    KernelInfo InTransform{
        options_in.str(), l_wk, g_wk, GetSolverFileNames(0), GetSolverKernelNames(0),
    };

    KernelInfo FilterTransform{
        options_filter.str(), l_wk, g_wk, GetSolverFileNames(1), GetSolverKernelNames(1),
    };

    KernelInfo OutTransform{
        options_out.str(), l_wk, g_wk, GetSolverFileNames(2), GetSolverKernelNames(2),
    };

    result.construction_params.push_back(InTransform);
    result.construction_params.push_back(FilterTransform);
    result.construction_params.push_back(OutTransform);

    result.invoker_factory =
        MakeWinogradInvokerFactory<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params);

    return result;
#else
    (void)params;
    MIOPEN_THROW(miopenStatusBadParm, "ConvMPAnydirectWinograd is not supported ");
#endif
}
template struct ConvMPAnydirectWinograd<2, 3>;
template struct ConvMPAnydirectWinograd<3, 2>;
template struct ConvMPAnydirectWinograd<3, 3>;
template struct ConvMPAnydirectWinograd<4, 3>;
template struct ConvMPAnydirectWinograd<3, 4>;
template struct ConvMPAnydirectWinograd<5, 3>;
template struct ConvMPAnydirectWinograd<3, 5>;
template struct ConvMPAnydirectWinograd<6, 3>;
template struct ConvMPAnydirectWinograd<3, 6>;

// context transformation
// for winograd buffers calculation using xdlops_convolution
template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvolutionContext ConvMPAnydirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
    GetTransformedConvContext(const ConvolutionContext& ctx) const
{
    DEFINE_GETXFORMHWSIZE(ctx)
    int batch_count = wino_xform_h * wino_xform_w * ctx.group_counts;
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? ctx.in_data_type
            : miopenFloat;

    WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
        wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Input, transform_data_type),
        wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Output, transform_data_type),
        wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Weight, transform_data_type);

    // GNCHW -> GCNHW
    TensorDescriptor in, wei, out;
    if(!ctx.direction.IsBackwardWrW())
    {
        miopenSet4dTensorDescriptor(&in,
                                    transform_data_type,
                                    1,
                                    wino_in.buff_info.size.c * batch_count,
                                    1,
                                    wino_in.buff_info.size.w * wino_in.buff_info.size.h *
                                        wino_in.buff_info.size.nk);

        miopenSet4dTensorDescriptor(&wei,
                                    transform_data_type,
                                    wino_wei.buff_info.size.nk * batch_count,
                                    wino_wei.buff_info.size.c,
                                    wino_wei.buff_info.size.h,
                                    wino_wei.buff_info.size.w);

        miopenSet4dTensorDescriptor(&out,
                                    transform_data_type,
                                    1,
                                    wino_out.buff_info.size.c * batch_count,
                                    1,
                                    wino_out.buff_info.size.w * wino_out.buff_info.size.h *
                                        wino_out.buff_info.size.nk);
    }
    else
    {
        miopenSet4dTensorDescriptor(&in,
                                    transform_data_type,
                                    1,
                                    batch_count,
                                    wino_in.buff_info.size.nk,
                                    wino_in.buff_info.size.w * wino_in.buff_info.size.h *
                                        wino_in.buff_info.size.c);

        miopenSet4dTensorDescriptor(&wei,
                                    transform_data_type,
                                    wino_wei.buff_info.size.nk * batch_count,
                                    1,
                                    1,
                                    wino_wei.buff_info.size.c * wino_wei.buff_info.size.h *
                                        wino_wei.buff_info.size.w);

        miopenSet4dTensorDescriptor(&out,
                                    transform_data_type,
                                    1,
                                    wino_out.buff_info.size.c * batch_count,
                                    wino_out.buff_info.size.nk,
                                    wino_out.buff_info.size.w * wino_out.buff_info.size.h);
    }
    // default conv_desc.
    // pads{0,0}, stride{1,1}, dilation {1, 1}
    // trans_output_pads = {0, 0},  group_count = gem_batch_count
    ConvolutionDescriptor conv_desc({0, 0}, {1, 1}, {1, 1}, {0, 0}, batch_count);

    auto dir = conv::Direction::Forward;

    ConvolutionContext transformed_ctx(in, wei, out, conv_desc, dir, 0);
    transformed_ctx.ExecutionContext::operator=(ctx);

    transformed_ctx.SetupFloats();
    return transformed_ctx;
}

// must be same as invoke_params in Invoker
template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
conv::DataInvokeParams GetTransformedInvokeContext(const ConvolutionContext& ctx,
                                                   const AnyInvokeParams& invoke_ctx)
{
#if MIOPEN_BACKEND_HIP
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_ANYD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? ctx.in_data_type
            : miopenFloat;
    WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
        wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Input, transform_data_type),
        wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Output, transform_data_type),
        wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Weight, transform_data_type);

    // There are no unread variables, but the compiler still shows a warning.
    // cppcheck-suppress unreadVariable
    const WinoOffsets transform_offset(wino_in.buff_info.total_byte_size,
                                       wino_out.buff_info.total_byte_size);

    const auto& data_ctx = invoke_ctx.CastTo<conv::DataInvokeParams>();

    auto workSpace = data_ctx.workSpace;

    const auto wino_in_ptr =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.in);
    const auto wino_w_ptr =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.wei);
    const auto wino_out_ptr =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.out);

    const auto transform_workSpaceSize = wino_in.buff_info.total_byte_size +
                                         wino_wei.buff_info.total_byte_size +
                                         wino_out.buff_info.total_byte_size;

    const auto gemm_workSpaceSize = data_ctx.workSpaceSize - transform_workSpaceSize;
    const auto gemm_workSpace =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_workSpaceSize);
    const auto zeroDesc           = TensorDescriptor();
    ConvDataTensors xdlops_tensor = ConvDataTensors(
        ConvFwdTensors{zeroDesc, wino_in_ptr, zeroDesc, wino_w_ptr, zeroDesc, wino_out_ptr});
    return conv::DataInvokeParams{xdlops_tensor, gemm_workSpace, gemm_workSpaceSize};
#else
    (void)invoke_ctx;
    (void)ctx;
    MIOPEN_THROW(miopenStatusBadParm, "ConvMPAnydirectWinograd is not supported ");
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
bool ConvMPAnydirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(
    const ConvolutionContext& ctx) const
{

    static const int wino_data_tile   = std::max(WinoDataH, WinoDataW);
    static const int wino_filter_tile = std::max(WinoFilterH, WinoFilterW);

    if(!ctx.direction.IsBackwardWrW())
    {
        if(wino_filter_tile != 3)
            return false;

        if(wino_data_tile == 6)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F6X3{}))
                return false;
        if(wino_data_tile == 5)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F5X3{}))
                return false;
        if(wino_data_tile == 4)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F4X3{}))
                return false;
        if(wino_data_tile == 3)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F3X3{}))
                return false;
        if(wino_data_tile == 2)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_XDLOPS_WINOGRAD_F2X3{}))
                return false;
    }
    else
    {
        if(wino_data_tile != 3)
            return false;

        if(wino_filter_tile == 6)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X6{}))
                return false;
        if(wino_filter_tile == 5)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X5{}))
                return false;
        if(wino_filter_tile == 4)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X4{}))
                return false;
        if(wino_filter_tile == 3)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X3{}))
                return false;
        if(wino_filter_tile == 2)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_MP_ANYD_WRW_XDLOPS_WINOGRAD_F3X2{}))
                return false;
    }

    return IsApplicableTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(ctx) &&
           ConvHipImplicitGemmForwardV4R4Xdlops().IsApplicable(GetTransformedConvContext(ctx));
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvSolution
ConvMPAnydirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceImplicitGemmForwardV4R4Xdlops& config,
    bool) const
{
    ConvSolution wino_transform =
        ConvMPAnydirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}.GetSolution(ctx);

    const ConvolutionContext xdlops_conv_ctx = GetTransformedConvContext(ctx);

    ConvSolution xdlops_conv =
        ConvHipImplicitGemmForwardV4R4Xdlops{}.GetSolution(xdlops_conv_ctx, config);

    ConvSolution result;
    result.workspce_sz = wino_transform.workspce_sz + xdlops_conv.workspce_sz;

    assert(xdlops_conv.construction_params.size() == 1);

    // change transform layout
    // GCNHW -> GNCHW
    // std::ostringstream additional_options_wei;
    // GenerateClangDefsym(additional_options_wei, "L_F_SWAP_NC", 1);
    // wino_transform.construction_params[1].comp_options += additional_options_wei.str();

    result.construction_params.push_back(wino_transform.construction_params[0]);
    result.construction_params.push_back(wino_transform.construction_params[1]);
    result.construction_params.push_back(wino_transform.construction_params[2]);
    result.construction_params.push_back(xdlops_conv.construction_params[0]);

    result.invoker_factory =
        MakeWinogradInvokerFactory<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, xdlops_conv.invoker_factory.value(), true);

    return result;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
PerformanceImplicitGemmForwardV4R4Xdlops
ConvMPAnydirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::Search(
    const ConvolutionContext& ctx, const AnyInvokeParams& invoke_ctx) const
{
    const auto transformed_invoke_ctx =
        GetTransformedInvokeContext<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(ctx,
                                                                                    invoke_ctx);
    return ConvHipImplicitGemmForwardV4R4Xdlops().Search(GetTransformedConvContext(ctx),
                                                         transformed_invoke_ctx);
}

template struct ConvMPAnydirectWinograd_xdlops<2, 3>;
template struct ConvMPAnydirectWinograd_xdlops<3, 2>;
template struct ConvMPAnydirectWinograd_xdlops<3, 3>;
template struct ConvMPAnydirectWinograd_xdlops<4, 3>;
template struct ConvMPAnydirectWinograd_xdlops<3, 4>;
template struct ConvMPAnydirectWinograd_xdlops<5, 3>;
template struct ConvMPAnydirectWinograd_xdlops<3, 5>;
template struct ConvMPAnydirectWinograd_xdlops<6, 3>;
template struct ConvMPAnydirectWinograd_xdlops<3, 6>;

} // namespace solver
} // namespace miopen
