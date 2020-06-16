/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include <miopen/stringutils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/tensor.hpp>
#include <miopen/solver.hpp>
#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)
#define WORKAROUND_SWDEV_203031 1 // See also issues #2075, #2067
#define WORKAROUND_SWDEV_234193 1
#endif

namespace miopen {
namespace solver {
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX)

// Introduces a number of shader-specific aliases (names) in the current scope at zero cost.
// These names represent shader parameters, e.g. shader C is batch_size etc and useful for
// programming.
#define DEFINE_GETXFORMHWSIZE(params)                                                             \
    const auto                                                                                    \
        wino_xform_h =                                                                            \
            solver::ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoXformHWSize(params, 0),                                              \
        wino_xform_w =                                                                            \
            solver::ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoXformHWSize(params, 1);

#define DEFINE_SHADER_ALIASES(params)           \
    const auto& C     = (params).batch_sz;      \
    const auto& N     = (params).n_outputs;     \
    const auto& K     = (params).n_inputs;      \
    const auto& out_H = (params).kernel_size_h; \
    const auto& out_W = (params).kernel_size_w; \
    const auto& R     = (params).in_height;     \
    const auto& S     = (params).in_width;      \
    const auto& H     = (params).out_height;    \
    const auto& W     = (params).out_width;     \
    DEFINE_GETXFORMHWSIZE(params)

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
struct InTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        const WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW> wino_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            GetSwappedNCLayout(MemLayout_t::HWCN),
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Input,
            wino_xform_h,
            wino_xform_w);

        const size_t u16limit = 1 << 16;
        const size_t tiles_per_wave =
            wave_size / (wino_xform_h > wino_xform_w ? wino_xform_h : wino_xform_w);
        // clang-format off
        const size_t chw_step       = tiles_per_wave
            * params.GetStream().GetMaxComputeUnits()
            * ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetGroupCountMult();
        const std::string name = params.GetStream().GetDeviceName();
        if(name.find("gfx8") != std::string::npos)
        {
            return false;
        }

        return (params.IsFp32() || params.IsFp16() || params.IsBfp16())
                && params.Is2d()
                && H < u16limit
                && W < u16limit
                && wino_info.wino_c < (1<<30)
                && N < u16limit
                && chw_step < u16limit
                && params.pad_h <= 3
                && params.pad_w <= 3;
        // clang-format on
    }
    static KernelInfo GetKernel(const ConvolutionContext& params)
    {
        DEFINE_GETXFORMHWSIZE(params)

        const std::vector<size_t> l_wk{wave_size, 1, 1};
        const size_t g_wk_0 =
            params.GetStream().GetMaxComputeUnits() *
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetGroupCountMult() *
            l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};
        std::ostringstream options;
        GenerateClangDefsym(options, "acc_type", 1);
        GenerateClangDefsym(options, "buf_type", (params.IsFp32() ? 1 : (params.IsFp16() ? 2 : 3)));
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", params.rmv.UseV3() ? 5 : 4);
        GenerateClangDefsym(options, "xformx_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformy_o_size", WinoDataH);
        GenerateClangDefsym(options, "xformx_d_size", wino_xform_w);
        GenerateClangDefsym(options, "xformy_d_size", wino_xform_h);
        GenerateClangDefsym(options, "xformx_f_size", WinoFilterW);
        GenerateClangDefsym(options, "xformy_f_size", WinoFilterH);
        GenerateClangDefsym(options, "fdilation_w", params.kernel_stride_w);
        GenerateClangDefsym(options, "fdilation_h", params.kernel_stride_h);

        GenerateClangDefsym(options, "MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16);

        return KernelInfo{
            options.str(),
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetSolverFileNames(0),
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetSolverKernelNames(0),
        };
    }
    static size_t GetBufferSize(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)
        const WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW> in_transform_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            MemLayout_t::HWNC,
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Input,
            wino_xform_h,
            wino_xform_w);
        (void)H;
        (void)W;
        return in_transform_info.buff_info.total_byte_size;
    }
};

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
struct FilterTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        const WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW> wino_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            GetSwappedNCLayout(MemLayout_t::HWCN),
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Input,
            wino_xform_h,
            wino_xform_w);

        const size_t u16limit = 1 << 16;
        const size_t tiles_per_wave =
            wave_size / wino_xform_h > wino_xform_w ? wino_xform_h : wino_xform_w;
        // clang-format off
        const size_t chw_step       = tiles_per_wave
            * params.GetStream().GetMaxComputeUnits()
            * ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetGroupCountMult();
        const std::string name = params.GetStream().GetDeviceName();
        if(name.find("gfx8") != std::string::npos)
        {
            return false;
        }
        return (params.IsFp32() || params.IsFp16() || params.IsBfp16())
                && params.Is2d()
                && H < u16limit
                && W < u16limit
                && wino_info.wino_c < (1<<30)
                && K < u16limit
                && chw_step < u16limit
                && params.pad_h <= 3
                && params.pad_w <= 3;
        // clang-format on
    }
    static KernelInfo GetKernel(const ConvolutionContext& params)
    {
        DEFINE_GETXFORMHWSIZE(params)

        const std::vector<size_t> l_wk{wave_size, 1, 1};
        const size_t g_wk_0 =
            params.GetStream().GetMaxComputeUnits() *
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetGroupCountMult() *
            l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};

        std::ostringstream options;
        GenerateClangDefsym(options, "acc_type", 1);
        GenerateClangDefsym(options, "buf_type", (params.IsFp32() ? 1 : (params.IsFp16() ? 2 : 3)));
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", params.rmv.UseV3() ? 5 : 4);
        GenerateClangDefsym(options, "MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16);
        GenerateClangDefsym(options, "xformx_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformy_o_size", WinoDataH);
        GenerateClangDefsym(options, "xformx_d_size", wino_xform_w);
        GenerateClangDefsym(options, "xformy_d_size", wino_xform_h);
        GenerateClangDefsym(options, "xformx_f_size", WinoFilterW);
        GenerateClangDefsym(options, "xformy_f_size", WinoFilterH);
        GenerateClangDefsym(options, "fdilation_w", params.kernel_stride_w);
        GenerateClangDefsym(options, "fdilation_h", params.kernel_stride_h);
        return KernelInfo{
            options.str(),
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetSolverFileNames(1),
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetSolverKernelNames(1),
        };
    }
    static size_t GetBufferSize(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        const WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
            filter_transform_info(N,
                                  K,
                                  C,
                                  out_H,
                                  out_W,
                                  R,
                                  S,
                                  MemLayout_t::HWNC,
                                  1,
                                  GetTypeSize(params.in_data_type),
                                  ConvWinoBuffType::Weight,
                                  wino_xform_h,
                                  wino_xform_w);
        (void)H;
        (void)W;
        return filter_transform_info.buff_info.total_byte_size;
    }
};

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
struct OutTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        return (params.IsFp32() || params.IsFp16() || params.IsBfp16()) && params.Is2d();
    }
    static KernelInfo GetKernel(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        auto dwords_per_ld = 1;
        const std::vector<size_t> l_wk{wave_size, 1, 1};
        auto ceil_val       = dwords_per_ld * l_wk[0];
        const size_t g_wk_0 = ((N * K + ceil_val - 1) / ceil_val) * l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};

        (void)H;
        (void)W;
        (void)C;
        (void)out_H;
        (void)out_W;
        (void)R;
        (void)S;

        std::ostringstream options;
        GenerateClangDefsym(options, "acc_type", 1);
        GenerateClangDefsym(options, "buf_type", (params.IsFp32() ? 1 : (params.IsFp16() ? 2 : 3)));
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", params.rmv.UseV3() ? 5 : 4);
        GenerateClangDefsym(options, "MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16);
        GenerateClangDefsym(options, "xformx_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformy_o_size", WinoDataH);
        GenerateClangDefsym(options, "xformx_d_size", wino_xform_w);
        GenerateClangDefsym(options, "xformy_d_size", wino_xform_h);
        GenerateClangDefsym(options, "xformx_f_size", WinoFilterW);
        GenerateClangDefsym(options, "xformy_f_size", WinoFilterH);
        GenerateClangDefsym(options, "fdilation_w", params.kernel_stride_w);
        GenerateClangDefsym(options, "fdilation_h", params.kernel_stride_h);

        return KernelInfo{
            options.str(),
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetSolverFileNames(2),
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
                GetSolverKernelNames(2),
        };
    }
    static size_t GetBufferSize(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        const WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW> OutTransform_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            GetSwappedNCLayout(MemLayout_t::HWNC),
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Output,
            wino_xform_h,
            wino_xform_w);
        (void)H;
        (void)W;
        return OutTransform_info.buff_info.total_byte_size;
    }
};

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
bool ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(
    const ConvolutionContext& params) const
{
// HIP backend required for sending ptr (buffer + offset)
// ROCBLAS for GEMM step

#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)
    static const int wino_data_tile   = std::max(WinoDataH, WinoDataW);
    static const int wino_filter_tile = std::max(WinoFilterH, WinoFilterW);

    if(wino_data_tile == 3 && wino_filter_tile == 2)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2{}) ||
           params.kernel_stride_h == 1)
            return false;
    if(wino_data_tile == 3 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3{}) ||
           params.kernel_stride_h == 1)
            return false;

    const std::string name = params.GetStream().GetDeviceName();
#if WORKAROUND_SWDEV_234193
    if(params.IsFp16() && (StartsWith(name, "gfx908") || StartsWith(name, "gfx906")))
    {
        if(wino_data_tile == 3 && wino_filter_tile == 4)
            if(!miopen::IsEnabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4{}))
                return false;
        if(wino_data_tile == 3 && wino_filter_tile == 5)
            if(!miopen::IsEnabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5{}))
                return false;
        if(wino_data_tile == 3 && wino_filter_tile == 6)
            if(!miopen::IsEnabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6{}))
                return false;
    }
    else
#endif
    {
        if(wino_data_tile == 3 && wino_filter_tile == 4)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4{}))
                return false;
        if(wino_data_tile == 3 && wino_filter_tile == 5)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5{}))
                return false;
        if(wino_data_tile == 3 && wino_filter_tile == 6)
            if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6{}))
                return false;
    }

    if(wino_data_tile == 7 && wino_filter_tile == 2)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2{}))
            return false;
    if(wino_data_tile == 7 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3{}))
            return false;
    if(wino_data_tile == 5 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3{}))
            return false;
    if(wino_data_tile == 5 && wino_filter_tile == 4)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4{}))
            return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV2orV3())
        return false;
    if(!params.Is2d())
        return false;
    if(!params.direction.IsBackwardWrW())
        return false;
    if(!(params.IsFp32() || params.IsFp16() || params.IsBfp16()))
        return false;

    if(!(InTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(params) &&
         OutTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(params) &&
         FilterTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(params)))
        return false;

    if(!(StartsWith(name, "gfx8") || StartsWith(name, "gfx9")))
        return false;

    {
        std::size_t limit = miopen::Value(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX{});
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
            const auto required = GetWorkspaceSize(params);
            MIOPEN_LOG_I2("Workspace required: " << required << ", limit: " << limit);
            if(required > limit)
                return false;
        }
    }

    // int offset for Workspace buffers.
    if((InTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetBufferSize(params) /
            GetTypeSize(params.in_data_type) +
        OutTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetBufferSize(params) /
            GetTypeSize(params.in_data_type)) >= (1LL << 31))
    {
        return false;
    }

    assert(params.weights_layout.length() == 0); // _weights_layout is not supported yet

    // clang-format off
    {
        const long input_line_size = 4 * params.in_width;
        const long input_feature_map_size = input_line_size * params.in_height;
        const long input_stack_size = input_feature_map_size * params.n_inputs;
        if (! (input_stack_size < (1U << 24)))
            return false;
    }
    bool ok = (
           (params.kernel_size_w == WinoDataW && params.kernel_size_h == WinoDataH)
        && (params.kernel_stride_w == 1
            ||
            (params.kernel_stride_w == 2 && params.kernel_size_h == 3 && params.kernel_size_w == 3)
            )
        && params.kernel_stride_h == params.kernel_stride_w
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.batch_sz < std::pow(2, 24)
        && params.n_inputs < std::pow(2, 24)
        && params.n_outputs < std::pow(2, 24)
        && params.in_height < std::pow(2, 24)
        && params.in_width < std::pow(2, 24)
        && params.bias == 0
        && params.in_layout == "NCHW"
        && params.group_counts == 1);
    // clang-format on
    return ok;
#else
    (void)params;
    return false;
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
size_t
ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetWorkspaceSize(
    const ConvolutionContext& params) const
{
    return InTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetBufferSize(params) +
           OutTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetBufferSize(params) +
           FilterTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetBufferSize(params);
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvSolution
ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolution(
    const ConvolutionContext& params) const
{
    ConvSolution result;
    result.workspce_sz = GetWorkspaceSize(params);

    result.construction_params.push_back(
        InTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetKernel(params));
    result.construction_params.push_back(
        FilterTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetKernel(params));
    result.construction_params.push_back(
        OutTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetKernel(params));

    return result;
}
template struct ConvWinograd3x3MultipassWrW<3, 2>;
template struct ConvWinograd3x3MultipassWrW<3, 3>;
template struct ConvWinograd3x3MultipassWrW<3, 4>;
template struct ConvWinograd3x3MultipassWrW<3, 5>;
template struct ConvWinograd3x3MultipassWrW<3, 6>;
template struct ConvWinograd3x3MultipassWrW<7, 2>;
template struct ConvWinograd3x3MultipassWrW<7, 3>;
template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 2>;
template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 3>;
template struct ConvWinograd3x3MultipassWrW<7, 2, 1, 1>;
template struct ConvWinograd3x3MultipassWrW<7, 3, 1, 1>;
template struct ConvWinograd3x3MultipassWrW<5, 3>;
template struct ConvWinograd3x3MultipassWrW<5, 4>;

} // namespace solver
} // namespace miopen
