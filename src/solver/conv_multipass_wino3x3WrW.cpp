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

#define CONV_MULTIPASS_WINO3X3WRW_CPP

#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/solver.hpp>
#if(MIOPEN_BACKEND_HIP && (MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE))
#define WORKAROUND_SWDEV_203031 1 // See also issues #2075, #2067
#define WORKAROUND_SWDEV_234193 1
#define WORKAROUND_ISSUE_1146 1 // check asm solver applicability for gfx90a
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
                && wino_info.buff_info.size.c < (1<<30)
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
                && wino_info.buff_info.size.c < (1<<30)
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

#if(MIOPEN_BACKEND_HIP && (MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE))
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
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    if(!(InTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(params) &&
         OutTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(params) &&
         FilterTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(params)))
        return false;

    if(!(StartsWith(name, "gfx8") || StartsWith(name, "gfx9")))
        return false;
#if WORKAROUND_ISSUE_1146
    if(name == "gfx90a")
        return false;
#endif

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
    if(!params.IsLayoutDefault())
    {
        return false;
    }

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

    result.invoker_factory = PrepareInvokerFactory(params, result.workspce_sz);

    return result;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
InvokerFactory
ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::PrepareInvokerFactory(
    const ConvolutionContext& params, std::size_t ws_sz) const
{
#if(MIOPEN_BACKEND_HIP && (MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE))
    int flags         = 0;
    int reserved      = 0;
    int* reserved_ptr = nullptr;
    int unused        = 0;
    int N, C, H, W, K, n_groups, out_H, out_W, R, S;

    GetCompiledInParameters(
        params, &C, &K, &R, &S, &N, &n_groups, &H, &W, &out_H, &out_W, &unused, &unused);
    // clang-format off
    BuffInfo
        in_buff_info(
            GetSwappedNCLayout(GetMemLayout_t(params.in_layout)),
            N, C, H, W,
            GetTypeSize(params.in_data_type)),
        out_buff_info(
            GetSwappedNCLayout(GetMemLayout_t(params.out_layout)),
            N, K, out_H, out_W,
            GetTypeSize(params.out_data_type)),
        weights_buff_info(
            // weights_layout unsupported ... GetSwappedNCLayout(GetMemLayout_t(params.weights_layout))
            GetSwappedNCLayout(MemLayout_t::NCHW),
            K, C, R, S,
            GetTypeSize(params.weights_data_type));

    int wino_xform_h = GetSolverWinoXformHWSize(params,0),
        wino_xform_w = GetSolverWinoXformHWSize(params,1);
    WinogradBufferInfo <WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
        // cppcheck-suppress unreadVariable
        wino_in(N,K,C,out_H,out_W,R,S,
            MemLayout_t::HWNC,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Input,
            wino_xform_h,
            wino_xform_w),
        // cppcheck-suppress unreadVariable
        wino_out(N,K,C,out_H,out_W,R,S,
            MemLayout_t::HWNC,
            GetTypeSize(params.out_data_type),
            ConvWinoBuffType::Output,
            wino_xform_h,
            wino_xform_w),
        // cppcheck-suppress unreadVariable
        wino_wei(N,K,C,out_H,out_W,R,S,
            MemLayout_t::HWNC,
            GetTypeSize(params.weights_data_type),
            ConvWinoBuffType::Weight,
            wino_xform_h,
            wino_xform_w);
    // clang-format on

    const size_t wino_in_offset = 0, wino_out_offset = wino_in.buff_info.total_byte_size,
                 wino_wei_offset = wino_out_offset + wino_out.buff_info.total_byte_size;

    const auto in_data_type = params.in_data_type;
    const auto pad_H        = params.pad_h;
    const auto pad_W        = params.pad_w;

    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            decltype(auto) invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();
            const auto& tensors          = invoke_params.tensors;
            float total_time             = 0;

            if(invoke_params.workSpaceSize < ws_sz)
                MIOPEN_THROW("Not enough workspace for ConvWinograd3x3MultipassWrW");

            for(const auto& kernel : kernels)
            {
                decltype(auto) cur_kernel = handle.Run(kernel);
                const BuffInfo* d_buf     = nullptr;
                const BuffInfo* o_buf     = nullptr;
                Data_t buff_out_adr       = nullptr;
                auto f_buf                = &weights_buff_info;
                auto const_buff_in_adr    = tensors.x;
                auto buff_in_adr          = invoke_params.workSpace;
                bool const_input          = false;
                float cur_time            = 0;
                int flat_GroupCountMult   = 1;

                size_t buff_in_addr_offset = 0, buff_out_addr_offset = 0;

                if(cur_kernel.GetName() == GetSolverKernelNames(0)) // Input Transform
                {
                    d_buf               = &in_buff_info;
                    o_buf               = &(wino_in.buff_info);
                    const_buff_in_adr   = tensors.x;
                    buff_out_adr        = invoke_params.workSpace;
                    buff_in_addr_offset = wino_in_offset;
                    const_input         = true;
                    flat_GroupCountMult = GetGroupCountMult();
                }
                else if(cur_kernel.GetName() == GetSolverKernelNames(1)) // Filter Transform
                {
                    d_buf                = &weights_buff_info;
                    o_buf                = &(wino_wei.buff_info);
                    const_buff_in_adr    = tensors.dy;
                    buff_out_adr         = invoke_params.workSpace;
                    buff_out_addr_offset = wino_wei_offset;
                    const_input          = true;
                    flat_GroupCountMult  = GetGroupCountMult();
                }
                else // Output and GEMM
                {
                    int m = N, n = K, k = wino_in.buff_info.size.c;
                    int lda = k, ldb = k, ldc = n;
                    int batch_count       = wino_xform_h * wino_xform_w;
                    long long int strideA = m * k * 1LL, strideB = k * n * 1LL,
                                  strideC = m * n * 1LL;
                    float alpha = 1., beta = 0.0;
                    // clang-format off
                    GemmDescriptor wino_gemm_desc{false,false,true,m,n,k,
                        lda,ldb,ldc,batch_count,strideA,strideB,
                                        strideC,alpha,beta,in_data_type};

                    CallGemmStridedBatched(handle,
                                        wino_gemm_desc,
                                        invoke_params.workSpace,
                                        static_cast<int>(wino_in_offset / GetTypeSize(in_data_type)),
                                        invoke_params.workSpace,
                                        static_cast<int>(wino_wei_offset / GetTypeSize(in_data_type)),
                                        invoke_params.workSpace,
                                        static_cast<int>(wino_out_offset / GetTypeSize(in_data_type)),
                                        nullptr,
                                GemmBackend_t::miopentensile);
                    // clang-format on

                    if(handle.IsProfilingEnabled())
                    {
                        cur_time = handle.GetKernelTime();
                        total_time += cur_time;
                        MIOPEN_LOG_I2("WRW_WINO_GEMM: " << cur_time);
                    }

                    d_buf               = &(wino_out.buff_info);
                    o_buf               = &(out_buff_info);
                    buff_in_adr         = invoke_params.workSpace;
                    buff_in_addr_offset = wino_out_offset;
                    buff_out_adr        = tensors.dw;
                }

                const auto input_ptr = static_cast<const void*>(
                    static_cast<const char*>(const_input ? const_buff_in_adr : buff_in_adr) +
                    buff_in_addr_offset);
                const auto output_ptr =
                    static_cast<void*>(static_cast<char*>(buff_out_adr) + buff_out_addr_offset);

                cur_kernel(N,
                           C,
                           H,
                           W,
                           K,
                           n_groups * flat_GroupCountMult,
                           flags,
                           reserved,
                           input_ptr,
                           reserved_ptr,
                           output_ptr,
                           reserved_ptr,
                           R,
                           S,
                           pad_H,
                           pad_W,
                           out_H,
                           out_W,
                           reserved_ptr,
                           reserved,
                           d_buf->byte_stride.nk,
                           d_buf->byte_stride.c,
                           d_buf->byte_stride.h,
                           d_buf->byte_stride.w,
                           f_buf->byte_stride.nk,
                           f_buf->byte_stride.c,
                           f_buf->byte_stride.h,
                           f_buf->byte_stride.w,
                           o_buf->byte_stride.nk,
                           o_buf->byte_stride.c,
                           o_buf->byte_stride.h,
                           o_buf->byte_stride.w);

                if(handle.IsProfilingEnabled())
                {
                    cur_time = handle.GetKernelTime();
                    total_time += cur_time;
                    MIOPEN_LOG_I2(cur_kernel.GetName() << ": " << cur_time);
                }
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(total_time);
            }
        };
    };
#else
    (void)params;
    (void)ws_sz;
    MIOPEN_THROW(miopenStatusBadParm, "MixedWrW3x3Winograd Unsupported ");
#endif
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
