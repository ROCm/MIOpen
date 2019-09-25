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
#endif

namespace miopen {
namespace solver {
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX)

// Introduces a number of shader-specific aliases (names) in the current scope at zero cost.
// These names represent shader parameters, e.g. shader C is batch_size etc and useful for
// programming.
#define DEFINE_SHADER_ALIASES(params)              \
    const auto& C      = (params).batch_sz;        \
    const auto& N      = (params).n_outputs;       \
    const auto& K      = (params).n_inputs;        \
    const auto& out_H  = (params).kernel_size_h;   \
    const auto& out_W  = (params).kernel_size_w;   \
    const auto& R      = (params).in_height;       \
    const auto& S      = (params).in_width;        \
    const auto& H      = (params).out_height;      \
    const auto& W      = (params).out_width;       \
    const auto& fdil_H = (params).kernel_stride_h; \
    const auto& fdil_W = (params).kernel_stride_w;

#define xform_f_size WinoFilterW
#define xform_o_size WinoDataW
#define xformy_d_size (xform_f_size + xform_o_size - 1)

template <int WinoDataW, int WinoFilterW>
struct InTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)
        const WinogradBufferInfo<WinoDataW, WinoFilterW> wino_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            fdil_H,
            fdil_W,
            GetSwappedNCLayout(MemLayout_t::HWCN),
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Input);

        const size_t u16limit       = 1 << 16;
        const size_t tiles_per_wave = wave_size / xformy_d_size;
        // clang-format off
        const size_t chw_step       = tiles_per_wave
            * params.GetStream().GetMaxComputeUnits()
            * ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetGroupCountMult();
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
                && params.pad_h <= 1
                && params.pad_w <= 1;
        // clang-format on
    }
    static KernelInfo GetKernel(const ConvolutionContext& params)
    {
        const std::vector<size_t> l_wk{wave_size, 1, 1};
        const size_t g_wk_0 =
            params.GetStream().GetMaxComputeUnits() *
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetGroupCountMult() * l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};
        std::ostringstream options;
        GenerateClangDefsym(options, "acc_type", 1);
        GenerateClangDefsym(options, "buf_type", (params.IsFp32() ? 1 : (params.IsFp16() ? 2 : 3)));
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", 4);
        GenerateClangDefsym(options, "xformx_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformy_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformx_f_size", WinoFilterW);
        GenerateClangDefsym(options, "xformy_f_size", WinoFilterW);
        GenerateClangDefsym(options, "fdilation_w", params.kernel_stride_w);
        GenerateClangDefsym(options, "fdilation_h", params.kernel_stride_h);

        GenerateClangDefsym(options, "MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16);

        return KernelInfo{
            options.str(),
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetSolverFileNames(0),
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetSolverKernelNames(0),
        };
    }
    static size_t GetBufferSize(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)
        const WinogradBufferInfo<WinoDataW, WinoFilterW> InTransform_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            fdil_H,
            fdil_W,
            MemLayout_t::HWNC,
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Input);
        (void)H;
        (void)W;
        return InTransform_info.buff_info.total_byte_size;
    }
};

template <int WinoDataW, int WinoFilterW>
struct FilterTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)
        const WinogradBufferInfo<WinoDataW, WinoFilterW> wino_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            fdil_H,
            fdil_W,
            GetSwappedNCLayout(MemLayout_t::HWCN),
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Input);

        const size_t u16limit       = 1 << 16;
        const size_t tiles_per_wave = wave_size / xformy_d_size;
        // clang-format off
        const size_t chw_step       = tiles_per_wave
            * params.GetStream().GetMaxComputeUnits()
            * ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetGroupCountMult();
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
                && params.pad_h <= 1
                && params.pad_w <= 1;
        // clang-format on
    }
    static KernelInfo GetKernel(const ConvolutionContext& params)
    {
        const std::vector<size_t> l_wk{wave_size, 1, 1};
        const size_t g_wk_0 =
            params.GetStream().GetMaxComputeUnits() *
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetGroupCountMult() * l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};

        std::ostringstream options;
        GenerateClangDefsym(options, "acc_type", 1);
        GenerateClangDefsym(options, "buf_type", (params.IsFp32() ? 1 : (params.IsFp16() ? 2 : 3)));
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", 4);
        GenerateClangDefsym(options, "MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16);
        GenerateClangDefsym(options, "xformx_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformy_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformx_f_size", WinoFilterW);
        GenerateClangDefsym(options, "xformy_f_size", WinoFilterW);
        GenerateClangDefsym(options, "fdilation_w", params.kernel_stride_w);
        GenerateClangDefsym(options, "fdilation_h", params.kernel_stride_h);
        return KernelInfo{
            options.str(),
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetSolverFileNames(1),
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetSolverKernelNames(1),
        };
    }
    static size_t GetBufferSize(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)
        const WinogradBufferInfo<WinoDataW, WinoFilterW> FilterTransform_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            fdil_H,
            fdil_W,
            MemLayout_t::HWNC,
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Weight);
        (void)H;
        (void)W;
        return FilterTransform_info.buff_info.total_byte_size;
    }
};

template <int WinoDataW, int WinoFilterW>
struct OutTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        return (params.IsFp32() || params.IsFp16() || params.IsBfp16()) && params.Is2d();
    }
    static KernelInfo GetKernel(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        const WinogradBufferInfo<WinoDataW, WinoFilterW> wino_weight(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            fdil_H,
            fdil_W,
            GetSwappedNCLayout(MemLayout_t::HWCN),
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Weight);
        auto dwords_per_ld = 1;
        const std::vector<size_t> l_wk{wave_size, 1, 1};
        auto ceil_val       = dwords_per_ld * l_wk[0];
        const size_t g_wk_0 = ((N * K + ceil_val - 1) / ceil_val) * l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};

        (void)H;
        (void)W;

        std::ostringstream options;
        GenerateClangDefsym(options, "acc_type", 1);
        GenerateClangDefsym(options, "buf_type", (params.IsFp32() ? 1 : (params.IsFp16() ? 2 : 3)));
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", 4);
        GenerateClangDefsym(options, "MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16);
        GenerateClangDefsym(options, "xformx_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformy_o_size", WinoDataW);
        GenerateClangDefsym(options, "xformx_f_size", WinoFilterW);
        GenerateClangDefsym(options, "xformy_f_size", WinoFilterW);
        GenerateClangDefsym(options, "fdilation_w", params.kernel_stride_w);
        GenerateClangDefsym(options, "fdilation_h", params.kernel_stride_h);

        return KernelInfo{
            options.str(),
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetSolverFileNames(2),
            ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetSolverKernelNames(2),
        };
    }
    static size_t GetBufferSize(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        const WinogradBufferInfo<WinoDataW, WinoFilterW> OutTransform_info(
            N,
            K,
            C,
            out_H,
            out_W,
            R,
            S,
            fdil_H,
            fdil_W,
            GetSwappedNCLayout(MemLayout_t::HWNC),
            1,
            GetTypeSize(params.in_data_type),
            ConvWinoBuffType::Output);
        (void)H;
        (void)W;
        return OutTransform_info.buff_info.total_byte_size;
    }
};

template <int WinoDataW, int WinoFilterW>
bool ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::IsApplicable(
    const ConvolutionContext& params) const
{
// HIP backend required for sending ptr (buffer + offset)
// ROCBLAS for GEMM step

#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)

    if(WinoDataW == 3 && WinoFilterW == 2)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2{}) ||
           params.kernel_stride_h == 1)
            return false;
    if(WinoDataW == 3 && WinoFilterW == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3{}) ||
           params.kernel_stride_h == 1)
            return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4{}) && WinoDataW == 3 &&
       WinoFilterW == 4)
        return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5{}) && WinoDataW == 3 &&
       WinoFilterW == 5)
        return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6{}) && WinoDataW == 3 &&
       WinoFilterW == 6)
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(params.rmv != rocm_meta_version::AMDHSA_1_0)
        return false;
    if(!params.Is2d())
        return false;
    if(!(params.IsFp32() || params.IsFp16() || params.IsBfp16()))
        return false;

    if(!(InTransform<WinoDataW, WinoFilterW>::IsApplicable(params) &&
         OutTransform<WinoDataW, WinoFilterW>::IsApplicable(params) &&
         FilterTransform<WinoDataW, WinoFilterW>::IsApplicable(params)))
        return false;

    const std::string name = params.GetStream().GetDeviceName();
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
    if((InTransform<WinoDataW, WinoFilterW>::GetBufferSize(params) /
            GetTypeSize(params.in_data_type) +
        OutTransform<WinoDataW, WinoFilterW>::GetBufferSize(params) /
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
           params.kernel_size_w == 3
        && params.kernel_size_h == 3
        && (params.kernel_stride_w == 1 || params.kernel_stride_w == 2)
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
        && params.direction.IsBackwardWrW()
        && params.group_counts == 1);
    // clang-format on
    return ok;
#else
    (void)params;
    return false;
#endif
}

template <int WinoDataW, int WinoFilterW>
size_t ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetWorkspaceSize(
    const ConvolutionContext& params) const
{
    return InTransform<WinoDataW, WinoFilterW>::GetBufferSize(params) +
           OutTransform<WinoDataW, WinoFilterW>::GetBufferSize(params) +
           FilterTransform<WinoDataW, WinoFilterW>::GetBufferSize(params);
}

template <int WinoDataW, int WinoFilterW>
ConvSolution ConvWinograd3x3MultipassWrW<WinoDataW, WinoFilterW>::GetSolution(
    const ConvolutionContext& params) const
{
    ConvSolution result;
    result.workspce_sz = GetWorkspaceSize(params);

    result.construction_params.push_back(InTransform<WinoDataW, WinoFilterW>::GetKernel(params));
    result.construction_params.push_back(
        FilterTransform<WinoDataW, WinoFilterW>::GetKernel(params));
    result.construction_params.push_back(OutTransform<WinoDataW, WinoFilterW>::GetKernel(params));

    return result;
}
template struct ConvWinograd3x3MultipassWrW<3, 2>;
template struct ConvWinograd3x3MultipassWrW<3, 3>;
template struct ConvWinograd3x3MultipassWrW<3, 4>;
template struct ConvWinograd3x3MultipassWrW<3, 5>;
template struct ConvWinograd3x3MultipassWrW<3, 6>;

} // namespace solver
} // namespace miopen
