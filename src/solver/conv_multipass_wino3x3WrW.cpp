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
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>

#include "miopen/solver.hpp"

namespace miopen {
namespace solver {

// Introduces a number of shader-specific aliases (names) in the current scope at zero cost.
// These names represent shader parameters, e.g. shader C is batch_size etc and useful for
// programming.
#define DEFINE_SHADER_ALIASES(params)           \
    const auto& C     = (params).batch_sz;      \
    const auto& N     = (params).n_outputs;     \
    const auto& K     = (params).n_inputs;      \
    const auto& out_H = (params).kernel_size_h; \
    const auto& out_W = (params).kernel_size_w; \
    const auto& R     = (params).in_height;     \
    const auto& S     = (params).in_width;      \
    const auto& H     = (params).out_height;    \
    const auto& W     = (params).out_width;

const int xform_f_size  = 4;
const int xform_o_size  = 3;
const int xformy_d_size = (xform_f_size + xform_o_size - 1);

struct InTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)
        WinogradBufferInfo<3, 4> wino_info(N,
                                           K,
                                           C,
                                           out_H,
                                           out_W,
                                           R,
                                           S,
                                           GetSwappedNCLayout(MemLayout_t::HWCN),
                                           1,
                                           GetTypeSize(params.in_data_type),
                                           ConvWinoBuffType::Input);

        const size_t u16limit       = 1 << 16;
        const size_t tiles_per_wave = wave_size / xformy_d_size;
        // clang-format off
        const size_t chw_step       = tiles_per_wave 
            * params.GetStream().GetMaxComputeUnits() 
            * ConvWinograd3x3MultipassWrW::GetGroupCountMult();
        
        return params.IsFp32()
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
        const size_t g_wk_0 = params.GetStream().GetMaxComputeUnits() *
                              ConvWinograd3x3MultipassWrW::GetGroupCountMult() * l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};

        return KernelInfo{
            "-Wa,-defsym,ROCM_METADATA_VERSION=4",
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW::GetSolverFileNames()[0],
            ConvWinograd3x3MultipassWrW::GetSolverKernelNames()[0],
        };
    }
};

struct FilterTransform
{
    static bool IsApplicable(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)
        WinogradBufferInfo<3, 4> wino_info(N,
                                           K,
                                           C,
                                           out_H,
                                           out_W,
                                           R,
                                           S,
                                           GetSwappedNCLayout(MemLayout_t::HWCN),
                                           1,
                                           GetTypeSize(params.in_data_type),
                                           ConvWinoBuffType::Input);

        const size_t u16limit       = 1 << 16;
        const size_t tiles_per_wave = wave_size / xformy_d_size;
        // clang-format off
        const size_t chw_step       = tiles_per_wave 
            * params.GetStream().GetMaxComputeUnits() 
            * ConvWinograd3x3MultipassWrW::GetGroupCountMult();

        return params.IsFp32()
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
        const size_t g_wk_0 = params.GetStream().GetMaxComputeUnits() *
                              ConvWinograd3x3MultipassWrW::GetGroupCountMult() * l_wk[0];

        const std::vector<size_t> g_wk{g_wk_0, 1, 1};

        return KernelInfo{
            "-Wa,-defsym,ROCM_METADATA_VERSION=4",
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW::GetSolverFileNames()[1],
            ConvWinograd3x3MultipassWrW::GetSolverKernelNames()[1],
        };
    }
};

struct OutTransform
{
    static bool IsApplicable(const ConvolutionContext& params) { return params.IsFp32(); }
    static KernelInfo GetKernel(const ConvolutionContext& params)
    {
        DEFINE_SHADER_ALIASES(params)

        WinogradBufferInfo<3, 4> wino_weight(N,
                                             K,
                                             C,
                                             out_H,
                                             out_W,
                                             R,
                                             S,
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

        return KernelInfo{
            "-Wa,-defsym,ROCM_METADATA_VERSION=4",
            l_wk,
            g_wk,
            ConvWinograd3x3MultipassWrW::GetSolverFileNames()[2],
            ConvWinograd3x3MultipassWrW::GetSolverKernelNames()[2],

        };
    }
};

bool ConvWinograd3x3MultipassWrW::IsApplicable(const ConvolutionContext& params) const
{
// HIP backend required for sending ptr (buffer + offset)
// ROCBLAS for GEMM step
#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)

    if(!params.use_asm_kernels)
        return false;
    if(params.rmv != rocm_meta_version::AMDHSA_1_0)
        return false;

    if(!(params.IsFp32()))
        return false;

    if(!(InTransform::IsApplicable(params) && OutTransform::IsApplicable(params) &&
         FilterTransform::IsApplicable(params)))
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos && name.find("gfx9") == std::string::npos)
    {
        return false;
    }

    // int offset for Workspace buffers.
    if(!(GetWorkspaceSize(params) < (1LL << 31)))
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
        && params.kernel_stride_w == 1
        && params.kernel_stride_h == 1
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
size_t ConvWinograd3x3MultipassWrW::GetWorkspaceSize(const ConvolutionContext& params) const
{
    DEFINE_SHADER_ALIASES(params)

    WinogradBufferInfo<3, 4> InTransform_info(N,
                                              K,
                                              C,
                                              out_H,
                                              out_W,
                                              R,
                                              S,
                                              GetSwappedNCLayout(MemLayout_t::HWNC),
                                              1,
                                              GetTypeSize(params.in_data_type),
                                              ConvWinoBuffType::Input);

    WinogradBufferInfo<3, 4> FilterTransform_info(N,
                                                  K,
                                                  C,
                                                  out_H,
                                                  out_W,
                                                  R,
                                                  S,
                                                  MemLayout_t::HWNC,
                                                  1,
                                                  GetTypeSize(params.in_data_type),
                                                  ConvWinoBuffType::Weight);

    WinogradBufferInfo<3, 4> OutTransform_info(N,
                                               K,
                                               C,
                                               out_H,
                                               out_W,
                                               R,
                                               S,
                                               GetSwappedNCLayout(MemLayout_t::HWNC),
                                               1,
                                               GetTypeSize(params.in_data_type),
                                               ConvWinoBuffType::Output);

    (void)H;
    (void)W;
    return InTransform_info.buff_info.total_byte_size +
           FilterTransform_info.buff_info.total_byte_size +
           OutTransform_info.buff_info.total_byte_size;
}

ConvSolution ConvWinograd3x3MultipassWrW::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    result.workspce_sz = GetWorkspaceSize(params);

    result.construction_params.push_back(InTransform::GetKernel(params));
    result.construction_params.push_back(FilterTransform::GetKernel(params));
    result.construction_params.push_back(OutTransform::GetKernel(params));

    return result;
}

} // namespace solver
} // namespace miopen
