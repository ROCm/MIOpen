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

#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/kernel_build_params.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3)

#define WINODATA 2
#define WINOFILTER 3

static inline int Ceil(const int v, const int m)
{
    assert(m > 0 && v >= 0);
    return (v + m - 1) / m;
}

static inline size_t quantize_up(size_t val, size_t factor) { return Ceil(val, factor) * factor; }

static inline int GetBestNGroupParam_Wrapper(const int R,
                                             const int S,
                                             const int R_stride,
                                             const int S_stride,
                                             const int C,
                                             const int K,
                                             const int OH,
                                             const int OW,
                                             const int pad_H,
                                             const int pad_W,
                                             const int N,
                                             const int idilation_w,
                                             const int idilation_h,
                                             const int n_groups,
                                             const int G)
{
    int o_tile     = WINODATA;
    int f_tile     = WINOFILTER;
    int r_factor   = f_tile * 2;
    int s_factor   = r_factor;
    int c_factor   = 2;
    int k_factor   = 32;
    int nwh_factor = 32;
    int w_factor   = o_tile * idilation_w * S_stride;
    int h_factor   = o_tile * idilation_h * R_stride;

    if(S_stride == 1 && idilation_w == 1 && S <= f_tile)
        s_factor = f_tile;
    if((R_stride == 1 && idilation_h == 1) || (R % (f_tile * 2)) == 1)
        r_factor = f_tile;
    if(S_stride == 2 || R_stride == 2 || idilation_w == 2 || idilation_h == 2)
        c_factor = 1;

    size_t g_s = quantize_up(S, s_factor);
    size_t g_r = quantize_up(R, r_factor);
    size_t g_c = quantize_up(C, c_factor);
    size_t g_k = quantize_up(K, k_factor);
    size_t g_w = OW;
    size_t g_h = OH;

    if((pad_W % 2 == 0) && (idilation_w > 1 || S_stride > 1))
        g_w += 1;
    if((pad_H % 2 == 1) && (idilation_h > 1 || R_stride > 1))
        g_h += 1;

    g_w            = quantize_up(g_w, w_factor);
    g_h            = quantize_up(g_h, h_factor);
    size_t g_n_w_h = quantize_up(g_w * g_h * N, nwh_factor * w_factor * h_factor);

    int best_n_groups_cnt = 1;
    double min_param      = 0;
    for(auto i = 1; i < n_groups; ++i)
    {
        size_t g_n_w_h_k =
            quantize_up(g_n_w_h * g_k, nwh_factor * w_factor * h_factor * k_factor * i);
        size_t granulated_mac_count = g_n_w_h_k * g_c * g_s * g_r;
        size_t n_groups_per_cu      = Ceil(i * G, n_groups);
        double perf_metric = static_cast<double>(n_groups_per_cu * granulated_mac_count) / i;
        if(i == 1)
            min_param = perf_metric;
        if(min_param > perf_metric * 1.08)
        {
            best_n_groups_cnt = i;
            min_param         = perf_metric;
        }
    }
    return best_n_groups_cnt;
}

int GetBestNGroupParam_euristicInit(const miopen::ConvolutionContext& params)
{
    const auto n_inputs_per_group  = params.n_inputs / params.group_counts,
               n_outputs_per_group = params.n_outputs / params.group_counts;

    if(params.direction.IsBackwardWrW())
    {
        return GetBestNGroupParam_Wrapper(params.in_height,
                                          params.in_width,
                                          params.kernel_dilation_h,
                                          params.kernel_dilation_w,
                                          params.batch_sz,    // N
                                          n_inputs_per_group, // K
                                          params.kernel_size_h,
                                          params.kernel_size_w,
                                          params.pad_w,
                                          params.pad_h,
                                          n_outputs_per_group, // C
                                          params.kernel_stride_h,
                                          params.kernel_stride_w,
                                          params.GetStream().GetMaxComputeUnits(),
                                          params.group_counts);
    }
    else
    {
        return GetBestNGroupParam_Wrapper(params.kernel_size_h, // RxS
                                          params.kernel_size_w,
                                          params.kernel_stride_h,
                                          params.kernel_stride_w,
                                          n_inputs_per_group,  // C
                                          n_outputs_per_group, // K
                                          params.out_height,   // OHxOW
                                          params.out_width,
                                          params.pad_w,
                                          params.pad_h,
                                          params.batch_sz, // N
                                          params.kernel_dilation_h,
                                          params.kernel_dilation_w,
                                          params.GetStream().GetMaxComputeUnits(),
                                          params.group_counts);
    }
}

/// \todo Consider re-using code from RxS.
static inline bool IsShaderContraintsMet(const int R,
                                         const int S,
                                         const int,
                                         const int,
                                         const int C,
                                         const int K,
                                         const int H,
                                         const int W,
                                         const int OH,
                                         const int OW,
                                         const int N,
                                         const miopen::ConvolutionContext& params)
{
    // Padding for bwd data shall not be negative.
    /// \todo Either remove WrW related code or re-use function from RxS
    if(params.direction.IsBackwardData() || params.direction.IsBackwardWrW())
    {
        if(!(0 <= params.GetBackwardPadW() && params.GetBackwardPadW() < std::pow(2, 16)))
            return false;
        if(!(0 <= params.GetBackwardPadH() && params.GetBackwardPadH() < std::pow(2, 16)))
            return false;
    }
    const auto grid_workgroup_count_x = params.GetStream().GetMaxComputeUnits();
    assert(params.weights_layout.length() == 0);
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

namespace miopen {
namespace solver {

bool ConvBinWinogradRxSf2x3::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.Is2d())
        return false;
    if(!params.IsFp32())
        return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;

    const auto name = params.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx9")))
        return false;

    // clang-format off
    if (! ( (params.kernel_stride_w == 1 || params.kernel_stride_w == 2)
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0
        && params.in_layout == "NCHW"))
        return false;
    // clang-format on

    const auto n_inputs_per_group  = params.n_inputs / params.group_counts,
               n_outputs_per_group = params.n_outputs / params.group_counts;

    if(params.direction.IsBackwardWrW())
    {
        if(params.kernel_stride_w == 2)
            return false;
        return IsShaderContraintsMet(params.in_height,
                                     params.in_width,
                                     params.kernel_dilation_h,
                                     params.kernel_dilation_w,
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
                                     params.kernel_stride_h,
                                     params.kernel_stride_w,
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

ConvSolution ConvBinWinogradRxSf2x3::GetSolution(const ConvolutionContext& params) const
{
    auto group_counts = params.group_counts;
    ConvSolution result;
    const auto n_groups = (group_counts == 1) ? params.GetStream().GetMaxComputeUnits()
                                              : GetBestNGroupParam_euristicInit(params);
    KernelInfo kernel;

    kernel.g_wk.push_back(512 * n_groups * params.group_counts);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernel.comp_options = options.GenerateFor(kbp::GcnAsm{});

    if(params.kernel_stride_w == 1)
    {
        kernel.kernel_name = "miopenSp3AsmConv_group_20_5_23_M_stride1";
        kernel.kernel_file = "Conv_Winograd_v20_5_23_M_stride1.s";
    }
    else if(params.kernel_stride_w == 2 && !params.direction.IsBackwardData())
    {
        kernel.kernel_name = "miopenSp3AsmConv_group_20_5_23_M_stride2";
        kernel.kernel_file = "Conv_Winograd_v20_5_23_M_stride2.s";
    }
    else // if(params.kernel_dilation_h == 2)
    {
        kernel.kernel_name = "miopenSp3AsmConv_group_20_5_23_M_dilation2";
        kernel.kernel_file = "Conv_Winograd_v20_5_23_M_dilation2.s";
    }

    result.construction_params.push_back(kernel);
    return result;
}

} // namespace solver
} // namespace miopen
