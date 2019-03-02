/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include "miopen/solver.hpp"
#include "miopen/env.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW)
/// \todo Detect at runtime and remove this var:
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_SRAM_EDC_DISABLED)

/// \return v rounded up (towards +inf) to the nearest multiple of m.
/// Defined for positive values only.
static inline int Ceiling(const int v, const int m)
{
    assert(m > 0 && v >= 0);
    if(v % m != 0)
    {
        return (v / m + 1) * m;
    }
    return v;
}

/// \return Value equivalent to ceil(x/y).
/// Defined for positive values only.
static inline int CeilDiv(const int x, const int y)
{
    assert(y > 0);
    return Ceiling(x, y) / y;
}

/// \return Value equivalent to floor(x/y).
/// Defined for positive values only.
static inline int FloorDiv(const int x, const int y)
{
    assert(x >= 0 && y > 0);
    return x / y;
}

static inline bool IsShaderContraintsMet(const int R,
                                         const int S,
                                         const int R_stride,
                                         const int S_stride,
                                         const int C,
                                         const int K,
                                         const int H,
                                         const int W,
                                         const int OH,
                                         const int OW,
                                         const int N,
                                         const miopen::ConvolutionContext& params,
                                         const bool fp16,
                                         const unsigned filter_tile_size)
{
    static const unsigned TILE    = filter_tile_size;
    static const unsigned TILE_X2 = TILE * 2;
    // Calculate padded filter size first.
    // If stride = 1: if S <= 3 it is padded to 3,
    // otherwise S is padded to smallest 6*n for some integer n
    // If stride = 2: S is always padded to smallest 6*n for some integer n
    int padded_S = 0;
    if(S_stride == 1)
    {
        if(S <= TILE)
        {
            padded_S = TILE;
        }
        else
        {
            padded_S = Ceiling(S, TILE_X2);
        }
    }
    else
    {
        padded_S = Ceiling(S, TILE_X2);
    }
    // If stride = 1: R is always padded to smallest 3*m for some integer m
    // If stride = 2: if R % 6 ==1 then R is padded to smallest 3*m for some
    // integer m, otherwise R is padded to smallest 6*m for some integer m
    int padded_R = 0;
    if(R_stride == 1)
    {
        padded_R = Ceiling(R, TILE);
    }
    else
    {
        if(R % TILE_X2 == 1)
        {
            padded_R = Ceiling(R, TILE);
        }
        else
        {
            padded_R = Ceiling(R, TILE_X2);
        }
    }
    // Check C restrictions:
    // For FP16, all C restrictions shall be multipled by 2.
    // This implicitly introduces restriction that C must be even.
    if(fp16 && C % 2 != 0)
    {
        return false;
    }
    // If stride == 1 and S <= 3 then C needs to be even, otherwise not
    if(S_stride == 1 && S <= TILE && C % (fp16 ? 4 : 2) != 0)
    {
        return false;
    }
    const bool is_dilated_stride_2 = (params.direction.IsBackwardData() && S_stride != 1);
    if(fp16)
    {
        if(is_dilated_stride_2)
        {
            if(C % 4 != 0)
                return false;
            // In dilation mode with stride== 2 the following should be satisfied:
            // C * (ceil(R/6) + floor((R+4)/6)) * ceil(S/6) >= 18*2 (fp16)
            const int k = CeilDiv(R, TILE_X2) + FloorDiv((R + TILE + 1), TILE_X2);
            const int l = CeilDiv(S, TILE_X2);
            if(C * k * l < 18 * 2)
                return false;
        }
        if(padded_R * padded_S * C < TILE * TILE * 18 * 2)
            return false;
    }
    else
    {
        // 9_0_14 readme: Additional limitations in the dilated case are R> 1 and  C %2==0
        if(is_dilated_stride_2)
        {
            if(!(R > 1))
                return false;
            if(!(C % 2 == 0))
                return false;
        }
        // If the padded_R x padded_S filter size from above is 3*k x 3*l
        // or (special case for dilated with stride 2) 3*k x 6*l, then
        // it should be that k*l*C  >=18
        assert(padded_R % TILE == 0 && padded_S % (is_dilated_stride_2 ? TILE_X2 : TILE) == 0);
        const int k = padded_R / TILE;
        const int l = padded_S / (is_dilated_stride_2 ? TILE_X2 : TILE);
        if(k * l * C < 18)
            return false;
    }
    // Padding for bwd data shall not be negative.
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
        && (K * OH * OW) <= std::pow(2, 28)
        && (K * R * S) <= std::pow(2, 28)
        && (C * R * S) <= std::pow(2, 28);
    // clang-format on
}

namespace miopen {
namespace solver {

bool ConvBinWinogradRxS::IsApplicable(const ConvolutionContext& params) const
{
    if(!(params.IsFp32() || params.IsFp16()))
        return false;
    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS{}))
        return false;
    if(params.direction.IsBackwardWrW())
    {
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW{}))
            return false;
        if(!(params.IsFp32() && params.kernel_stride_w == 1 && params.kernel_stride_h == 1))
            return false; // WrW is only for fp32 and no stride for now.
    }

    const bool fp16 = params.IsFp16();
    if(fp16 || params.direction.IsBackwardWrW())
    { // These are supplied in asm source format.
        if(!params.use_asm_kernels)
            return false;
    }
    else
    { // fp32 Fwd/Bwd tile=3 kernels are in binary format.
        if(!params.use_binaries)
            return false;
    }

    const auto name = params.GetStream().GetDeviceName();
    // clang-format off
    if (fp16)
    {
        if (! (name == "gfx906" && params.rmv == rocm_meta_version::AMDHSA_1_0))
            return false;
    }
    else
    {
        if (params.direction.IsBackwardWrW())
        {
            if (! ((name == "gfx900" && params.rmv == rocm_meta_version::AMDHSA_1_0)
                || (name == "gfx906" && params.rmv == rocm_meta_version::AMDHSA_1_0)))
                return false;
        }
        else
        {
            if (! ((name == "gfx803" && (params.rmv == rocm_meta_version::V3 || params.rmv == rocm_meta_version::AMDHSA_1_0))
                || (name == "gfx900" && (params.rmv == rocm_meta_version::V3 || params.rmv == rocm_meta_version::AMDHSA_1_0))
                || (name == "gfx906" && params.rmv == rocm_meta_version::AMDHSA_1_0)))
                return false;
        }
    }

    if (! (params.kernel_stride_w <= 2 // -u inp_u 1 or 2
        && params.kernel_stride_w == params.kernel_stride_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.bias == 0
        && params.group_counts == 1
        && params.in_layout == "NCHW"))
        return false;
    // clang-format on

    if(params.direction.IsBackwardWrW())
    {
        return IsShaderContraintsMet(params.in_height,
                                     params.in_width,
                                     params.kernel_dilation_h,
                                     params.kernel_dilation_w,
                                     params.batch_sz, // N
                                     params.n_inputs, // K
                                     params.out_height,
                                     params.out_width,
                                     params.kernel_size_h,
                                     params.kernel_size_w,
                                     params.n_outputs, // C
                                     params,
                                     fp16,
                                     2);
    }
    else
    {
        return IsShaderContraintsMet(params.kernel_size_h, // RxS
                                     params.kernel_size_w,
                                     params.kernel_stride_h,
                                     params.kernel_stride_w,
                                     params.n_inputs,  // C
                                     params.n_outputs, // K
                                     params.in_height, // HxW
                                     params.in_width,
                                     params.out_height, // OHxOW
                                     params.out_width,
                                     params.batch_sz, // N
                                     params,
                                     fp16,
                                     3);
    }
}

ConvSolution ConvBinWinogradRxS::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    const auto n_groups = params.GetStream().GetMaxComputeUnits();
    const auto name     = params.GetStream().GetDeviceName();
    KernelInfo kernel;

    kernel.g_wk.push_back(512 * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    if(params.IsFp16())
    {
        kernel.kernel_name = "sp3AsmConvRxSU";
        kernel.kernel_file = "Conv_Winograd_";
        if(miopen::IsEnabled(MIOPEN_DEBUG_SRAM_EDC_DISABLED{}))
            kernel.kernel_file += "v13_3_12";
        else
            kernel.kernel_file += "v14_3_3";
        kernel.kernel_file += "_fp16dot_stride";

        if(params.kernel_stride_w == 2)
        {
            if(params.direction.IsForward())
                kernel.kernel_file += "2_dec";
            else
                kernel.kernel_file += "2_dil";
        }
        else
        {
            kernel.kernel_file += "1";
        }
        kernel.kernel_file += ".s";
    }
    else if(params.direction.IsBackwardWrW())
    {
        kernel.kernel_name = "sp3AsmConvRxSU";
        kernel.kernel_file = "Conv_Winograd_v16_3_0_stride1.s";
    }
    else
    {
        kernel.kernel_name = "sp3AsmConvRxSU";
        kernel.kernel_file = "conv_3x3_wheel_alpha_v9_0_15";

        if(params.kernel_stride_w == 2)
        {
            if(params.direction.IsForward())
                kernel.kernel_file += "_stride_2_dec";
            else
                kernel.kernel_file += "_stride_2_dil";
        }

        kernel.kernel_file += ("_" + name);

        if(params.rmv == rocm_meta_version::V3)
            kernel.kernel_file += "_m30";
        else if(params.rmv == rocm_meta_version::AMDHSA_1_0)
            kernel.kernel_file += "_md10";
        else
            MIOPEN_THROW("ConvBinWinogradRxS: Unsupported metadata version.");

        kernel.kernel_file += ".so";
    }
    result.construction_params.push_back(kernel);
    return result;
}
} // namespace solver
} // namespace miopen
