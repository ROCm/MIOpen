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

/// \return v rounded up (towards +inf) to the nearest multiple of m.
/// Defined for positive values only.
static int Ceiling(const int v, const int m)
{
    assert(m > 0 && v >= 0);
    if(v % m != 0)
    {
        return (v / m + 1) * m;
    }
    return v;
}

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_RXS)

namespace miopen {
namespace solver {

bool ConvBinWinogradRxS::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.use_binaries)
    {
        return false;
    }

    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_RXS{}))
    {
        return false;
    }

    if(!(params.rmv == rocm_meta_version::V3 || params.rmv == rocm_meta_version::AMDHSA_1_0))
    {
        return false;
    }
    if(!params.direction.IsForward()) // FIXME
    {
        return false;
    }
    const auto name                    = params.GetStream().GetDeviceName();
    const auto device_is_gfx9_no_xnack = (name == "gfx900");
    const bool device_is_gfx8_no_xnack =
        (name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804");
    if(!device_is_gfx8_no_xnack && !device_is_gfx9_no_xnack)
    {
        return false;
    }
    // Check if kernel is suitable for the problem description
    // and able to correctly run with given parameters.
    // Calculate padded filter size first.
    // If stride = 1: if S <= 3 it is padded to 3,
    // otherwise S is padded to smallest 6*n for some integer n
    // If stride = 2: S is always padded to smallest 6*n for some integer n
    int padded_S = 0;
    if(params.kernel_stride0 == 1)
    {
        if(params.kernel_size0 <= 3)
        {
            padded_S = 3;
        }
        else
        {
            padded_S = Ceiling(params.kernel_size0, 6);
        }
    }
    else
    {
        padded_S = Ceiling(params.kernel_size0, 6);
    }
    // If stride = 1: R is always padded to smallest 3*m for some integer m
    // If stride = 2: if R % 6 ==1 then R is padded to smallest 3*m for some
    // integer m,
    // otherwise R is padded to smallest 6*m for some integer m
    int padded_R = 0;
    if(params.kernel_stride1 == 1)
    {
        padded_R = Ceiling(params.kernel_size1, 3);
    }
    else
    {
        if(params.kernel_size1 % 6 == 1)
        {
            padded_R = Ceiling(params.kernel_size1, 3);
        }
        else
        {
            padded_R = Ceiling(params.kernel_size1, 6);
        }
    }
    // Check C restrictions:
    // If stride == 1 and S <= 3 then C needs to be even, otherwise not
    if(params.kernel_stride0 == 1 && params.kernel_size0 <= 3 && params.n_inputs % 2 != 0)
    {
        return false;
    }
    // If the padded filter size from above is 3*k x 3*l, then it should be that
    // k*l*C  >=18
    {
        assert(padded_R % 3 == 0 && padded_S % 3 == 0);
        const int k = padded_R / 3;
        const int l = padded_S / 3;
        if(k * l * params.n_inputs < 18)
        {
            return false;
        }
    }
    const auto grid_workgroup_count_x = params.GetStream().GetMaxComputeUnits();
    assert(params.weights_layout.length() == 0); // FIXME _weights_layout is not supported yet.
    // clang-format off
    return params.pad0 < std::pow(2, 16)                     // -q   pad_w   uint16
           && params.pad1 < std::pow(2, 16)                  // -p   pad_h   uint16
           && params.kernel_size0 < std::pow(2, 16)          // -x   wei_w   S uint16
           && params.kernel_size1 < std::pow(2, 16)          // -y   wei_h   R uint16
           && params.kernel_stride0 <= 2                     // -u   inp_u   1 or 2
           && params.kernel_stride1 <= 2                     // -v   inp_v   1 or 2
           && params.kernel_stride0 == params.kernel_stride1 // Stride 1x1 or 2x2.
           && params.kernel_dilation0 <= 1
           && params.kernel_dilation1 <= 1
           && params.kernel_dilation0 == params.kernel_dilation1 // Dilation 1x1.
           && params.bias == 0
           && params.batch_sz < std::pow(2, 16)
           && params.n_inputs < std::pow(2, 16)  // -c   wei_c
           && params.n_outputs < std::pow(2, 16) // -k   wei_k
           && params.in_height < std::pow(2, 16) // -H   inp_h
           && params.in_width < std::pow(2, 16)  // -W   inp_w
           && grid_workgroup_count_x < std::pow(2, 16)
           && (params.n_inputs * params.in_height * params.in_width) <= std::pow(2, 28)
           && (params.n_outputs * params.in_height * params.in_width) <= std::pow(2, 28)
           && (params.n_inputs * params.kernel_size0 * params.kernel_size1) <= std::pow(2, 28)
           && (params.n_outputs * params.kernel_size0 * params.kernel_size1) <= std::pow(2, 28)
           && params.in_layout == "NCHW";
    // && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" )
    // clang-format on
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

    kernel.kernel_name = "sp3AsmConvRxSU";
    kernel.kernel_file = "conv";
    if(params.kernel_stride0 == 2)
    {
        kernel.kernel_file += "_u2v2";
    }
    else
    {
        kernel.kernel_file += "_u1v1";
    }
    //    if(params.kernel_dilation0 == 2)
    //    {
    //        kernel.kernel_file += "_l2j2";
    //    }
    //    else
    //    {
    kernel.kernel_file += "_l1j1";
    //    }
    kernel.kernel_file += "_wheel_alpha_v9_0_15";
    if(name.find("gfx8") != std::string::npos)
    {
        kernel.kernel_file += "_gfx803";
    }
    else
    {
        kernel.kernel_file += "_gfx900";
    }

    if(params.rmv == rocm_meta_version::V3)
    {
        kernel.kernel_file += "_m30";
    }
    else if(params.rmv == rocm_meta_version::AMDHSA_1_0)
    {
        kernel.kernel_file += "_md10";
    }
    else
    {
        MIOPEN_THROW("ConvBinWinogradRxS: Unsupported metadata version.");
    }

    kernel.kernel_file += ".so";

    result.construction_params.push_back(kernel);
    return result;
}
} // namespace solver
} // namespace miopen
