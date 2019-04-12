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

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_3X3)

namespace miopen {
namespace solver {

bool ConvBinWinograd3x3U::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.use_binaries)
    {
        return false;
    }

    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_3X3{}))
    {
        return false;
    }

    // Check if device is able to run this kernel.
    const auto name = params.GetStream().GetDeviceName();
    // clang-format off
    if (! ((name == "gfx803" && (params.rmv == rocm_meta_version::V1
                              || params.rmv == rocm_meta_version::V2
                              || params.rmv == rocm_meta_version::V3
                              || params.rmv == rocm_meta_version::AMDHSA_1_0 ))
        || (name == "gfx900" && (params.rmv == rocm_meta_version::V3
                              || params.rmv == rocm_meta_version::AMDHSA_1_0))
        || (name == "gfx906" && params.rmv == rocm_meta_version::AMDHSA_1_0)))
        return false;
    // clang-format on

    // Check if kernel is suitable for the problem description
    // and able to correctly run with given parameters.
    const auto device_is_gfx8         = (name.find("gfx8") != std::string::npos);
    const auto grid_workgroup_count_x = params.GetStream().GetMaxComputeUnits();
    assert(params.weights_layout.length() == 0); // weights_layout is not supported yet.
    // clang-format off
    return params.pad_w == 1
        && params.pad_h == 1
        && params.kernel_size_w == 3
        && params.kernel_size_h == 3
        && params.kernel_stride_w == 1
        && params.kernel_stride_h == 1
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.batch_sz < std::pow(2, 16)
        && params.n_inputs < std::pow(2, 16)
        && params.n_outputs < std::pow(2, 16)
        && params.in_height < std::pow(2, 16)
        && params.in_width < std::pow(2, 16)
        && grid_workgroup_count_x < std::pow(2, 16)
        && (params.n_inputs * params.in_height * params.in_width) <= std::pow(2, 28)
        && (params.n_outputs * params.in_height * params.in_width) <= std::pow(2, 28)
        && (params.n_inputs * params.kernel_size_w * params.kernel_size_h) <= std::pow(2, 28)
        && (params.n_outputs * params.kernel_size_w * params.kernel_size_h) <= std::pow(2, 28)
        && params.n_inputs % 2 == 0 && params.n_inputs >= (device_is_gfx8 ? 16 : 18)
        && params.IsFp32()
        && params.group_counts == 1
        && params.in_layout == "NCHW";
        /// \todo _n_inputs > 18 is a requirement of the v7 shader and NOT a dependency on gfx9
        /// The current way of implemenation is a hack as gfx8 uses v3.0 shader and gfx9 uses v7.
        /// && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" )
        /// Actually, K<->C flpping is controlled by separate flag, so we can support either
        /// layout in both directions.

    // clang-format on
}

ConvSolution ConvBinWinograd3x3U::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    const auto n_groups = params.GetStream().GetMaxComputeUnits();
    const auto name     = params.GetStream().GetDeviceName();

    KernelInfo kernel;

    kernel.g_wk.clear();
    kernel.g_wk.push_back(512 * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.clear();
    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.kernel_name = "sp3AsmConv3x3F";
    if(name == "gfx803")
    {
        if(params.rmv == rocm_meta_version::V1)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m10.so";
        else if(params.rmv == rocm_meta_version::V2)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m21.so";
        else if(params.rmv == rocm_meta_version::V3)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m30.so";
        else if(params.rmv == rocm_meta_version::AMDHSA_1_0)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10.so";
        else
            MIOPEN_THROW("conv_3x3_wheel_alpha_v3_0b_gfx803: Unsupported metadata version.");
    }
    else if(name == "gfx900")
    {
        if(params.rmv == rocm_meta_version::V3)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900.so";
        else if(params.rmv == rocm_meta_version::AMDHSA_1_0)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10.so";
        else
            MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b_gfx900: Unsupported metadata version.");
    }
    else if(name == "gfx906")
    {
        if(params.rmv == rocm_meta_version::AMDHSA_1_0)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx906_md10.so";
        else
            MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b_gfx906: Unsupported metadata version.");
    }
    else
    {
        MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b: Unsupported device.");
    }
    result.construction_params.push_back(kernel);
    return result;
}
} // namespace solver
} // namespace miopen
