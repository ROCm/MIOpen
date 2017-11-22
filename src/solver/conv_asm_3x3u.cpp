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

#include <unordered_map>
#include <sstream>
#include "miopen/env.hpp"
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include "miopen/gcn_asm_utils.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3U_PERF_VALS)

namespace miopen {
namespace solver {

bool ConvAsm3x3U::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.assembler_available)
    {
        return false;
    }
    if(!(params.rmv == rocm_meta_version::V1 || params.rmv == rocm_meta_version::V2 ||
         params.rmv == rocm_meta_version::V3 || params.rmv == rocm_meta_version::AMDHSA_1_0))
    {
        return false;
    }

    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos)
    { // Any gfx8 device is ok.
        return false;
    }
    assert(params.weights_layout.length() == 0); // FIXME _weights_layout is not supported yet.
    return params.pad0 == 1 && params.pad1 == 1 && params.kernel_stride0 == 1 &&
           params.kernel_stride1 == 1 && params.kernel_size0 == 3 && params.kernel_size1 == 3 &&
           params.n_inputs > 0 && params.n_inputs % 4 == 0 && params.in_width > 3 &&
           params.in_width <= 1000 && params.in_layout == "NCHW";
    // && (params.forward ? params.weights_layout == "KCHW" : params.weights_layout == "CKHW" ) //
    // See fixme above.
}

bool ConvAsm3x3U::IsFast(const ConvolutionContext& params) const { return params.in_width >= 50; }

ConvSolution ConvAsm3x3U::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    std::string perf_vals;
    {
        const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3U_PERF_VALS{});
        if(p_asciz && std::strlen(p_asciz) == 3)
        {
            perf_vals = std::string(p_asciz);
        }
    }
    if(perf_vals.empty())
    {
        perf_vals = "220";
        {
            /// Optimal values found on Gfx8 with 56 CUs (R9 Fury).
            /// \todo Test on devices with 64 CUs (e.g. R9 Nano) and expand
            /// implementation if optimal values are different.
            static_assert('9' - '0' == 9, "Characters must be in ASCII encoding");
            static const std::unordered_map<std::string, std::string> perf_vals_map({
                //          W   H   c   n  k   dir  "fpw olpw lwc"
                {MakeLutKey(54, 54, 64, 8, 64, 0), "820"},
                {MakeLutKey(54, 54, 64, 8, 64, 1), "820"},
                {MakeLutKey(56, 56, 128, 8, 256, 0), "840"},
                {MakeLutKey(56, 56, 128, 8, 256, 1), "840"},
                {MakeLutKey(56, 56, 128, 16, 256, 0), "840"},
                {MakeLutKey(56, 56, 128, 16, 256, 1), "840"},
                {MakeLutKey(60, 6, 64, 16, 128, 0), "420"},
                {MakeLutKey(60, 6, 64, 16, 128, 1), "260"},
                {MakeLutKey(112, 112, 64, 8, 128, 0), "820"},
                {MakeLutKey(112, 112, 64, 8, 128, 1), "820"},
                {MakeLutKey(112, 112, 64, 16, 128, 0), "820"},
                {MakeLutKey(112, 112, 64, 16, 128, 1), "820"},
                {MakeLutKey(120, 12, 32, 16, 64, 0), "413"},
                {MakeLutKey(120, 12, 32, 16, 64, 1), "420"},
                {MakeLutKey(240, 24, 16, 16, 32, 0), "420"},
                {MakeLutKey(240, 24, 16, 16, 32, 1), "810"},
            });
            const auto key = params.direction.IsForward() ? MakeLutKey(params.in_width,
                                                                       params.in_height,
                                                                       params.n_inputs,
                                                                       params.batch_sz,
                                                                       params.n_outputs,
                                                                       1)
                                                          : MakeLutKey(params.in_width,
                                                                       params.in_height,
                                                                       params.n_outputs,
                                                                       params.batch_sz,
                                                                       params.n_inputs,
                                                                       0);
            const auto found = perf_vals_map.find(key);
            if(found != perf_vals_map.end())
            {
                perf_vals = found->second;
            }
        }
    }

    const int filters_per_wave      = perf_vals[0] - '0';
    const int output_lines_per_wave = perf_vals[1] - '0';
    const int limit_wave_cnt        = perf_vals[2] - '0';
    const auto w64_chunks           = (params.in_width + 63) / 64;
    const auto active_lanes         = (params.in_width + w64_chunks - 1) / w64_chunks;

    std::ostringstream options;
    GenerateClangDefsym(options, "batch_size", params.batch_sz);
    GenerateClangDefsym(options, "img_width", params.in_width);
    GenerateClangDefsym(options, "img_height", params.in_height);
    GenerateClangDefsym(options, "input_channels", params.n_inputs);
    GenerateClangDefsym(options, "output_channels", params.n_outputs);
    GenerateClangDefsym(options, "weights_layout", params.direction.IsForward() ? 0 : 1);
    GenerateClangDefsym(options, "reverse_weights", params.direction.IsForward() ? 0 : 1);
    GenerateClangDefsym(options, "filters_per_wave", filters_per_wave);
    GenerateClangDefsym(options, "output_lines_per_wave", output_lines_per_wave);
    GenerateClangDefsym(options, "limit_wave_cnt", limit_wave_cnt);
    GenerateClangDefsym(options, "no_params_file", 1);
    GenerateClangDefsym(options, "enable_debug_output", 0);
    GenerateClangDefsym(options,
                        "ROCM_METADATA_VERSION",
                        (params.rmv == rocm_meta_version::V1)
                            ? 1
                            : (params.rmv == rocm_meta_version::V2)
                                  ? 2
                                  : (params.rmv == rocm_meta_version::V3) ? 3 : 4);

    KernelInfo construction_params;
    construction_params.comp_options = options.str();

    construction_params.l_wk.push_back(active_lanes);
    construction_params.l_wk.push_back(1);
    construction_params.l_wk.push_back(1);

    construction_params.g_wk.push_back(
        active_lanes * ((params.n_outputs + filters_per_wave - 1) / filters_per_wave));
    construction_params.g_wk.push_back((params.in_height + output_lines_per_wave - 1) /
                                       output_lines_per_wave);
    construction_params.g_wk.push_back(params.batch_sz);

    construction_params.kernel_file = "conv3x3.s";
    construction_params.kernel_name = "gcnAsmConv3x3U";

    result.construction_params.push_back(construction_params);
    return result;
}
} // namespace solver
} // namespace miopen
