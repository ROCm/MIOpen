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
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>

#include "half.hpp"

using half_float::half;

namespace miopen {
namespace solver {

bool PerformanceConfigConvBiasActivAsm1x1U::IsValid(const ConvolutionContext& config) const
{
    const int sgprs = 25 + 7 + 2 * k_mult * c_mult;
    if(!(sgprs < 102)) /// \todo This is valid for Gfx8 and Gfx9. Check for newer parts.
        return false;

    return PerformanceConfigConvAsm1x1U::IsValid(config);
}

inline bool PerformanceConfigConvBiasActivAsm1x1U::
operator==(const PerformanceConfigConvBiasActivAsm1x1U& other) const
{
    // clang-format off
            return read_size == other.read_size
                && k_mult == other.k_mult
                && chunks_per_wave == other.chunks_per_wave
                && chunk_size == other.chunk_size
                && n_mult == other.n_mult
                && c_mult == other.c_mult
                && waves_c_in_group == other.waves_c_in_group
                && waves_k_in_group == other.waves_k_in_group
                && use_spare_set == other.use_spare_set; // clang-format on
}

PerformanceConfigConvBiasActivAsm1x1U
ConvBiasActivAsm1x1U::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvBiasActivAsm1x1U pp;
    pp.EuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

int ConvBiasActivAsm1x1U::RunAndMeasureSolution(miopen::Handle& profile_h,
                                                Data_t bot_ocl_buf,
                                                Data_t top_ocl_buf,
                                                Data_t wei_ocl_buf,
                                                Data_t bias_ocl_buf,
                                                const ConvolutionContext& params,
                                                const ConvSolution& solution,
                                                float& elapsed_time) const
{
    assert(bias_ocl_buf != nullptr);
    (void)params;
    KernelInfo k_info;
    k_info = solution.construction_params[0];

    std::ostringstream cba_options;
    GenerateClangDefsym(cba_options, "activ_mode", 3);
    GenerateClangDefsym(cba_options, "bias_mode", 1);
    GenerateClangDefsym(cba_options, "fusion_mode", 1);
    GenerateClangDefsym(cba_options, "enable_activ", 1);

#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        // ConvolutionContext::general_compile_options is for OpenCL kernels
        // and thus not applicable for assembly.
        auto kernel = profile_h.AddKernel("",
                                          "",
                                          "conv1x1u_bias_activ.s",
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options + cba_options.str());

        if(params.out_data_type == miopenHalf)
        {
            short unused = 0;
            auto alpha   = half(1.0);
            auto beta    = half(0.0);
            auto gamma   = half(1.0);
            kernel(alpha, beta, gamma, unused, bot_ocl_buf, top_ocl_buf, wei_ocl_buf, bias_ocl_buf);
        }
        else
        {
            int unused  = 0;
            float alpha = 1.0;
            float beta  = 0.0;
            float gamma = 1.0;
            kernel(alpha, beta, gamma, unused, bot_ocl_buf, top_ocl_buf, wei_ocl_buf, bias_ocl_buf);
        }

        elapsed_time = profile_h.GetKernelTime();
    }
#ifdef NDEBUG
    catch(miopen::Exception&)
    {
        return -1;
    }
#endif
    return 0;
}

PerformanceConfigConvBiasActivAsm1x1U
ConvBiasActivAsm1x1U::Search(const ConvolutionContext& context) const
{
    ConvolutionContext cba_context = context;

    cba_context.bias    = 1;
    cba_context.bias_sz = cba_context.n_outputs * ((context.out_data_type == miopenHalf) ? 2 : 4);
    return GenericSearch(*this, cba_context);
}

} // namespace solver
} // namespace miopen
