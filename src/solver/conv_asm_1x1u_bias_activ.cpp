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

#include <miopen/conv/fused_data_invoke_params.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/solver.hpp>

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
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

PerformanceConfigConvBiasActivAsm1x1U
ConvBiasActivAsm1x1U::Search(const ConvolutionContext& context, const AnyInvokeParams&) const
{
    auto cba_context    = context;
    cba_context.bias    = 1;
    cba_context.bias_sz = cba_context.n_outputs * ((context.out_data_type == miopenHalf) ? 2 : 4);
    if(!context.direction.IsForward())
        MIOPEN_THROW("Only inference supported.");

    /// Workaround: Fused conv API does not pass user-allocated buffers here,
    /// but we need these buffers for search.
    auto& handle        = cba_context.GetStream();
    const auto bias_buf = handle.Create(cba_context.bias_sz);
    const auto in_buf   = handle.Create(cba_context.bot_sz);
    const auto wei_buf  = handle.Create(cba_context.weights_sz);
    const auto out_buf  = handle.Create(cba_context.top_sz);

    auto tensors    = FusedConvDataTensors{};
    tensors.in      = in_buf.get();
    tensors.w       = wei_buf.get();
    tensors.out     = out_buf.get();
    tensors.inDesc  = context.conv_problem.GetIn();
    tensors.wDesc   = context.conv_problem.GetWeights();
    tensors.outDesc = context.conv_problem.GetOut();
    tensors.bias    = bias_buf.get();

    const auto fused_invoke_ctx = conv::FusedDataInvokeParams(tensors, nullptr, 0);

    return GenericSearch(*this, cba_context, fused_invoke_ctx);
}

ConvSolution ConvBiasActivAsm1x1U::GetSolution(const ConvolutionContext& params,
                                               const PerformanceConfigConvAsm1x1U& config,
                                               bool disableConfigOverrideFromEnv) const
{
    auto sol = ConvAsm1x1U::GetSolution(params, config, disableConfigOverrideFromEnv);

    if(sol.construction_params.size() != 1)
        MIOPEN_THROW("ConvBiasActivAsm1x1U expects only one kernel");

    auto& kernel_info       = sol.construction_params[0];
    kernel_info.kernel_file = "conv1x1u_bias_activ.s";

    if(params.is_for_generic_search)
    {
        std::ostringstream cba_options;
        GenerateClangDefsym(cba_options, "activ_mode", 3);
        GenerateClangDefsym(cba_options, "bias_mode", 1);
        GenerateClangDefsym(cba_options, "fusion_mode", 1);
        GenerateClangDefsym(cba_options, "enable_activ", 1);
        kernel_info.comp_options += cba_options.str();
    }

    const auto out_data_type = params.conv_problem.GetOutDataType();

    sol.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& kernel       = handle.Run(kernels[0]);
            const auto& invoke_ctx   = primitive_parameters.CastTo<conv::FusedDataInvokeParams>();
            const auto& tensors      = invoke_ctx.tensors;
            const auto& bot_ocl_buf  = tensors.in;
            const auto& wei_ocl_buf  = tensors.w;
            const auto& top_ocl_buf  = tensors.out;
            const auto& bias_ocl_buf = tensors.bias;

            if(out_data_type == miopenHalf)
            {
                short unused = 0;
                auto alpha   = half(1.0);
                auto beta    = half(0.0);
                auto gamma   = half(1.0);
                kernel(alpha,
                       beta,
                       gamma,
                       unused,
                       bot_ocl_buf,
                       top_ocl_buf,
                       wei_ocl_buf,
                       bias_ocl_buf);
            }
            else
            {
                int unused  = 0;
                float alpha = 1.0;
                float beta  = 0.0;
                float gamma = 1.0;
                kernel(alpha,
                       beta,
                       gamma,
                       unused,
                       bot_ocl_buf,
                       top_ocl_buf,
                       wei_ocl_buf,
                       bias_ocl_buf);
            }
        };
    };

    return sol;
}

} // namespace solver
} // namespace miopen
