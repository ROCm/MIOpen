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

#include <miopen/conv/solvers.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>
#include <miopen/mlo_internal.hpp>

#define WORKAROUND_ISSUE_1146 1 // check asm solver applicability for gfx90a

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvAsm5x10u2v2b1::IsApplicable(const ExecutionContext& ctx,
                                     const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!ctx.rmv.IsV2orV3())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = ctx.GetStream().GetDeviceName();
    const bool device_is_gfx8_9_no_xnack =
        (name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804" ||
         name == "gfx900" || name == "gfx904" || name == "gfx906" || name == "gfx908");
#if WORKAROUND_ISSUE_1146
    if(name == "gfx90a")
        return false;
#endif
    if(!device_is_gfx8_9_no_xnack)
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(problem.IsTensorsCasted() || problem.IsFp8() || problem.IsBfp8())
        return false;

    // Min image + padding shall be not smaller than filter matrix.
    const int min_out_width  = 138;
    const int min_out_height = 16;
    // These two found experimentally.
    const int max_out_width  = 8192 - 1;
    const int max_out_height = 131077 - 1;

    // clang-format off
    return problem.GetPadW() == 0                    // -q   pad_w   fixed
        && problem.GetPadH() == 0                    // -p   pad_h   fixed
        && problem.GetKernelStrideW() == 2           // -v   inp_v   fixed
        && problem.GetKernelStrideH() == 2           // -u   inp_u   fixed
        && problem.GetWeightsWidth() == 10          // -x   wei_w   fixed
        && problem.GetWeightsHeight() == 5          // -y   wei_h   fixed
        && problem.GetDilationW() == 1
        && problem.GetDilationH() == 1
        && problem.GetOutChannels() % 16 == 0       // -c   wei_c   no upper limit
        && problem.GetInChannels() >= 16            // -k   wei_k   no upper limit
        && problem.GetOutWidth() >= min_out_width   // -W   inp_w
        && problem.GetOutWidth() <= max_out_width
        && problem.GetOutHeight() >= min_out_height // -H   inp_h
        && problem.GetOutHeight() <= max_out_height
        && problem.IsFp32()
        && problem.GetGroupCount() == 1
        && problem.GetOutLayout() == "NCHW";          // hardcoded
        // && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" )
    // clang-format on
}

ConvSolution ConvAsm5x10u2v2b1::GetSolution(const ExecutionContext& ctx,
                                            const ProblemDescription& problem) const
{
    ConvSolution result;
    std::ostringstream options;
    GenerateClangDefsym(options, "inp_h", problem.GetOutHeight());
    GenerateClangDefsym(options, "inp_w", problem.GetOutWidth());
    GenerateClangDefsym(options, "wei_c", problem.GetOutChannels());
    GenerateClangDefsym(options, "wei_k", problem.GetInChannels());
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    KernelInfo constr_params;
    constr_params.comp_options = options.str();

    constr_params.l_wk.push_back(64);
    constr_params.l_wk.push_back(8);
    constr_params.l_wk.push_back(1);

    // global-work = [align(out_w,64), (align(out_h,4)/4)*align(wei_c/2,8), batch_n]
    constr_params.g_wk.push_back(AlignUp(problem.GetInWidth(), 64));
    constr_params.g_wk.push_back(static_cast<size_t>(AlignUp(problem.GetInHeight(), 4) / 4 *
                                                     AlignUp(problem.GetOutChannels() / 2, 8)));
    constr_params.g_wk.push_back(problem.GetBatchSize());

    constr_params.kernel_file = "conv5x10u2v2b1.s";
    constr_params.kernel_name = "miopenConv5x10u2v2b1";

    result.construction_params.push_back(constr_params);
    result.invoker_factory = &miopen::conv::MakeGenericXWYPadInvoker;
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
