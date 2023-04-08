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

#include <sstream>
#include <miopen/solver.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>

#define WORKAROUND_ISSUE_1146 1 // check asm solver applicability for gfx90a

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224)

namespace miopen {
namespace solver {

bool ConvAsm7x7c3h224w224k64u2v2p3q3f1::IsApplicable(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224{}))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!ctx.rmv.IsV2orV3())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = ctx.GetStream().GetDeviceName();
#if WORKAROUND_ISSUE_1146
    if(name == "gfx90a")
        return false;
#endif
    if(!(name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804" ||
         name == "gfx900" || name == "gfx904" || name == "gfx906" || name == "gfx908"))
    {
        return false;
    }
    if(!problem.direction.IsForward())
    {
        return false;
    }
    if(!problem.IsLayoutDefault())
    {
        return false;
    }

    // clang-format off
    return problem.pad_w == 3            // -q
        && problem.pad_h == 3            // -p
        && problem.kernel_stride_w == 2    // -v
        && problem.kernel_stride_h == 2    // -u
        && problem.kernel_size_w == 7    // -x
        && problem.kernel_size_h == 7    // -y
        && problem.kernel_dilation_w == 1
        && problem.kernel_dilation_h == 1
        && problem.n_inputs == 3         // -c
        && problem.n_outputs == 64       // -k
        && problem.in_width == 224       // -W
        && problem.in_height == 224      // -H
        && problem.IsFp32()
        && problem.group_counts == 1
        && problem.in_layout == "NCHW";
        // && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" )
    // clang-format on
}

ConvSolution ConvAsm7x7c3h224w224k64u2v2p3q3f1::GetSolution(const ExecutionContext& ctx,
                                                            const ProblemDescription& problem) const
{
    ConvSolution result;
    const int out_w =
        (problem.in_width + problem.pad_w * 2 + problem.kernel_stride_w - problem.kernel_size_w) /
        problem.kernel_stride_w; // (inp_w + 2*pad_w + inp_v - wei_w) / inp_v
    const int out_h =
        (problem.in_height + problem.pad_h * 2 + problem.kernel_stride_h - problem.kernel_size_h) /
        problem.kernel_stride_h; // (inp_h + 2*pad_h + inp_u - wei_h) / inp_u

    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    KernelInfo constr_params;
    constr_params.comp_options = options.str();

    constr_params.l_wk.push_back(64);
    constr_params.l_wk.push_back(8);
    constr_params.l_wk.push_back(1);

    // global-work = [align(out_w,64), (align(out_h,4)/4)*align(wei_k/2,8), batch_n]
    constr_params.g_wk.push_back(AlignUp(out_w, 64));
    constr_params.g_wk.push_back(
        static_cast<size_t>(AlignUp(out_h, 4) / 4 * AlignUp(problem.n_outputs / 2, 8)));
    constr_params.g_wk.push_back(problem.batch_sz);

    constr_params.kernel_file = "conv7x7c3h224w224k64u2v2p3q3f1.s";
    constr_params.kernel_name = "miopenGcnAsmConv7x7c3h224w224k64u2v2p3q3f1";

    result.construction_params.push_back(constr_params);
    result.invoker_factory = &conv::MakeGenericXWYPadInvoker;
    return result;
}
} // namespace solver
} // namespace miopen
