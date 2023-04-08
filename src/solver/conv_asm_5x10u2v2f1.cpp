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

#include <miopen/solver.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/handle.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>

#define WORKAROUND_ISSUE_1146 1 // check asm solver applicability for gfx90a

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2)

namespace miopen {
namespace solver {

bool ConvAsm5x10u2v2f1::IsApplicable(const ExecutionContext& ctx,
                                     const ProblemDescription& problem) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2{}))
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
    const bool device_is_gfx8_9_no_xnack =
        (name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804" ||
         name == "gfx900" || name == "gfx904" || name == "gfx906" || name == "gfx908");
#if WORKAROUND_ISSUE_1146
    if(name == "gfx90a")
        return false;
#endif
    if(!device_is_gfx8_9_no_xnack)
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

    // Min image + padding shall be not smaller than filter matrix.
    const int min_in_width  = problem.kernel_size_w - problem.pad_w * 2;
    const int min_in_height = problem.kernel_size_h - problem.pad_h * 2;
    // These two found experimentally.
    const int max_in_width  = 8192 - 1;
    const int max_in_height = 131077 - 1;

    // clang-format off
    return 0 <= problem.pad_w && problem.pad_w <= 5 // -q   pad_w   // [0..5] for now FIXME
        && 0 <= problem.pad_h && problem.pad_h <= 5 // -p   pad_h   // [0..5] for now FIXME
        && problem.kernel_stride_w == 2           // -v   inp_v   fixed
        && problem.kernel_stride_h == 2           // -u   inp_u   fixed
        && problem.kernel_size_w == 10            // -x   wei_w   fixed
        && problem.kernel_size_h == 5             // -y   wei_h   fixed
        && problem.kernel_dilation_w == 1
        && problem.kernel_dilation_h == 1
        && problem.n_inputs >= 1                 // -c   wei_c   no upper limit
        && problem.n_outputs % 16 == 0           // -k   wei_k   no upper limit
        && problem.n_outputs >= 1
        && problem.in_width >= min_in_width      // -W   inp_w
        && problem.in_width <= max_in_width
        && problem.in_height >= min_in_height    // -H   inp_h
        && problem.in_height <= max_in_height
        && problem.IsFp32()
        && problem.group_counts == 1
        && problem.in_layout == "NCHW";          // hardcoded
        // && (problem.forward ? problem.weights_layout == "KCHW" : problem.weights_layout == "CKHW" )
    // clang-format on
}

static inline int AlignUp(int val, unsigned step)
{
    return static_cast<int>(((static_cast<unsigned>(val) + step - 1) / step) * step);
}

ConvSolution ConvAsm5x10u2v2f1::GetSolution(const ExecutionContext& ctx,
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
    GenerateClangDefsym(options, "inp_h", problem.in_height);
    GenerateClangDefsym(options, "inp_w", problem.in_width);
    GenerateClangDefsym(options, "wei_c", problem.n_inputs);
    GenerateClangDefsym(options, "wei_k", problem.n_outputs);
    GenerateClangDefsym(options, "wei_layout", 0); // 0: KCHW, 1: CKHW
    GenerateClangDefsym(options, "pad_w", problem.pad_w);
    GenerateClangDefsym(options, "pad_h", problem.pad_h);
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    KernelInfo construction_params;
    construction_params.comp_options = options.str();

    construction_params.l_wk.push_back(64);
    construction_params.l_wk.push_back(8);
    construction_params.l_wk.push_back(1);

    // global-work = [align(out_w,64), (align(out_h,4)/4)*align(wei_k/2,8), batch_n]
    construction_params.g_wk.push_back(AlignUp(out_w, 64));
    construction_params.g_wk.push_back(
        static_cast<size_t>(AlignUp(out_h, 4) / 4 * AlignUp(problem.n_outputs / 2, 8)));
    construction_params.g_wk.push_back(problem.batch_sz);

    construction_params.kernel_file = "conv5x10u2v2f1.s";
    construction_params.kernel_name = "miopenConv5x10u2v2f1";

    result.construction_params.push_back(construction_params);
    result.invoker_factory = &conv::MakeGenericXWYPadInvoker;

    return result;
}
} // namespace solver
} // namespace miopen
