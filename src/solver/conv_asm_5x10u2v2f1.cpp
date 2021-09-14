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

bool ConvAsm5x10u2v2f1::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2{}))
        return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.Is2d())
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!params.rmv.IsV2orV3())
        return false;

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = params.GetStream().GetDeviceName();
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
    if(!params.direction.IsForward())
    {
        return false;
    }
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    // Min image + padding shall be not smaller than filter matrix.
    const int min_in_width  = params.kernel_size_w - params.pad_w * 2;
    const int min_in_height = params.kernel_size_h - params.pad_h * 2;
    // These two found experimentally.
    const int max_in_width  = 8192 - 1;
    const int max_in_height = 131077 - 1;

    // clang-format off
    return 0 <= params.pad_w && params.pad_w <= 5 // -q   pad_w   // [0..5] for now FIXME
        && 0 <= params.pad_h && params.pad_h <= 5 // -p   pad_h   // [0..5] for now FIXME
        && params.kernel_stride_w == 2           // -v   inp_v   fixed
        && params.kernel_stride_h == 2           // -u   inp_u   fixed
        && params.kernel_size_w == 10            // -x   wei_w   fixed
        && params.kernel_size_h == 5             // -y   wei_h   fixed
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.n_inputs >= 1                 // -c   wei_c   no upper limit
        && params.n_outputs % 16 == 0           // -k   wei_k   no upper limit
        && params.n_outputs >= 1
        && params.in_width >= min_in_width      // -W   inp_w
        && params.in_width <= max_in_width
        && params.in_height >= min_in_height    // -H   inp_h
        && params.in_height <= max_in_height
        && params.IsFp32()
        && params.group_counts == 1
        && params.in_layout == "NCHW";          // hardcoded
        // && (params.forward ? params.weights_layout == "KCHW" : params.weights_layout == "CKHW" )
    // clang-format on
}

static inline int AlignUp(int val, unsigned step)
{
    return static_cast<int>(((static_cast<unsigned>(val) + step - 1) / step) * step);
}

ConvSolution ConvAsm5x10u2v2f1::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    const int out_w =
        (params.in_width + params.pad_w * 2 + params.kernel_stride_w - params.kernel_size_w) /
        params.kernel_stride_w; // (inp_w + 2*pad_w + inp_v - wei_w) / inp_v
    const int out_h =
        (params.in_height + params.pad_h * 2 + params.kernel_stride_h - params.kernel_size_h) /
        params.kernel_stride_h; // (inp_h + 2*pad_h + inp_u - wei_h) / inp_u

    std::ostringstream options;
    GenerateClangDefsym(options, "inp_h", params.in_height);
    GenerateClangDefsym(options, "inp_w", params.in_width);
    GenerateClangDefsym(options, "wei_c", params.n_inputs);
    GenerateClangDefsym(options, "wei_k", params.n_outputs);
    GenerateClangDefsym(options, "wei_layout", 0); // 0: KCHW, 1: CKHW
    GenerateClangDefsym(options, "pad_w", params.pad_w);
    GenerateClangDefsym(options, "pad_h", params.pad_h);
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", params.rmv.UseV3() ? 5 : 4);

    KernelInfo construction_params;
    construction_params.comp_options = options.str();

    construction_params.l_wk.push_back(64);
    construction_params.l_wk.push_back(8);
    construction_params.l_wk.push_back(1);

    // global-work = [align(out_w,64), (align(out_h,4)/4)*align(wei_k/2,8), batch_n]
    construction_params.g_wk.push_back(AlignUp(out_w, 64));
    construction_params.g_wk.push_back(AlignUp(out_h, 4) / 4 * AlignUp(params.n_outputs / 2, 8));
    construction_params.g_wk.push_back(params.batch_sz);

    construction_params.kernel_file = "conv5x10u2v2f1.s";
    construction_params.kernel_name = "miopenConv5x10u2v2f1";

    result.construction_params.push_back(construction_params);
    result.invoker_factory = &conv::MakeGenericXWYPadInvoker;

    return result;
}
} // namespace solver
} // namespace miopen
