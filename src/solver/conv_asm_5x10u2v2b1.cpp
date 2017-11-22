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
#include "miopen/solver.hpp"
#include "miopen/gcn_asm_utils.hpp"

namespace miopen {
namespace solver {

bool ConvAsm5x10u2v2b1::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.assembler_available)
    {
        return false;
    }
    if(!(params.rmv == rocm_meta_version::V1 || params.rmv == rocm_meta_version::V3 ||
         params.rmv == rocm_meta_version::AMDHSA_1_0))
    {
        return false;
    }

    const std::string name = params.GetStream().GetDeviceName();
    const bool device_is_gfx8_9_no_xnack =
        (name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804" ||
         name == "gfx900");
    if(!device_is_gfx8_9_no_xnack)
    {
        return false;
    }
    if(!params.direction.IsBackwardData())
    {
        return false;
    }
    assert(params.weights_layout.length() == 0); // FIXME _weights_layout is not supported yet.

    // Min image + padding shall be not smaller than filter matrix.
    const int min_out_width  = 138;
    const int min_out_height = 16;
    // These two found experimentally.
    const int max_out_width  = 8192 - 1;
    const int max_out_height = 131077 - 1;

    return                                   // Opt. Param   Restrictions in source
        params.pad0 == 0                     // -q   pad_w   fixed
        && params.pad1 == 0                  // -p   pad_h   fixed
        && params.kernel_stride0 == 2        // -u   inp_u   fixed
        && params.kernel_stride1 == 2        // -v   inp_v   fixed
        && params.kernel_size0 == 10         // -x   wei_w   fixed
        && params.kernel_size1 == 5          // -y   wei_h   fixed
        && params.n_outputs % 16 == 0        // -c   wei_c   no upper limit
        && params.n_inputs >= 16             // -k   wei_k   no upper limit
        && params.out_width >= min_out_width // -W   inp_w
        && params.out_width <= max_out_width && params.out_height >= min_out_height // -H   inp_h
        && params.out_height <= max_out_height &&
        params.out_layout == "NCHW"; //              hardcoded
    // && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" ) // See
    // fixme above.
}

ConvSolution ConvAsm5x10u2v2b1::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    std::ostringstream options;
    GenerateClangDefsym(options, "inp_h", params.out_height);
    GenerateClangDefsym(options, "inp_w", params.out_width);
    GenerateClangDefsym(options, "wei_c", params.n_outputs);
    GenerateClangDefsym(options, "wei_k", params.n_inputs);
    GenerateClangDefsym(
        options,
        "ROCM_METADATA_VERSION",
        (params.rmv == rocm_meta_version::V1) ? 1 : (params.rmv == rocm_meta_version::V3) ? 3 : 4);

    KernelInfo constr_params;
    constr_params.comp_options = options.str();

    constr_params.l_wk.push_back(64);
    constr_params.l_wk.push_back(8);
    constr_params.l_wk.push_back(1);

    // global-work = [align(out_w,64), (align(out_h,4)/4)*align(wei_c/2,8), batch_n]
    constr_params.g_wk.push_back(AlignUp(params.in_width, 64));
    constr_params.g_wk.push_back(AlignUp(params.in_height, 4) / 4 *
                                 AlignUp(params.n_outputs / 2, 8));
    constr_params.g_wk.push_back(params.batch_sz);

    constr_params.kernel_file = "conv5x10u2v2b1.s";
    constr_params.kernel_name = "conv5x10u2v2b1";

    result.construction_params.push_back(constr_params);
    return result;
}
} // namespace solver
} // namespace miopen
