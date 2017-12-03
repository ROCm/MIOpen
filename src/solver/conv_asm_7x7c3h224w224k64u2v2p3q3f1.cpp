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
#include "miopen/solver.hpp"
#include "miopen/gcn_asm_utils.hpp"

namespace miopen {
namespace solver {

bool ConvAsm7x7c3h224w224k64u2v2p3q3f1::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.assembler_available)
    {
        return false;
    }
    if(!((params.rmv == rocm_meta_version::V3) || (params.rmv == rocm_meta_version::AMDHSA_1_0)))
    {
        return false;
    }

    const std::string name = params.GetStream().GetDeviceName();
    if(!(name == "gfx800" || name == "gfx802" || name == "gfx803" || name == "gfx804" ||
         name == "gfx900"))
    {
        return false;
    }
    if(!params.direction.IsForward())
    {
        return false;
    }
    assert(params.weights_layout.length() == 0); // FIXME _weights_layout is not supported yet.

    // Opt. Param   Restrictions in source
    return params.pad0 == 3               // -q
           && params.pad1 == 3            // -p
           && params.kernel_stride0 == 2  // -u
           && params.kernel_stride1 == 2  // -v
           && params.kernel_size0 == 7    // -x
           && params.kernel_size1 == 7    // -y
           && params.n_inputs == 3        // -c
           && params.n_outputs == 64      // -k
           && params.in_width == 224      // -W
           && params.in_height == 224     // -H
           && params.in_layout == "NCHW"; //              hardcoded
    // && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" )
}

ConvSolution ConvAsm7x7c3h224w224k64u2v2p3q3f1::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    const int out_w =
        (params.in_width + params.pad0 * 2 + params.kernel_stride0 - params.kernel_size0) /
        params.kernel_stride0; // (inp_w + 2*pad_w + inp_u - wei_w) / inp_u
    const int out_h =
        (params.in_height + params.pad1 * 2 + params.kernel_stride1 - params.kernel_size1) /
        params.kernel_stride1; // (inp_h + 2*pad_h + inp_v - wei_h) / inp_v

    std::ostringstream options;
    GenerateClangDefsym(
        options, "ROCM_METADATA_VERSION", (params.rmv == rocm_meta_version::V3) ? 3 : 4);
    KernelInfo constr_params;
    constr_params.comp_options = options.str();

    constr_params.l_wk.push_back(64);
    constr_params.l_wk.push_back(8);
    constr_params.l_wk.push_back(1);

    // global-work = [align(out_w,64), (align(out_h,4)/4)*align(wei_k/2,8), batch_n]
    constr_params.g_wk.push_back(AlignUp(out_w, 64));
    constr_params.g_wk.push_back(AlignUp(out_h, 4) / 4 * AlignUp(params.n_outputs / 2, 8));
    constr_params.g_wk.push_back(params.batch_sz);

    constr_params.kernel_file = "conv7x7c3h224w224k64u2v2p3q3f1.s";
    constr_params.kernel_name = "gcnAsmConv7x7c3h224w224k64u2v2p3q3f1";

    result.construction_params.push_back(constr_params);
    return result;
}
} // namespace solver
} // namespace miopen
