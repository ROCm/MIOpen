/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <cstddef>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include "implicitgemm_util.hpp"
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/conv/asm_implicit_gemm.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS)

namespace miopen {
namespace solver {

static inline std::vector<TunableImplicitGemmGTCDynamic_t>&
GetImplicitGemmWrwGTCDynamicXdlopsKernelList()
{
    // retrieve dynamic igemm wrw pass's possible kernel name
    // clang-format off
    static std::vector<TunableImplicitGemmGTCDynamic_t> kernel_param_list {
        { "wrw", "fp32",   4,   0, 256, 128,  16,  64,  32,   1,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   0},
        { "wrw", "fp32",   4,   0, 256, 128,  16,  64,  32,   1,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   1},
        { "wrw", "fp32",   4,   0, 256, 128,   8,  64,  32,   1,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   2,   1, 128},   {1,   4,   1,   1},   {1,   2,   1, 128},   0},
        { "wrw", "fp32",   4,   0, 256, 128,   8,  64,  32,   1,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   2,   1, 128},   {1,   4,   1,   1},   {1,   2,   1, 128},   1},
        { "wrw", "fp32",   1,   1, 256, 128,  16,  64,  32,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1, 256, 128,  16,  64,  32,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1, 256, 128,   8,  64,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1, 256, 128,   8,  64,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1, 256, 128,  16,  64,  32,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1, 256, 128,  16,  64,  32,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1, 256, 128,   8,  64,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1, 256, 128,   8,  64,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   0, 256,  64,  16,  64,  16,   1,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   0},
        { "wrw", "fp32",   4,   0, 256,  64,  16,  64,  16,   1,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   1},
        { "wrw", "fp32",   1,   1, 256,  64,  16,  64,  16,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1, 256,  64,  16,  64,  16,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1, 256,  64,   8,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1, 256,  64,   8,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1, 256,  64,   4,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   4,   1,  64},   {1,   1,   1,   1},   {1,   4,   1,  64},   0},
        { "wrw", "fp32",   1,   1, 256,  64,   4,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   4,   1,  64},   {1,   1,   1,   1},   {1,   4,   1,  64},   1},
        { "wrw", "fp32",   4,   1, 256,  64,  16,  64,  16,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1, 256,  64,  16,  64,  16,   1,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1, 256,  64,   8,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1, 256,  64,   8,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1, 256,  64,   4,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   4,   1,  64},   {1,   1,   1,   1},   {1,   4,   1,  64},   0},
        { "wrw", "fp32",   4,   1, 256,  64,   4,  64,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   4,   1,  64},   {1,   1,   1,   1},   {1,   4,   1,  64},   1},
        { "wrw", "fp32",   1,   1, 256,  32,  16,  64,   4,   1,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1, 256,  32,  16,  64,   4,   1,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1, 256,  32,   8,  64,   4,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1, 256,  32,   8,  64,   4,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1, 256,  32,  16,  64,   4,   1,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1, 256,  32,  16,  64,   4,   1,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1, 256,  32,   8,  64,   4,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1, 256,  32,   8,  64,   4,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   0, 128, 128,  16,  32,  32,   1,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   0},
        { "wrw", "fp32",   4,   0, 128, 128,  16,  32,  32,   1,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   1},
        { "wrw", "fp32",   1,   1, 128, 128,  16,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1, 128, 128,  16,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1, 128, 128,   8,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1, 128, 128,   8,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1, 128, 128,  16,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1, 128, 128,  16,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1, 128, 128,   8,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1, 128, 128,   8,  32,  32,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   0, 128,  64,  16,  32,   8,   1,   1,   2,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   0},
        { "wrw", "fp32",   4,   0, 128,  64,  16,  32,   8,   1,   1,   2,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   1},
        { "wrw", "fp32",   1,   1, 128,  64,  16,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1, 128,  64,  16,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1, 128,  64,   8,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1, 128,  64,   8,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1, 128,  64,  16,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1, 128,  64,  16,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1, 128,  64,   8,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1, 128,  64,   8,  32,   8,   1,   1,   2,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1, 128,  32,  16,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1, 128,  32,  16,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1, 128,  32,   8,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1, 128,  32,   8,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1, 128,  32,  16,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1, 128,  32,  16,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1, 128,  32,   8,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1, 128,  32,   8,  32,   8,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1,  64, 256,  16,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1,  64, 256,  16,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1,  64, 256,   8,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1,  64, 256,   8,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1,  64, 256,  16,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1,  64, 256,  16,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1,  64, 256,   8,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1,  64, 256,   8,  16,  64,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1,  64, 128,  16,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1,  64, 128,  16,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1,  64, 128,   8,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1,  64, 128,   8,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1,  64, 128,  16,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1,  64, 128,  16,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1,  64, 128,   8,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1,  64, 128,   8,   8,  32,   1,   2,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1,  64,  64,  16,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1,  64,  64,  16,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1,  64,  64,   8,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1,  64,  64,   8,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1,  64,  64,  16,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1,  64,  64,  16,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1,  64,  64,   8,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1,  64,  64,   8,  16,  16,   1,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1,  64,  32,  16,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1,  64,  32,  16,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1,  64,  32,   8,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1,  64,  32,   8,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   4,   1,  64,  32,  16,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1,  64,  32,  16,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1,  64,  32,   8,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   4,   1,  64,  32,   8,  32,   8,   1,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1,  64,  16,  16,  64,   4,   1,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   1,   1,  64,  16,  16,  64,   4,   1,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   1,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   1},
        { "wrw", "fp32",   1,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   0},
        { "wrw", "fp32",   4,   1,  64,  16,  16,  64,   4,   1,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   0},
        { "wrw", "fp32",   4,   1,  64,  16,  16,  64,   4,   1,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   1},
        { "wrw", "fp32",   4,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   1},
        { "wrw", "fp32",   4,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   0},
        { "wrw", "fp32",  16,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   1},
        { "wrw", "fp32",  16,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   0},
        { "wrw", "fp32",  16,   1,  64,   4,  16,  64,   4,   1,   1,   1,   1,   1,   {1,   1,  16,   1},   {1,  16,   1,   4},   {1,   1,   1,   1},   {1,  16,   1,   4},   1},
        { "wrw", "fp32",  16,   1,  64,   4,  16,  64,   4,   1,   1,   1,   1,   1,   {1,   1,  16,   1},   {1,  16,   1,   4},   {1,   1,   1,   1},   {1,  16,   1,   4},   0},
        { "wrw", "fp32",   1,   1,  32,  32,   8,  16,  16,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   1,   1,  32,  32,   8,  16,  16,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   8,   1,  32,  32,   8,  16,  16,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
        { "wrw", "fp32",   8,   1,  32,  32,   8,  16,  16,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "wrw", "fp32",   1,   1,  16,  32,  16,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   0},
        { "wrw", "fp32",   1,   1,  16,  32,  16,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   1},
        { "wrw", "fp32",   1,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   0},
        { "wrw", "fp32",   1,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   1},
        { "wrw", "fp32",   4,   1,  16,  32,  16,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   0},
        { "wrw", "fp32",   4,   1,  16,  32,  16,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   1},
        { "wrw", "fp32",   4,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   0},
        { "wrw", "fp32",   4,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   1},
        { "wrw", "fp32",   8,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   0},
        { "wrw", "fp32",   8,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   1},
        { "wrw", "fp32",   8,   1,  32,  16,   8,  32,   8,   1,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  16},   {1,   1,   1,   1},   {1,   8,   1,  16},   1},
        { "wrw", "fp32",   8,   1,  32,  16,   8,  32,   8,   1,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  16},   {1,   1,   1,   1},   {1,   8,   1,  16},   0},
        // clang-format on
    };
    return kernel_param_list;
}

static inline int find_tunable(const std::vector<TunableImplicitGemmGTCDynamic_t> tunables,
                               const int gemm_m_per_block,
                               const int gemm_n_per_block,
                               const int gemm_k_per_block,
                               const int gemm_k_global_split,
                               const int nxb,
                               const int nxe)
{
    int i;
    for(i = 0; i < tunables.size(); i++)
    {
        if((tunables[i].gemm_m_per_block == gemm_m_per_block) &&
           (tunables[i].gemm_n_per_block == gemm_n_per_block) &&
           (tunables[i].gemm_k_per_block == gemm_k_per_block) &&
           (tunables[i].gemm_k_global_split == gemm_k_global_split) && (tunables[i].nxb == nxb) &&
           (tunables[i].nxe == nxe))
        {
            break;
        }
    }
    return i;
}

static inline int if_gemm_k_global_split(const ConvolutionContext& ctx,
                                         const int gemm_m_per_block,
                                         const int gemm_n_per_block,
                                         const int gemm_k_per_block,
                                         const int b)
{
    int gemm_k_global_split = 0;
    const auto& n           = ctx.batch_sz;
    const auto& k           = ctx.n_inputs;
    const auto& c           = ctx.n_outputs;
    const auto& y           = ctx.kernel_size_h;
    const auto& x           = ctx.kernel_size_w;

    const auto& gemm_m = k;
    const auto gemm_n  = c * y * x;

    int max_grid_size = 1200;

    int grid_size;
    // assume that gemm m/n can be divided with no remainder by gemm m/n per block
    grid_size = (gemm_m / gemm_m_per_block) * (gemm_n / gemm_n_per_block);
    if((n % 2 == 0) && (grid_size < max_grid_size) && ((n >> 1) * b % gemm_k_per_block == 0))
    {
        gemm_k_global_split = 1;
    }
    else
    {
        gemm_k_global_split = 0;
    }
    return gemm_k_global_split;
}

static inline float CallImplicitGemmWrwDynamic(const miopen::Handle& handle,
                                               const conv::ProblemDescription& conv_problem,
                                               ConstData_t src,
                                               ConstData_t dst,
                                               Data_t wei,
                                               const std::vector<KernelInvoke>& kernels,
                                               const int log2_gemm_k_global_splits)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];

    int hi         = conv_problem.GetOutHeight();
    int wi         = conv_problem.GetOutWidth();
    int n          = conv_problem.GetInBatchSize();
    int k          = conv_problem.GetInChannels();
    int c          = conv_problem.GetOutChannels();
    int ho         = conv_problem.GetInHeight();
    int wo         = conv_problem.GetInWidth();
    int stride_h   = conv_problem.GetInHeight() > 1 ? conv_problem.GetKernelStrideH() : 1;
    int stride_w   = conv_problem.GetInWidth() > 1 ? conv_problem.GetKernelStrideW() : 1;
    int dilation_h = conv_problem.GetWeightsHeight() > 1 ? conv_problem.GetDilationH() : 1;
    int dilation_w = conv_problem.GetWeightsWidth() > 1 ? conv_problem.GetDilationW() : 1;
    int pad_h      = conv_problem.GetPadH();
    int pad_w      = conv_problem.GetPadW();
    int y          = conv_problem.GetWeightsHeight();
    int x          = conv_problem.GetWeightsWidth();

    // std::cout << "nchiwi: " << n << " " << c  << " " << hi << " " << wi << std::endl;
    // std::cout << "nkhowo: " << n << " " << k  << " " << ho << " " << wo << std::endl;
    // std::cout << "kcyx: " << k << " " << c  << " " << y << " " << x << std::endl;

    MIOPEN_LOG_I2(kernel.GetName() << " with groups for reduction: "
                                   << (1 << log2_gemm_k_global_splits));

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(wei);
    opArgs.emplace_back(dst);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(log2_gemm_k_global_splits);
    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed = handle.GetKernelTime();

    return elapsed;
}

// find wrw dynamic kernel by a simple algo
// check wether this kernel can be applicable
static inline std::tuple<bool, // is valid
                         int,  // tunable index
                         int,  // block_size
                         int,  // grid_size
                         int>  // gemm_k_split
    FindImplicitGemmWrwGTCDynamicXdlopsKernel(const ConvolutionContext& ctx)
{
    const auto& n         = ctx.batch_sz;
    const auto& k         = ctx.n_inputs;
    const auto& c         = ctx.n_outputs;
    const auto& ho        = ctx.in_height;
    const auto& wo        = ctx.in_width;
    const auto& y         = ctx.kernel_size_h;
    const auto& x         = ctx.kernel_size_w;
    const auto stride_h   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const auto stride_w   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const auto dilation_h = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    const auto dilation_w = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    const auto& pad_h     = ctx.pad_h;
    const auto& pad_w     = ctx.pad_w;

    const auto gemm_n  = c * y * x;
    const auto& gemm_m = k;

    std::vector<TunableImplicitGemmGTCDynamic_t> tunables =
        GetImplicitGemmWrwGTCDynamicXdlopsKernelList();

    /* applicable table (except 128x128 case):
    gemm_m/gemmn        256 64  32  16  4
                --------------------------
                256 |   0  |1  |0  |0  |0
                64  |   1  |1  |0  |0  |1
                32  |   1  |1  |1  |1  |0
                16  |   0  |1  |0  |0  |0

    */
    int max_grid_size                 = 1200;
    int sel_index                     = -1;
    int sel_block_size                = 0;
    int sel_grid_size                 = 0;
    int sel_log2_gemm_k_global_splits = 0;

    int num_cu                = 120;
    std::vector<int> nxb_list = {16, 8, 4, 1};
    std::vector<int> nxe_list = {0, 1};

    // i=log2(gemm_m_per_block*gemm_n_per_block)  to find largest kernel
    // when pack=0, means no need to search with pack image size. when pack=1, we need pack
    for(int pack = 0; pack < 2; pack++)
    {
        // switch l and r to get differnet kernel size like 256*64 or 64*256
        for(int i = 15; i > 7; i--)
        {
            int r, l;
            r = (i + 1) >> 1;
            l = i - r;
            while(l > 1 && r < 9)
            {
                for(int swap = 0; swap < 2; swap++)
                {
                    const auto gemm_m_per_block = swap == 0 ? 1 << r : 1 << l;
                    const auto gemm_n_per_block = swap == 0 ? 1 << l : 1 << r;

                    if(gemm_m % gemm_m_per_block != 0)
                        continue;

                    for(int j = 4; j > 1; j--)
                    {
                        const auto gemm_k_per_block = 1 << j;
                        for(const auto& nxb : nxb_list)
                        {
                            for(const auto& nxe : nxe_list)
                            {
                                const auto b =
                                    pack == 0
                                        ? ho * wo
                                        : (nxe == 0 ? ho * wo : ((ho * wo + nxb - 1) / nxb) * nxb);
                                const auto gemm_k = n * b;
                                if(c % (gemm_n_per_block / (nxe == 0 ? 1 : nxe)) != 0)
                                    continue;
                                if(gemm_k % gemm_k_per_block != 0)
                                    continue;
                                if(nxe == 0)
                                {
                                    if((x != 1) || (y != 1) || (stride_h != 1) || (stride_w != 1) ||
                                       (dilation_h != 1) || (dilation_w != 1) || (pad_h != 0) ||
                                       (pad_w != 0))
                                        continue;
                                }

                                int gemm_k_global_split = if_gemm_k_global_split(
                                    ctx, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block, b);
                                int tunable_index = find_tunable(tunables,
                                                                 gemm_m_per_block,
                                                                 gemm_n_per_block,
                                                                 gemm_k_per_block,
                                                                 gemm_k_global_split,
                                                                 nxb,
                                                                 nxe);
                                if(tunable_index < 0 || tunable_index >= tunables.size())
                                    continue;

                                int log2_gemm_k_global_splits = 0;
                                int grid_size =
                                    (gemm_m / gemm_m_per_block) * (gemm_n / gemm_n_per_block);
                                int block_size = tunables[tunable_index].GetBlockSize();
                                for(int gs = 0; gs < 8; gs++)
                                {
                                    if((grid_size << gs) > max_grid_size)
                                        break;

                                    if((n % (1 << gs)) != 0)
                                    {
                                        break;
                                    }

                                    if((n >> gs) * b % gemm_k_per_block != 0)
                                    {
                                        break;
                                    }
                                    log2_gemm_k_global_splits = gs;
                                }

                                if(gemm_k_global_split == 0)
                                    log2_gemm_k_global_splits = 0;

                                // std::cout << tunable_index << std::endl;
                                grid_size = grid_size << log2_gemm_k_global_splits;

                                if(block_size >= sel_block_size && grid_size > sel_grid_size)
                                {
                                    sel_block_size                = block_size;
                                    sel_grid_size                 = grid_size;
                                    sel_index                     = tunable_index;
                                    sel_log2_gemm_k_global_splits = log2_gemm_k_global_splits;
                                    break;
                                }
                            }
                        }
                        if(sel_grid_size > num_cu * 2)
                            break;
                    }
                    if(sel_grid_size > num_cu * 2)
                        break;
                }
                if(sel_grid_size > num_cu * 2)
                    break;
                r++;
                l--;
            }
            if(sel_grid_size > num_cu)
                break;
        }
    }
    bool is_valid = !(sel_index < 0 || sel_index >= tunables.size());

    return std::make_tuple(
        is_valid, sel_index, sel_block_size, sel_grid_size, sel_log2_gemm_k_global_splits);
}

bool ConvAsmImplicitGemmGTCDynamicWrwXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS{}))
        return false;

    const auto device_name = ctx.GetStream().GetDeviceName();
    if(device_name != "gfx908")
        return false;

    if(!ctx.use_asm_kernels)
        return false;

    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(ctx.group_counts != 1)
        return false;

    bool is_valid;
    std::tie(is_valid, std::ignore, std::ignore, std::ignore, std::ignore) =
        FindImplicitGemmWrwGTCDynamicXdlopsKernel(ctx);

    return is_valid;
}

ConvSolution
ConvAsmImplicitGemmGTCDynamicWrwXdlops::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;

    KernelInfo kernel;
    std::ostringstream options;

    std::vector<TunableImplicitGemmGTCDynamic_t> kernel_configs =
        GetImplicitGemmWrwGTCDynamicXdlopsKernelList();

    bool is_valid;
    int kernel_index;
    int block_size;
    int grid_size;
    int log2_gemm_k_global_splits;
    std::string kernel_name;

    std::tie(is_valid, kernel_index, block_size, grid_size, log2_gemm_k_global_splits) =
        FindImplicitGemmWrwGTCDynamicXdlopsKernel(ctx);

    if(!is_valid)
        MIOPEN_THROW("this kernel should not run with igemm dynamic!");

    kernel_name = kernel_configs[kernel_index].GetKernelName();

    // std::cout << "tuple=" << grid_size << " " << log2_gemm_k_global_splits << std::endl;

    result.workspce_sz = 0;

    kernel.kernel_file = "igemm_wrw_gtc_gfx908.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
    * grid dims is in unit of work item.
    * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
    */
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    kernel.comp_options = options.str();

    MIOPEN_LOG_I2(kernel.kernel_file + ":" + kernel.kernel_name);

    result.construction_params.push_back(kernel);

    const auto& conv_problem = ctx.conv_problem;

    result.invoker_factory = [conv_problem,
                              log2_gemm_k_global_splits](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k_wrw) { return handle.Run(k_wrw); });
            float elapsed = 0;
            float zero    = 0.f;

            SetTensor(handle, tensors.dwDesc, tensors.dw, &zero);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();

            elapsed += CallImplicitGemmWrwDynamic(handle,
                                                  conv_problem,
                                                  tensors.x,
                                                  tensors.dy,
                                                  tensors.dw,
                                                  ks,
                                                  log2_gemm_k_global_splits);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    return result;
}

} // namespace solver
} // namespace miopen
