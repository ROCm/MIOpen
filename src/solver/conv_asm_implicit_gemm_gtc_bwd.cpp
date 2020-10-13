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
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

static inline const std::vector<TunableImplicitGemmGTCDynamic_t>&
GetImplicitGemmGtcDynamicBwdTunablesList()
{
    // clang-format off
    static const std::vector<TunableImplicitGemmGTCDynamic_t> tunables = {
        { "bwd", "fp32",  16,   0, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {2,   1,   1,   4},   {1,   8,   1,  32},   {2,   1,   1,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   4,   0, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {2,   1,   1,   4},   {1,   8,   1,  32},   {2,   1,   1,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {2,   1,   1,   4},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",  16,   0, 128, 128,   8,  32,  32,   1,   1,   2,   2,   {1,   1,   1,   4},   {1,   8,   1,  32},   {1,   1,   1,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   4,   0, 128, 128,   8,  32,  32,   1,   1,   2,   2,   {1,   1,   1,   4},   {1,   8,   1,  32},   {1,   1,   1,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0, 128, 128,   8,  32,  32,   1,   1,   2,   2,   {1,   1,   1,   4},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",  16,   1, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   4,   1, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128, 128,   8,  32,  32,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",  16,   0, 128, 256,  16,  32,  64,   1,   1,   2,   2,   {1,   1,   2,   4},   {1,  16,   1,  16},   {1,   1,   4,   4},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   4,   0, 128, 256,  16,  32,  64,   1,   1,   2,   2,   {1,   1,   2,   4},   {1,  16,   1,  16},   {1,   1,   4,   4},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   0, 128, 256,  16,  32,  64,   1,   1,   2,   2,   {2,   1,   1,   4},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",  16,   0, 128, 256,   8,  32,  64,   1,   1,   2,   2,   {1,   1,   1,   4},   {1,   8,   1,  32},   {1,   1,   2,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   4,   0, 128, 256,   8,  32,  64,   1,   1,   2,   2,   {1,   1,   1,   4},   {1,   8,   1,  32},   {1,   1,   2,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0, 128, 256,   8,  32,  64,   1,   1,   2,   2,   {1,   1,   1,   4},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128, 256,  16,  32,  64,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   4,   1, 128, 256,  16,  32,  64,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",  16,   1, 128, 256,  16,  32,  64,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128, 256,   8,  32,  64,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   4,   0, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {1,   1,   4,   4},   {1,  16,   1,  16},   {1,   1,   2,   4},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",  16,   0, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {1,   1,   4,   4},   {1,  16,   1,  16},   {1,   1,   2,   4},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   0, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {1,   1,   4,   4},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   4,   0, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   1,   2,   4},   {1,   8,   1,  32},   {1,   1,   1,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",  16,   0, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   1,   2,   4},   {1,   8,   1,  32},   {1,   1,   1,   4},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   1,   2,   4},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   4,   1, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {2,   1,   8,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",  16,   1, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {2,   1,   8,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {2,   1,   8,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0, 256,  64,  16,  64,  16,   1,   1,   2,   2,   {2,   1,   8,   1},   {1,   8,   1,  32},   {2,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 256,  64,  16,  64,  16,   1,   1,   2,   2,   {2,   1,   8,   1},   {1,   8,   1,  32},   {2,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 256,  64,   4,  64,  16,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   4,   1,  64},   {1,   1,   1,   1},   {1,   4,   1,  64},   0},
        { "bwd", "fp32",   1,   0, 128,  64,  16,  32,   8,   1,   2,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128,  64,  16,  32,   8,   1,   2,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128,  64,   8,  32,   8,   1,   2,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  64, 256,  16,  16,  64,   1,   1,   2,   2,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64, 256,  16,  16,  64,   1,   1,   2,   2,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64, 256,   8,  16,  64,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  64, 128,  16,   8,  32,   2,   1,   2,   2,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64, 128,  16,   8,  32,   2,   1,   2,   2,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64, 128,   8,   8,  32,   2,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  64,  64,  16,  16,  16,   1,   1,   2,   2,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64,  64,  16,  16,  16,   1,   1,   2,   2,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64,  64,   8,  16,  16,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 256,  32,  16,  64,   4,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   0, 256,  32,  16,  64,   4,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   1, 256,  32,   8,  64,   4,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0, 256,  16,  16,  64,   4,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   1, 256,  16,  16,  64,   4,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   0, 128,  32,  16,  32,   8,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128,  32,  16,  32,   8,   1,   1,   2,   2,   {2,   1,   4,   1},   {1,   8,   1,  32},   {2,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128,  32,   8,  32,   8,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  32, 256,  16,   4,  64,   2,   1,   2,   2,   {2,   1,   1,   1},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  32, 256,  16,   4,  64,   2,   1,   2,   2,   {2,   1,   1,   1},   {1,   8,   1,  32},   {2,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  32, 256,   8,   4,  64,   2,   1,   2,   2,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  32, 128,  16,   8,  32,   1,   1,   2,   2,   {2,   1,   1,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  32, 128,  16,   8,  32,   1,   1,   2,   2,   {2,   1,   1,   1},   {1,   8,   1,  32},   {2,   1,   4,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1, 128,  16,   8,  64,  16,   1,   1,   1,   1,   {1,   1,   8,   1},   {1,   8,   1,  16},   {1,   1,   1,   1},   {1,   8,   1,  16},   0},
        { "bwd", "fp32",   1,   0, 128,  16,   8,  64,  16,   1,   1,   1,   1,   {1,   1,   8,   1},   {1,   8,   1,  16},   {1,   1,   1,   1},   {1,   8,   1,  16},   0},
        { "bwd", "fp32",   1,   1,  64,  32,  16,  32,   8,   1,   2,   1,   1,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  64,  32,  16,  32,   8,   1,   2,   1,   1,   {2,   1,   2,   1},   {1,   8,   1,  32},   {2,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64,  32,   8,  32,   8,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  64,  16,  16,  64,   4,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   0,  64,  16,  16,  64,   4,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   1,  64,   8,  16,  64,   4,   1,   1,   1,   1,   {1,   1,   8,   1},   {1,  16,   1,   8},   {1,   1,   1,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   0,  64,   8,  16,  64,   4,   1,   1,   1,   1,   {1,   1,   8,   1},   {1,  16,   1,   8},   {1,   1,   1,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   1,  64,   4,  16,  64,   4,   1,   1,   1,   1,   {1,   1,  16,   1},   {1,  16,   1,   4},   {1,   1,   1,   1},   {1,  16,   1,   4},   0},
        { "bwd", "fp32",   4,   1,  64,   4,  16,  64,   4,   1,   1,   1,   1,   {1,   1,  16,   1},   {1,  16,   1,   4},   {1,   1,   1,   1},   {1,  16,   1,   4},   0},
        { "bwd", "fp32",   1,   0,  64,   4,  16,  64,   4,   1,   1,   1,   1,   {1,   1,  16,   1},   {1,  16,   1,   4},   {1,   1,   1,   1},   {1,  16,   1,   4},   0},
        { "bwd", "fp32",   4,   0,  64,   4,  16,  64,   4,   1,   1,   1,   1,   {1,   1,   4,   4},   {1,  16,   1,   4},   {1,   1,   1,   1},   {1,  16,   1,   4},   0},
        { "bwd", "fp32",   1,   1,  32,  64,  16,   8,  32,   2,   1,   1,   1,   {2,   1,   1,   1},   {1,   8,   1,  32},   {2,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  32,  64,   8,   8,  32,   2,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  32,  64,   8,   8,  32,   2,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  32,  32,   8,  16,  16,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   0,  32,  32,   8,  16,  16,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
        { "bwd", "fp32",   1,   1,  32,  16,  16,  32,   8,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,   8},   {1,   1,   2,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   0,  32,  16,  16,  32,   8,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,   8},   {1,   1,   2,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   1,  32,  16,   4,  32,   8,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   4,   1,  16},   {1,   1,   1,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",   4,   1,  32,  16,   4,  32,   8,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   4,   1,  16},   {1,   1,   1,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",   8,   1,  32,  16,   4,  32,   8,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   4,   1,  16},   {1,   1,   1,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",  16,   1,  32,  16,   4,  32,   8,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   4,   1,  16},   {1,   1,   1,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",   1,   1,  16, 256,  16,   4,  64,   1,   1,   2,   2,   {1,   1,   1,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   0,  16, 256,  16,   4,  64,   1,   1,   2,   2,   {1,   1,   1,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   1,  16, 128,   8,  16,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   8,   1},   {1,   8,   1,  16},   0},
        { "bwd", "fp32",   1,   0,  16, 128,   8,  16,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   8,   1},   {1,   8,   1,  16},   0},
        { "bwd", "fp32",   1,   1,  16,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   0,  16,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
        { "bwd", "fp32",   1,   1,  16,  32,  16,   8,  32,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   0,  16,  32,  16,   8,  32,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   1,  16,  32,   4,   8,  32,   2,   1,   1,   1,   {1,   1,   1,   1},   {1,   4,   1,  16},   {1,   1,   2,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",   4,   1,  16,  32,   4,   8,  32,   2,   1,   1,   1,   {1,   1,   1,   1},   {1,   4,   1,  16},   {1,   1,   2,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",   8,   1,  16,  32,   4,   8,  32,   2,   1,   1,   1,   {1,   1,   1,   1},   {1,   4,   1,  16},   {1,   1,   2,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",  16,   1,  16,  32,   4,   8,  32,   2,   1,   1,   1,   {1,   1,   1,   1},   {1,   4,   1,  16},   {1,   1,   2,   1},   {1,   4,   1,  16},   0},
        { "bwd", "fp32",   1,   1,   8,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   8},   {1,   1,   8,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   0,   8,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   8},   {1,   1,   8,   1},   {1,  16,   1,   8},   0},
        { "bwd", "fp32",   1,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   0},
        { "bwd", "fp32",   1,   0,   4,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   0},
    };
    // clang-format on
    return tunables;
}

/*
* Return with true if kernel is found, and its kernel_name, block_size, grid_size as well.
* Return with false if no kernel can be executed with input ConvolutionContext
*/
static bool FindImplicitGemmGtcDynamicBwdKernel(const ConvolutionContext& ctx,
                                                std::string& kernel_name,
                                                int& block_size,
                                                int& grid_size)
{
    auto tunables  = GetImplicitGemmGtcDynamicBwdTunablesList();
    auto pConfig   = tunables.begin();
    int hi         = ctx.out_height;
    int wi         = ctx.out_width;
    int n          = ctx.batch_sz;
    int k          = ctx.n_inputs;
    int c          = ctx.n_outputs;
    int ho         = ctx.in_height;
    int wo         = ctx.in_width;
    int stride_h   = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
    int stride_w   = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
    int dilation_h = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    int dilation_w = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    int pad_h      = ctx.pad_h;
    int pad_w      = ctx.pad_w;
    int y          = ctx.kernel_size_h;
    int x          = ctx.kernel_size_w;

    int gcd_stride_dilation_h = gcd(stride_h, dilation_h);
    int gcd_stride_dilation_w = gcd(stride_w, dilation_w);
    int y_tilda               = stride_h / gcd_stride_dilation_h;
    int x_tilda               = stride_w / gcd_stride_dilation_w;

    int h_tilda = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
    int w_tilda = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

    int y_dot = integer_divide_ceil(y, y_tilda);
    int x_dot = integer_divide_ceil(x, x_tilda);

    int h_tilda_left = std::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
    int w_tilda_left = std::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

    int h_tilda_right = std::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
    int w_tilda_right = std::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

    int h_tilda_slice = h_tilda_right - h_tilda_left;
    int w_tilda_slice = w_tilda_right - w_tilda_left;
    int num_of_gemm   = y_tilda * x_tilda;
    // clang-format on
    int gemm_m = c;
    int gemm_n = n * h_tilda_slice * w_tilda_slice;

    for(; pConfig != tunables.end(); pConfig++)
    {
        if(pConfig->nxe == 0)
        {
            if((x != 1) || (y != 1) || (stride_h != 1) || (stride_w != 1) || (dilation_h != 1) ||
               (dilation_w != 1) || (pad_h != 0) || (pad_w != 0))
            {
                continue;
            }
        }
        if((gemm_n % pConfig->gemm_n_per_block != 0) || (gemm_m % pConfig->gemm_m_per_block != 0))
        {
            continue;
        }
        if(pConfig->gemm_n_per_block % pConfig->nxb != 0)
        {
            continue;
        }
        //# ho * wo is 4x, gemm_n is 256, hence need batch size 256/4=64x
        if(n % (pConfig->gemm_n_per_block / pConfig->nxb) != 0)
        {
            continue;
        }
        if((h_tilda_slice * w_tilda_slice) % pConfig->nxb != 0)
        {
            continue;
        }
        bool gemm_k_valid = true;
        for(int gemm_id = 0; gemm_id < num_of_gemm; gemm_id++)
        {
            int i_y_tilda          = gemm_id / x_tilda;
            int i_x_tilda          = gemm_id % x_tilda;
            int y_dot_slice        = (i_y_tilda + 1) * y_dot <= y ? y_dot : y % y_dot;
            int x_dot_slice        = (i_x_tilda + 1) * x_dot <= x ? x_dot : x % x_dot;
            int gemm_k             = k * y_dot_slice * x_dot_slice;
            bool is_gemm_not_empty = gemm_k > 0;
            if(is_gemm_not_empty)
            {
                if(gemm_k % pConfig->gemm_k_per_block != 0)
                    gemm_k_valid = false;
            }
        }
        if(!gemm_k_valid)
            continue;
        // if all valid, break out;
        break;
    }
    if(pConfig != tunables.end())
    {
        kernel_name = pConfig->GetKernelName();

        block_size = pConfig->GetBlockSize();

        grid_size = integer_divide_ceil(gemm_m, pConfig->gemm_m_per_block) *
                    integer_divide_ceil(gemm_n, pConfig->gemm_n_per_block);
        return true;
    }
    return false;
}

bool ConvAsmImplicitGemmGTCDynamicBwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx908")))
        return false;

    if(!ctx.direction.IsBackwardData())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(ctx.group_counts != 1)
        return false;
    std::string kernel_name;
    int block_size;
    int grid_size;
    return FindImplicitGemmGtcDynamicBwdKernel(ctx, kernel_name, block_size, grid_size);
}

ConvSolution
ConvAsmImplicitGemmGTCDynamicBwdXdlops::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;
    KernelInfo kernel;
    std::ostringstream options;

    std::string kernel_name;
    int block_size;
    int grid_size;
    bool ret = FindImplicitGemmGtcDynamicBwdKernel(ctx, kernel_name, block_size, grid_size);
    if(!ret)
        MIOPEN_THROW("this kernel should not run with igemm dynamic!");

    kernel.kernel_file = "igemm_bwd_gtc_gfx908.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
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

    result.invoker_factory = conv::MakeImplGemmDynamicBackwardDataInvokerFactory(ctx);
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace solver
} // namespace miopen
