/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include <miopen/handle.hpp>
#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/conv/asm_implicit_gemm.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC)

#define BWD_MAX_GEMM_K_SPLITS 8
// #define DEBUG_IGEMM_ASM_BWD_NHWC_CHECK_VALID_TILE_LIST

namespace miopen {
namespace solver {

static const inline std::vector<PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC>&
GetBwdXdlopsNHWCConfigList()
{
    // clang-format off
    static const  std::vector<PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC> kernel_param_list {
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  64,  16, 32, 32,  2, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 256,  64,  16, 32, 32,  2, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  64,  16, 32, 32,  2, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 256,  64,  16, 32, 32,  2, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  64,   4, 32, 32,  2, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  4,  1, 64}, { 1, 1, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  64,   4, 32, 32,  2, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  4,  1, 64}, { 1, 1, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  32,  16, 32, 32,  2, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 256,  32,  16, 32, 32,  2, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  32,  16, 32, 32,  2, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 256,  32,  16, 32, 32,  2, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  32,   8, 32, 32,  2, 1, 1, 2, 1, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  32,   8, 32, 32,  2, 1, 1, 2, 1, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  32,   4, 64, 32,  1, 1, 1, 2, 1, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1,  4,  1, 32}, { 1, 1, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 256,  32,   4, 64, 32,  1, 1, 1, 2, 1, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1,  4,  1, 32}, { 1, 1, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128, 128,  16, 32, 32,  2, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128, 128,  16, 32, 32,  2, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128, 128,  16, 32, 32,  2, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128, 128,  16, 32, 32,  2, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128, 128,   8, 32, 32,  2, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 4, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128, 128,   8, 32, 32,  2, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 4, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128, 128,   4, 32, 32,  2, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 2, 1}, {  1,  4,  1, 64}, { 1, 1, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128, 128,   4, 32, 32,  2, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 2, 1}, {  1,  4,  1, 64}, { 1, 1, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,  32, 32, 32,  2, 1, 1, 1, 2, 1, 0, 0, 0, 1, { 1,16, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  64,  32, 32, 32,  2, 1, 1, 1, 2, 0, 0, 0, 0, 1, { 1,16, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,  32, 32, 32,  2, 1, 1, 1, 2, 1, 0, 1, 0, 1, { 1,16, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  64,  32, 32, 32,  2, 1, 1, 1, 2, 0, 0, 1, 0, 1, { 1,16, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,  16, 32, 32,  2, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  64,  16, 32, 32,  2, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,  16, 32, 32,  2, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  64,  16, 32, 32,  2, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,  16, 32, 32,  2, 1, 1, 1, 2, 1, 0, 0, 0, 1, { 1, 8, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  64,  16, 32, 32,  2, 1, 1, 1, 2, 0, 0, 0, 0, 1, { 1, 8, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,  16, 32, 32,  2, 1, 1, 1, 2, 1, 0, 1, 0, 1, { 1, 8, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  64,  16, 32, 32,  2, 1, 1, 1, 2, 0, 0, 1, 0, 1, { 1, 8, 1, 1}, {  1,  2,  4, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,   8, 32, 32,  2, 1, 1, 2, 1, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,   8, 32, 32,  2, 1, 1, 2, 1, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,   4, 64, 32,  1, 1, 1, 1, 1, 1, 0, 0, 1, 0, { 1, 1, 2, 1}, {  1,  4,  1, 64}, { 1, 1, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  64,   4, 64, 32,  1, 1, 1, 1, 1, 1, 0, 1, 1, 0, { 1, 1, 2, 1}, {  1,  4,  1, 64}, { 1, 1, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,  32, 32, 32,  2, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 8, 1}, {  1,  8,  1, 16}, { 1, 4, 2, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  32,  32, 32, 32,  2, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 8, 1}, {  1,  8,  1, 16}, { 1, 4, 2, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,  32, 32, 32,  2, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 8, 1}, {  1,  8,  1, 16}, { 1, 4, 2, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  32,  32, 32, 32,  2, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 8, 1}, {  1,  8,  1, 16}, { 1, 4, 2, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,  16, 32, 32,  2, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  32,  16, 32, 32,  2, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,  16, 32, 32,  2, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0, 128,  32,  16, 32, 32,  2, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,   8, 32, 32,  2, 1, 1, 1, 1, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,   8, 32, 32,  2, 1, 1, 1, 1, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,   4, 64, 32,  1, 1, 1, 1, 1, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  4,  1, 32}, { 1, 1, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1, 128,  32,   4, 64, 32,  1, 1, 1, 1, 1, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  4,  1, 32}, { 1, 1, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64, 256,  16, 32, 32,  2, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 4, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64, 256,  16, 32, 32,  2, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 4, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64, 256,  16, 32, 32,  2, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 4, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64, 256,  16, 32, 32,  2, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 4, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64, 128,  16, 32, 32,  2, 1, 1, 1, 2, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64, 128,  16, 32, 32,  2, 1, 1, 1, 2, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64, 128,  16, 32, 32,  2, 1, 1, 1, 2, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64, 128,  16, 32, 32,  2, 1, 1, 1, 2, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 2, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,  32, 16, 16,  4, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  64,  32, 16, 16,  4, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,  32, 16, 16,  4, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  64,  32, 16, 16,  4, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,   8, 16, 16,  1, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 2, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,   8, 16, 16,  1, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 2, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,   4, 16, 16,  1, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 1, 1}, {  1,  4,  1, 64}, { 1, 1, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  64,   4, 16, 16,  1, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 1, 1}, {  1,  4,  1, 64}, { 1, 1, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  32,  32, 16, 16,  4, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  32,  32, 16, 16,  4, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  32,  32, 16, 16,  4, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  32,  32, 16, 16,  4, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  32,  16, 16, 16,  4, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  32,  16, 16, 16,  4, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  32,  16, 16, 16,  4, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  32,  16, 16, 16,  4, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  16,  32, 16, 16,  4, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  8,  1, 16}, { 1, 4, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  16,  32, 16, 16,  4, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  8,  1, 16}, { 1, 4, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  16,  32, 16, 16,  4, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  8,  1, 16}, { 1, 4, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  16,  32, 16, 16,  4, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  8,  1, 16}, { 1, 4, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  16,  16, 16, 16,  4, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 2, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  16,  16, 16, 16,  4, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 2, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  64,  16,  16, 16, 16,  4, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 2, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  64,  16,  16, 16, 16,  4, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 2, 1, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  32,  64,  32, 16, 16,  4, 1, 1, 1, 2, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  32,  64,  32, 16, 16,  4, 1, 1, 1, 2, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  32,  64,  32, 16, 16,  4, 1, 1, 1, 2, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  32,  64,  32, 16, 16,  4, 1, 1, 1, 2, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  16,  64,  32, 16, 16,  4, 1, 1, 1, 2, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 16}, { 1, 4, 4, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  16,  64,  32, 16, 16,  4, 1, 1, 1, 2, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 16}, { 1, 4, 4, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 1,  16,  64,  32, 16, 16,  4, 1, 1, 1, 2, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 16}, { 1, 4, 4, 1}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenFloat,  0, 0,  16,  64,  32, 16, 16,  4, 1, 1, 1, 2, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 16}, { 1, 4, 4, 1}, {  1,  8,  1, 16}},

        {"bwd", "nhwc", miopenHalf,  0, 1, 256, 128,  32, 32, 32,  8, 2, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256, 128,  32, 32, 32,  8, 2, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256, 128,  32, 32, 32,  8, 2, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256, 128,  32, 32, 32,  8, 2, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256, 128,   8, 64, 32,  4, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 4, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256, 128,   8, 64, 32,  4, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 4, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  64,  32, 32, 32,  8, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  64,  32, 32, 32,  8, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  64,  32, 32, 32,  8, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  64,  32, 32, 32,  8, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  64,  16, 64, 32,  4, 1, 1, 1, 2, 1, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  64,  16, 64, 32,  4, 1, 1, 1, 2, 0, 0, 0, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  64,  16, 64, 32,  4, 1, 1, 1, 2, 1, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  64,  16, 64, 32,  4, 1, 1, 1, 2, 0, 0, 1, 0, 0, { 1, 4, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  64,   8, 64, 16,  4, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  64,   8, 64, 16,  4, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  32,  32, 64, 16,  4, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  32,  32, 64, 16,  4, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  32,  32, 64, 16,  4, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  32,  32, 64, 16,  4, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 8, 4, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  32,  16, 64, 16,  4, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  2,  1,128}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  32,  16, 64, 16,  4, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  2,  1,128}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  32,  16, 64, 16,  4, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  2,  1,128}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 256,  32,  16, 64, 16,  4, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  2,  1,128}, { 1, 2, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  32,   8, 64, 16,  4, 1, 1, 2, 1, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 256,  32,   8, 64, 16,  4, 1, 1, 2, 1, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 256,  32, 32, 32,  8, 1, 2, 2, 2, 1, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128, 256,  32, 32, 32,  8, 1, 2, 2, 2, 0, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 256,  32, 32, 32,  8, 1, 2, 2, 2, 1, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128, 256,  32, 32, 32,  8, 1, 2, 2, 2, 0, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 128,  32, 32, 32,  8, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128, 128,  32, 32, 32,  8, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 128,  32, 32, 32,  8, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128, 128,  32, 32, 32,  8, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 128,  16, 32, 32,  4, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1, 16,  1, 16}, { 1, 1, 8, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 128,  16, 32, 32,  4, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1, 16,  1, 16}, { 1, 1, 8, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 128,   8, 32, 32,  4, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 4, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128, 128,   8, 32, 32,  4, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 4, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  64,  32, 32, 32,  8, 1, 1, 1, 2, 1, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128,  64,  32, 32, 32,  8, 1, 1, 1, 2, 0, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  64,  32, 32, 32,  8, 1, 1, 1, 2, 1, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128,  64,  32, 32, 32,  8, 1, 1, 1, 2, 0, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  64,  16, 32, 32,  4, 1, 1, 2, 1, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1, 16,  1, 16}, { 1, 1, 4, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  64,  16, 32, 32,  4, 1, 1, 2, 1, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1, 16,  1, 16}, { 1, 1, 4, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  64,   8, 32, 32,  4, 1, 1, 2, 1, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  64,   8, 32, 32,  4, 1, 1, 2, 1, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  32,  32, 64, 16,  4, 1, 1, 1, 1, 1, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128,  32,  32, 64, 16,  4, 1, 1, 1, 1, 0, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  32,  32, 64, 16,  4, 1, 1, 1, 1, 1, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 0, 128,  32,  32, 64, 16,  4, 1, 1, 1, 1, 0, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 64}, { 1, 2, 1, 2}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  32,  16, 64, 16,  4, 1, 1, 1, 1, 1, 0, 0, 1, 0, { 1, 1, 8, 1}, {  1, 16,  1, 16}, { 1, 1, 2, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  32,  16, 64, 16,  4, 1, 1, 1, 1, 1, 0, 1, 1, 0, { 1, 1, 8, 1}, {  1, 16,  1, 16}, { 1, 1, 2, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  32,   8, 64, 16,  4, 1, 1, 1, 1, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1, 128,  32,   8, 64, 16,  4, 1, 1, 1, 1, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1,  8,  1, 32}, { 1, 1, 1, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64, 256,  32, 32, 32,  8, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64, 256,  32, 32, 32,  8, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64, 256,  32, 32, 32,  8, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64, 256,  32, 32, 32,  8, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 4}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64, 128,  32, 32, 32,  8, 1, 1, 2, 1, 1, 0, 0, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64, 128,  32, 32, 32,  8, 1, 1, 2, 1, 0, 0, 0, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64, 128,  32, 32, 32,  8, 1, 1, 2, 1, 1, 0, 1, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64, 128,  32, 32, 32,  8, 1, 1, 2, 1, 0, 0, 1, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 64}, { 1, 8, 1, 2}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,  64, 16, 16, 16, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  8,  1, 32}, { 1, 8, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  64,  64, 16, 16, 16, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  8,  1, 32}, { 1, 8, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,  64, 16, 16, 16, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  8,  1, 32}, { 1, 8, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  64,  64, 16, 16, 16, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  8,  1, 32}, { 1, 8, 1, 2}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  4,  1, 64}, { 1, 4, 1, 1}, {  1,  4,  1, 64}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 4, 1}, {  1, 16,  1, 16}, { 1, 1, 4, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,  16, 16, 16,  4, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 4, 1}, {  1, 16,  1, 16}, { 1, 1, 4, 1}, {  1, 16,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,   8, 16, 16,  4, 1, 1, 2, 2, 1, 0, 0, 1, 0, { 1, 1, 2, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  64,   8, 16, 16,  4, 1, 1, 2, 2, 1, 0, 1, 1, 0, { 1, 1, 2, 1}, {  1,  8,  1, 32}, { 1, 1, 2, 1}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  32,  32, 64, 16,  4, 1, 1, 1, 1, 1, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 2}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  32,  32, 64, 16,  4, 1, 1, 1, 1, 0, 0, 0, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 2}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  32,  32, 64, 16,  4, 1, 1, 1, 1, 1, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 2}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  32,  32, 64, 16,  4, 1, 1, 1, 1, 0, 0, 1, 0, 0, { 1, 8, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 2}, {  1,  8,  1, 16}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  32,  16, 64, 16,  4, 1, 1, 1, 1, 1, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  32,  16, 64, 16,  4, 1, 1, 1, 1, 0, 0, 0, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  64,  32,  16, 64, 16,  4, 1, 1, 1, 1, 1, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  64,  32,  16, 64, 16,  4, 1, 1, 1, 1, 0, 0, 1, 0, 0, { 1, 4, 2, 1}, {  1,  4,  1, 32}, { 1, 4, 1, 1}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  32, 128,  32, 16, 64,  4, 1, 1, 1, 1, 1, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 4}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  32, 128,  32, 16, 64,  4, 1, 1, 1, 1, 0, 0, 0, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 4}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  32, 128,  32, 16, 64,  4, 1, 1, 1, 1, 1, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 4}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  32, 128,  32, 16, 64,  4, 1, 1, 1, 1, 0, 0, 1, 0, 0, { 1, 4, 1, 1}, {  1,  8,  1, 32}, { 1, 4, 1, 4}, {  1,  8,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  32,  64,  32, 16, 64,  4, 1, 1, 1, 1, 1, 0, 0, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 32}, { 1, 8, 1, 2}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  32,  64,  32, 16, 64,  4, 1, 1, 1, 1, 0, 0, 0, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 32}, { 1, 8, 1, 2}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 1,  32,  64,  32, 16, 64,  4, 1, 1, 1, 1, 1, 0, 1, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 32}, { 1, 8, 1, 2}, {  1,  4,  1, 32}},
        {"bwd", "nhwc", miopenHalf,  0, 0,  32,  64,  32, 16, 64,  4, 1, 1, 1, 1, 0, 0, 1, 0, 0, { 1, 8, 1, 1}, {  1,  4,  1, 32}, { 1, 8, 1, 2}, {  1,  4,  1, 32}},
    };
    // clang-format on
    return kernel_param_list;
}

static std::tuple<std::string, // kernel_name
                  size_t,      // block_size
                  size_t>      // grid_size
GetImplicitGemmGtcDynamicBwdXdlopsNHWCKernel(
    const ConvolutionContext& ctx, const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC& config)
{
    const auto group = ctx.group_counts;
    const auto hi    = ctx.out_height;
    const auto wi    = ctx.out_width;
    const auto n     = ctx.batch_sz;
    // const auto k          = ctx.n_inputs;
    const auto c          = ctx.n_outputs;
    const auto ho         = ctx.in_height;
    const auto wo         = ctx.in_width;
    const auto stride_h   = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
    const auto stride_w   = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
    const auto dilation_h = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    const auto dilation_w = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    const auto pad_h      = ctx.pad_h;
    const auto pad_w      = ctx.pad_w;
    const auto y          = ctx.kernel_size_h;
    const auto x          = ctx.kernel_size_w;

    const auto gcd_stride_dilation_h = gcd(stride_h, dilation_h);
    const auto gcd_stride_dilation_w = gcd(stride_w, dilation_w);
    const auto y_tilda               = stride_h / gcd_stride_dilation_h;
    const auto x_tilda               = stride_w / gcd_stride_dilation_w;

    const auto h_tilda = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
    const auto w_tilda = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

    // const auto y_dot = integer_divide_ceil(y, y_tilda);
    // const auto x_dot = integer_divide_ceil(x, x_tilda);

    const auto h_tilda_left = std::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
    const auto w_tilda_left = std::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

    const auto h_tilda_right = std::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
    const auto w_tilda_right = std::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

    const auto h_tilda_slice = h_tilda_right - h_tilda_left;
    const auto w_tilda_slice = w_tilda_right - w_tilda_left;
    const auto num_of_gemm   = y_tilda * x_tilda;
    const auto gemm_m        = n * h_tilda_slice * w_tilda_slice;
    const auto gemm_n        = c / group;

    size_t block_size = config.BlockSize();
    size_t grid_size  = group * integer_divide_ceil(gemm_m, config.gemm_m_per_block) *
                       integer_divide_ceil(gemm_n, config.gemm_n_per_block) *
                       (1 << config.gemm_k_global_split);
    if(config.multihead != 0)
        grid_size *= num_of_gemm;
    std::string kernel_name = config.ToKernelName(ctx);
    return std::make_tuple(kernel_name, block_size, grid_size);
}

void PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC::HeuristicInit(const ConvolutionContext& ctx)
{
    static const std::vector<std::tuple<int, int, int>> tile_list_fp32 = {
        std::make_tuple(128, 128, 16),
        std::make_tuple(128, 64, 16),
        std::make_tuple(128, 64, 32),
        std::make_tuple(64, 128, 16),
        std::make_tuple(128, 32, 32),
        std::make_tuple(128, 32, 16),
        std::make_tuple(256, 64, 16),
        std::make_tuple(64, 256, 16),
        std::make_tuple(64, 64, 32),
        std::make_tuple(64, 32, 32),
        std::make_tuple(64, 32, 16),
        std::make_tuple(64, 16, 32),
        std::make_tuple(64, 16, 16),
        std::make_tuple(32, 64, 32),
        std::make_tuple(16, 64, 32),
    };

    static const std::vector<std::tuple<int, int, int>> tile_list_fp16 = {
        std::make_tuple(128, 128, 32),
        std::make_tuple(256, 128, 32),
        std::make_tuple(128, 256, 32),
        std::make_tuple(128, 64, 32),
        std::make_tuple(64, 128, 32),
        std::make_tuple(256, 64, 32),
        std::make_tuple(64, 256, 32),
        std::make_tuple(64, 64, 64),
        std::make_tuple(64, 64, 16),
        std::make_tuple(256, 32, 32),
        std::make_tuple(128, 32, 32),
        std::make_tuple(32, 128, 32),
        std::make_tuple(64, 32, 32),
        std::make_tuple(64, 32, 16),
        std::make_tuple(32, 64, 32),
    };

#ifdef DEBUG_IGEMM_ASM_BWD_NHWC_CHECK_VALID_TILE_LIST
    const auto& c_list = GetBwdXdlopsNHWCConfigList();
    for(const auto& tile : tile_list_fp16)
    {
        int mp, np, kp;
        std::tie(mp, np, kp) = tile;
        bool found           = false;
        for(const auto& config : c_list)
        {
            if(config.precision == miopenFloat)
                continue;
            if(config.gemm_m_per_block == mp && config.gemm_n_per_block == np &&
               config.gemm_k_per_block == kp &&
               !(config.tensor_a_thread_lengths[1] == 1 && config.tensor_b_thread_lengths[1] == 1))
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            MIOPEN_LOG_E("fp16 list  can't find " << mp << "x" << np << "x" << kp);
            MIOPEN_THROW(miopenStatusInternalError);
        }
    }
    for(const auto& tile : tile_list_fp32)
    {
        int mp, np, kp;
        std::tie(mp, np, kp) = tile;
        bool found           = false;
        for(const auto& config : c_list)
        {
            if(config.precision == miopenHalf)
                continue;
            if(config.gemm_m_per_block == mp && config.gemm_n_per_block == np &&
               config.gemm_k_per_block == kp &&
               !(config.tensor_a_thread_lengths[1] == 1 && config.tensor_b_thread_lengths[1] == 1))
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            MIOPEN_LOG_E("fp32 list  can't find " << mp << "x" << np << "x" << kp);
            MIOPEN_THROW(miopenStatusInternalError);
        }
    }
#endif

    const auto group      = ctx.group_counts;
    const auto hi         = ctx.out_height;
    const auto wi         = ctx.out_width;
    const auto n          = ctx.batch_sz;
    const auto k          = ctx.n_inputs;
    const auto c          = ctx.n_outputs;
    const auto ho         = ctx.in_height;
    const auto wo         = ctx.in_width;
    const auto stride_h   = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
    const auto stride_w   = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
    const auto dilation_h = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    const auto dilation_w = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    const auto pad_h      = ctx.pad_h;
    const auto pad_w      = ctx.pad_w;
    const auto y          = ctx.kernel_size_h;
    const auto x          = ctx.kernel_size_w;

    const auto gcd_stride_dilation_h = gcd(stride_h, dilation_h);
    const auto gcd_stride_dilation_w = gcd(stride_w, dilation_w);
    const auto y_tilda               = stride_h / gcd_stride_dilation_h;
    const auto x_tilda               = stride_w / gcd_stride_dilation_w;

    const auto h_tilda = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
    const auto w_tilda = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

    const auto h_tilda_left = std::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
    const auto w_tilda_left = std::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

    const auto h_tilda_right = std::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
    const auto w_tilda_right = std::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

    const auto h_tilda_slice = h_tilda_right - h_tilda_left;
    const auto w_tilda_slice = w_tilda_right - w_tilda_left;
    // const auto num_of_gemm   = y_tilda * x_tilda;
    const auto gemm_m = n * h_tilda_slice * w_tilda_slice;
    const auto gemm_n = c / group;
    const auto gemm_k_even =
        k / group; // this is not the gemm_k, but in most case we prefer k be evenly divided

    bool unit_conv = (x == 1) && (y == 1) && (stride_h == 1) && (stride_w == 1) &&
                     (dilation_h == 1) && (dilation_w == 1) && (pad_h == 0) && (pad_w == 0);
    bool not_support_vector_store = ctx.IsFp16() && ((c / group) % 2 != 0);
    int m_per_block, n_per_block, k_per_block;

    std::tie(m_per_block, n_per_block, k_per_block) = HeuristicInitMacroTileNoPadGemmK(
        gemm_m, gemm_n, gemm_k_even, ctx.IsFp32() ? tile_list_fp32 : tile_list_fp16);

    MIOPEN_LOG_I("m_per_block:" << m_per_block << ", n_per_block:" << n_per_block
                                << ", k_per_block:" << k_per_block);

    if((m_per_block == 0 && n_per_block == 0 && k_per_block == 0) || not_support_vector_store)
    {
        // not found, let's try  gemm_k pad now.
        const auto& config_list = GetBwdXdlopsNHWCConfigList();
        size_t min_pad_pixel    = std::numeric_limits<std::size_t>::max();
        size_t selected_index   = 0;
        for(size_t i = 0; i < config_list.size(); i++)
        {
            const auto& config = config_list[i];
            if(!((ctx.IsFp16() && config.precision == miopenHalf) ||
                 (ctx.IsFp32() && config.precision == miopenFloat)))
                continue;
            if(!(config.tensor_a_thread_lengths[1] == 1 && config.tensor_b_thread_lengths[1] == 1))
                continue;

            size_t cur_pad_pixel =
                ComputeMatrixPadSize(
                    gemm_m, config.gemm_m_per_block, gemm_k_even, config.gemm_k_per_block) +
                ComputeMatrixPadSize(
                    gemm_n, config.gemm_n_per_block, gemm_k_even, config.gemm_k_per_block) +
                ComputeMatrixPadSize(
                    gemm_m, config.gemm_m_per_block, gemm_n, config.gemm_n_per_block);
            if(cur_pad_pixel < min_pad_pixel)
            {
                min_pad_pixel  = cur_pad_pixel;
                selected_index = i;
            }
        }
        CopyParameters(config_list[selected_index]);
    }
    else
    {
        // found a suitable m/n/k, now let's prepare other parmater and initialize one
        const auto& config_list = GetBwdXdlopsNHWCConfigList();
        for(const auto& config : config_list)
        {
            if(!((ctx.IsFp16() && config.precision == miopenHalf) ||
                 (ctx.IsFp32() && config.precision == miopenFloat)))
                continue;

            if(m_per_block == config.gemm_m_per_block && n_per_block == config.gemm_n_per_block &&
               k_per_block == config.gemm_k_per_block)
            {
                bool need_k_split = false;
                if(ctx.IsFp16())
                {
                    // fp16 have extra limitation on c size, which dicide if need use need_k_split
                    // or not
                    if(c % 8 != 0 && c % 2 == 0)
                    {
                        need_k_split = true;
                    }
                }
                size_t current_grid_size;
                std::tie(std::ignore, std::ignore, current_grid_size) =
                    GetImplicitGemmGtcDynamicBwdXdlopsNHWCKernel(ctx, config);
                size_t gks = ComputeLog2GemmKGlobalSplitsWith2DMerge(current_grid_size,
                                                                     1200,
                                                                     k / group,
                                                                     1,
                                                                     config.gemm_k_per_block,
                                                                     BWD_MAX_GEMM_K_SPLITS);
                need_k_split |= gks != 0;
                MIOPEN_LOG_I("into current m_per_block:" << m_per_block
                                                         << ", n_per_block:" << n_per_block
                                                         << ", k_per_block:" << k_per_block);
                if((unit_conv && config.nxe == 0) || (!unit_conv && config.nxe != 0))
                {
                    CopyParameters(config);
                    if(need_k_split)
                        gemm_k_global_split = static_cast<int>(gks);
                    return;
                }
                else
                    continue;
            }
        }
        MIOPEN_LOG_E("can't find a suitable heuristic config");
        MIOPEN_THROW(miopenStatusInternalError);
    }
}
bool PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC::IsValidValue() const
{
    if(IsDefaultConstructed())
        return true;
    const auto& config_list = GetBwdXdlopsNHWCConfigList();
    if(index >= config_list.size())
        return false;
    return *this == config_list[index];
}
bool PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC::SetNextValue(
    const ConvolutionContext& /*config*/)
{
    if(use_spare_set)
    {
        const auto& config_list = GetBwdXdlopsNHWCConfigList();
        if(IsDefaultConstructed())
        {
            CopyParameters(config_list[index]);
        }
        else
        {
            if(gemm_k_global_split != 0)
            {
                if(NextLinear<1, BWD_MAX_GEMM_K_SPLITS>(gemm_k_global_split))
                    index++;
                else
                    return true;
            }
            else
            {
                index++;
            }
            if(index >= config_list.size())
                return false;
            CopyParameters(config_list[index]);
        }
        return true;
    }
    else
    {
        // always break generic search of main set (no spare), make sure we can use spare set
        return false;
    }
}
bool PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC::IsValid(const ConvolutionContext& ctx) const
{
    if(IsDefaultConstructed())
        return false;

    if(!((ctx.IsFp16() && precision == miopenHalf) || (ctx.IsFp32() && precision == miopenFloat)))
        return false;

    const auto group      = ctx.group_counts;
    const auto k          = ctx.n_inputs;
    const auto c          = ctx.n_outputs;
    const auto stride_h   = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
    const auto stride_w   = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
    const auto dilation_h = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    const auto dilation_w = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    const auto pad_h      = ctx.pad_h;
    const auto pad_w      = ctx.pad_w;
    const auto y          = ctx.kernel_size_h;
    const auto x          = ctx.kernel_size_w;

    bool unit_conv = (x == 1) && (y == 1) && (stride_h == 1) && (stride_w == 1) &&
                     (dilation_h == 1) && (dilation_w == 1) && (pad_h == 0) && (pad_w == 0);

    if(!(tensor_a_thread_lengths[1] == 1 && merge_e == 1))
    {
        // in case k split too large
        if(gemm_k_global_split != 0 && (gemm_k_per_block << gemm_k_global_split) > (k / group))
            return false;
        // gemm_k need be multiply of gemm_k_per_block
        if(((k >> gemm_k_global_split) / group) % gemm_k_per_block != 0)
            return false;
    }

    if(ctx.IsFp16() && !(tensor_a_thread_lengths[1] == 1 && tensor_b_thread_lengths[3] == 1 &&
                         merge_e == 1 && gemm_k_global_split == 0))
    {
        if(gemm_k_global_split != 0)
        {
            if((c / group) % 2 != 0)
                return false;
        }
        else
        {
            if((c / group) % gcd(gemm_n_per_block, vector_store == 0 ? 8 : vector_store) != 0)
                return false;
        }
    }

    if((nxe == 0) && !unit_conv)
    {
        return false;
    }

    // add more restriction for spare
    if(use_spare_set)
    {
        // non 1x1 kernel(except padding gemm_k) can't run 1x1 case
        if(unit_conv &&
           ((nxe != 0) && !(tensor_a_thread_lengths[1] == 1 && tensor_b_thread_lengths[1] == 1)))
            return false;
    }
    return true;
}

PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC::GetPerformanceConfig(
    const ConvolutionContext& params) const
{
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}
bool ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC::IsValidPerformanceConfig(
    const ConvolutionContext& problem,
    const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC& config) const
{
    return config.IsValidValue() && config.IsValid(problem);
}
PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC::Search(const ConvolutionContext& ctx,
                                                   const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

bool ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC::IsApplicable(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC{}))
        return false;

    const auto device_name = ctx.GetStream().GetDeviceName();
    if((device_name != "gfx908") && (device_name != "gfx90a"))
        return false;

    if(!ctx.use_asm_kernels)
        return false;

    if(!ctx.direction.IsBackwardData())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32() && !ctx.IsFp16())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(!ctx.IsLayoutNHWC())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false; // NOLINT (readability-simplify-boolean-expr)

    return true;
}
ConvSolution ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC& config,
    bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;
    KernelInfo kernel;
    std::ostringstream options;
    (void)disableConfigOverrideFromEnv;

    std::string kernel_name;
    size_t block_size;
    size_t grid_size;

    std::tie(kernel_name, block_size, grid_size) =
        GetImplicitGemmGtcDynamicBwdXdlopsNHWCKernel(ctx, config);

    kernel.kernel_file = kernel_name + ".s";
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

    MIOPEN_LOG_I2("ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC: " + config.ToString());

    result.invoker_factory =
        conv::MakeImplGemmDynamicBackwardDataXdlopsNHWCInvokerFactory(ctx, config);
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace solver
} // namespace miopen
