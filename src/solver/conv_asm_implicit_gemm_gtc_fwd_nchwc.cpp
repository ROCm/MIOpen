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
#include <miopen/util_sol.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_DLOPS_NCHWC)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16)

// #define DEBUG_IGEMM_ASM_FWD_NCHWC_CHECK_VALID_TILE_LIST

namespace miopen {
namespace solver {

static const inline std::vector<PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC>&
GetFwdDlopsNCHWCConfigList()
{
    // clang-format off
    static const  std::vector<PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC> kernel_param_list {
        //generated dictionary
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 256, 128,  32,  8,  8, 4, 2, 2, 8, 4, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 256, 128,  32,  8,  8, 4, 2, 2, 8, 4, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 256, 128,  32,  8,  8, 4, 2, 2, 8, 4, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 256, 128,  32,  8,  8, 4, 2, 2, 8, 4, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 256,  32,  8,  8, 2, 4, 2, 8, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 8, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 256,  32,  8,  8, 2, 4, 2, 8, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 8, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 256,  32,  8,  8, 2, 4, 2, 8, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 8, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 256,  32,  8,  8, 2, 4, 2, 8, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 8, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 224, 128,  32,  8,  8, 2, 4, 7, 2, 4, { 1, 1, 1,28}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 224, 128,  32,  8,  8, 2, 4, 7, 2, 4, { 1, 1, 1,28}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 224, 128,  32,  8,  8, 2, 4, 7, 2, 4, { 1, 1, 1,28}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 224, 128,  32,  8,  8, 2, 4, 7, 2, 4, { 1, 1, 1,28}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 224,  32,  8,  8, 4, 2, 2, 7, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 7, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 224,  32,  8,  8, 4, 2, 2, 7, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 7, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 224,  32,  8,  8, 4, 2, 2, 7, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 7, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 224,  32,  8,  8, 4, 2, 2, 7, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 7, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 192, 128,  32,  8,  8, 2, 4, 3, 4, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 192, 128,  32,  8,  8, 2, 4, 3, 4, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 192, 128,  32,  8,  8, 2, 4, 3, 4, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 192, 128,  32,  8,  8, 2, 4, 3, 4, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 192,  32,  8,  8, 2, 4, 2, 6, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 192,  32,  8,  8, 2, 4, 2, 6, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 192,  32,  8,  8, 2, 4, 2, 6, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 192,  32,  8,  8, 2, 4, 2, 6, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 192,  64,  32,  8,  8, 2, 4, 3, 2, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 192,  64,  32,  8,  8, 2, 4, 3, 2, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 192,  64,  32,  8,  8, 2, 4, 3, 2, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 192,  64,  32,  8,  8, 2, 4, 3, 2, 4, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64, 192,  32,  8,  8, 2, 4, 1, 6, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64, 192,  32,  8,  8, 2, 4, 1, 6, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64, 192,  32,  8,  8, 2, 4, 1, 6, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64, 192,  32,  8,  8, 2, 4, 1, 6, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 6, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 160, 128,  32,  8,  8, 2, 4, 5, 2, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 160, 128,  32,  8,  8, 2, 4, 5, 2, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 160, 128,  32,  8,  8, 2, 4, 5, 2, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 160, 128,  32,  8,  8, 2, 4, 5, 2, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 160,  32,  8,  8, 4, 2, 2, 5, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 160,  32,  8,  8, 4, 2, 2, 5, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 160,  32,  8,  8, 4, 2, 2, 5, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 160,  32,  8,  8, 4, 2, 2, 5, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 160,  64,  32,  8,  8, 2, 4, 5, 1, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 160,  64,  32,  8,  8, 2, 4, 5, 1, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 160,  64,  32,  8,  8, 2, 4, 5, 1, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 160,  64,  32,  8,  8, 2, 4, 5, 1, 4, { 1, 1, 1,20}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64, 160,  32,  8,  8, 4, 2, 1, 5, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64, 160,  32,  8,  8, 4, 2, 1, 5, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64, 160,  32,  8,  8, 4, 2, 1, 5, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64, 160,  32,  8,  8, 4, 2, 1, 5, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 5, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 128,  32,  8,  8, 2, 4, 2, 4, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 128,  32,  8,  8, 2, 4, 2, 4, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 128,  32,  8,  8, 2, 4, 2, 4, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 128,  32,  8,  8, 2, 4, 2, 4, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128,  96,  32,  8,  8, 4, 2, 2, 3, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 3, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128,  96,  32,  8,  8, 4, 2, 2, 3, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 3, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128,  96,  32,  8,  8, 4, 2, 2, 3, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 3, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128,  96,  32,  8,  8, 4, 2, 2, 3, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 3, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  96, 128,  32,  8,  8, 4, 2, 3, 2, 4, { 1, 1, 1,12}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  96, 128,  32,  8,  8, 4, 2, 3, 2, 4, { 1, 1, 1,12}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  96, 128,  32,  8,  8, 4, 2, 3, 2, 4, { 1, 1, 1,12}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  96, 128,  32,  8,  8, 4, 2, 3, 2, 4, { 1, 1, 1,12}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  96,  96,  32,  8,  8, 2, 2, 3, 3, 4, { 1, 1, 1,24}, {  1,  8,  1, 16}, { 1, 1, 6, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  96,  96,  32,  8,  8, 2, 2, 3, 3, 4, { 1, 1, 1,24}, {  1,  8,  1, 16}, { 1, 1, 6, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  96,  96,  32,  8,  8, 2, 2, 3, 3, 4, { 1, 1, 1,24}, {  1,  8,  1, 16}, { 1, 1, 6, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  96,  96,  32,  8,  8, 2, 2, 3, 3, 4, { 1, 1, 1,24}, {  1,  8,  1, 16}, { 1, 1, 6, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128,  64,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128,  64,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128,  64,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128,  64,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 2, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64, 128,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64, 128,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64, 128,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64, 128,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 32}, { 1, 1, 4, 4}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128,  32,  32,  8,  8, 4, 1, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 2, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128,  32,  32,  8,  8, 4, 1, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 2, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128,  32,  32,  8,  8, 4, 1, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 2, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128,  32,  32,  8,  8, 4, 1, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 2, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  32, 128,  32,  8,  8, 1, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 16}, { 1, 1, 8, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  32, 128,  32,  8,  8, 1, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 16}, { 1, 1, 8, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  32, 128,  32,  8,  8, 1, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 16}, { 1, 1, 8, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  32, 128,  32,  8,  8, 1, 4, 2, 2, 4, { 1, 1, 1, 8}, {  1,  8,  1, 16}, { 1, 1, 8, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64,  64,  64,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1, 16,  1,  8}, { 1, 1, 8, 4}, {  1, 16,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64,  64,  64,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1, 16,  1,  8}, { 1, 1, 8, 4}, {  1, 16,  1,  8}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64,  64,  64,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1, 16,  1,  8}, { 1, 1, 8, 4}, {  1, 16,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64,  64,  64,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1, 16,  1,  8}, { 1, 1, 8, 4}, {  1, 16,  1,  8}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64,  64,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 16}, { 1, 1, 4, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64,  64,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 16}, { 1, 1, 4, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64,  64,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 16}, { 1, 1, 4, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64,  64,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1, 16}, { 1, 1, 4, 4}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64,  32,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  8}, { 1, 1, 4, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64,  32,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  8}, { 1, 1, 4, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64,  32,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  8}, { 1, 1, 4, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64,  32,  32,  8,  8, 4, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  8}, { 1, 1, 4, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  32,  64,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1,  8}, { 1, 1, 8, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  32,  64,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1,  8}, { 1, 1, 8, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  32,  64,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1,  8}, { 1, 1, 8, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  32,  64,  32,  8,  8, 2, 4, 2, 2, 4, { 1, 1, 1,16}, {  1,  8,  1,  8}, { 1, 1, 8, 4}, {  1,  8,  1,  8}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  32,  32,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  4}, { 1, 1, 8, 4}, {  1,  8,  1,  4}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  32,  32,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  4}, { 1, 1, 8, 4}, {  1,  8,  1,  4}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  32,  32,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  4}, { 1, 1, 8, 4}, {  1,  8,  1,  4}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  32,  32,  32,  8,  8, 2, 2, 2, 2, 4, { 1, 1, 1,32}, {  1,  8,  1,  4}, { 1, 1, 8, 4}, {  1,  8,  1,  4}},

        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 256, 128,  32,  8,  8, 4, 2, 2, 8, 8, { 1, 1, 1,32}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 256, 128,  32,  8,  8, 4, 2, 2, 8, 8, { 1, 1, 1,32}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 256, 128,  32,  8,  8, 4, 2, 2, 8, 8, { 1, 1, 1,32}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 256, 128,  32,  8,  8, 4, 2, 2, 8, 8, { 1, 1, 1,32}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 256,  32,  8,  8, 2, 4, 2, 8, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 4, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 256,  32,  8,  8, 2, 4, 2, 8, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 4, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 256,  32,  8,  8, 2, 4, 2, 8, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 4, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 256,  32,  8,  8, 2, 4, 2, 8, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 4, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 224, 128,  64,  8,  8, 2, 4, 7, 2, 8, { 1, 1, 1,56}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 224, 128,  64,  8,  8, 2, 4, 7, 2, 8, { 1, 1, 1,56}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 224, 128,  64,  8,  8, 2, 4, 7, 2, 8, { 1, 1, 1,56}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 224, 128,  64,  8,  8, 2, 4, 7, 2, 8, { 1, 1, 1,56}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 224,  64,  8,  8, 4, 2, 2, 7, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 7, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 224,  64,  8,  8, 4, 2, 2, 7, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 7, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 224,  64,  8,  8, 4, 2, 2, 7, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 7, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 224,  64,  8,  8, 4, 2, 2, 7, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 7, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 192, 128,  32,  8,  8, 2, 4, 3, 4, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 192, 128,  32,  8,  8, 2, 4, 3, 4, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 192, 128,  32,  8,  8, 2, 4, 3, 4, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 192, 128,  32,  8,  8, 2, 4, 3, 4, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 192,  32,  8,  8, 2, 4, 2, 6, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 192,  32,  8,  8, 2, 4, 2, 6, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 192,  32,  8,  8, 2, 4, 2, 6, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 192,  32,  8,  8, 2, 4, 2, 6, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 192,  64,  32,  8,  8, 2, 4, 3, 2, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 192,  64,  32,  8,  8, 2, 4, 3, 2, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 192,  64,  32,  8,  8, 2, 4, 3, 2, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 192,  64,  32,  8,  8, 2, 4, 3, 2, 8, { 1, 1, 1,24}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64, 192,  32,  8,  8, 2, 4, 1, 6, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64, 192,  32,  8,  8, 2, 4, 1, 6, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64, 192,  32,  8,  8, 2, 4, 1, 6, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64, 192,  32,  8,  8, 2, 4, 1, 6, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 3, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 160, 128,  64,  8,  8, 2, 4, 5, 2, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 160, 128,  64,  8,  8, 2, 4, 5, 2, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 160, 128,  64,  8,  8, 2, 4, 5, 2, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 160, 128,  64,  8,  8, 2, 4, 5, 2, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 160,  64,  8,  8, 4, 2, 2, 5, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 160,  64,  8,  8, 4, 2, 2, 5, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 160,  64,  8,  8, 4, 2, 2, 5, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 160,  64,  8,  8, 4, 2, 2, 5, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 160,  64,  64,  8,  8, 2, 4, 5, 1, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 2, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 160,  64,  64,  8,  8, 2, 4, 5, 1, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 2, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 160,  64,  64,  8,  8, 2, 4, 5, 1, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 2, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 160,  64,  64,  8,  8, 2, 4, 5, 1, 8, { 1, 1, 1,40}, {  1,  8,  1, 32}, { 1, 1, 2, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64, 160,  64,  8,  8, 4, 2, 1, 5, 8, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64, 160,  64,  8,  8, 4, 2, 1, 5, 8, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64, 160,  64,  8,  8, 4, 2, 1, 5, 8, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64, 160,  64,  8,  8, 4, 2, 1, 5, 8, { 1, 1, 1,16}, {  1,  8,  1, 32}, { 1, 1, 5, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128, 128,  32,  8,  8, 2, 4, 2, 4, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128, 128,  32,  8,  8, 2, 4, 2, 4, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128, 128,  32,  8,  8, 2, 4, 2, 4, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128, 128,  32,  8,  8, 2, 4, 2, 4, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128,  96,  64,  8,  8, 4, 2, 2, 3, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 3, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128,  96,  64,  8,  8, 4, 2, 2, 3, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 3, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128,  96,  64,  8,  8, 4, 2, 2, 3, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 3, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128,  96,  64,  8,  8, 4, 2, 2, 3, 8, { 1, 1, 1,32}, {  1,  8,  1, 32}, { 1, 1, 3, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  96, 128,  64,  8,  8, 4, 2, 3, 2, 8, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  96, 128,  64,  8,  8, 4, 2, 3, 2, 8, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  96, 128,  64,  8,  8, 4, 2, 3, 2, 8, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  96, 128,  64,  8,  8, 4, 2, 3, 2, 8, { 1, 1, 1,24}, {  1,  8,  1, 32}, { 1, 1, 4, 8}, {  1,  8,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  96,  96,  32,  8,  8, 2, 2, 3, 3, 8, { 1, 1, 1,24}, {  1,  4,  1, 32}, { 1, 1, 3, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  96,  96,  32,  8,  8, 2, 2, 3, 3, 8, { 1, 1, 1,24}, {  1,  4,  1, 32}, { 1, 1, 3, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  96,  96,  32,  8,  8, 2, 2, 3, 3, 8, { 1, 1, 1,24}, {  1,  4,  1, 32}, { 1, 1, 3, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  96,  96,  32,  8,  8, 2, 2, 3, 3, 8, { 1, 1, 1,24}, {  1,  4,  1, 32}, { 1, 1, 3, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128,  64,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128,  64,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128,  64,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128,  64,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 64}, { 1, 1, 1, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64, 128,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64, 128,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64, 128,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64, 128,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 64}, { 1, 1, 2, 8}, {  1,  4,  1, 64}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1, 128,  32,  32,  8,  8, 4, 1, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 32}, { 1, 1, 1, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1, 128,  32,  32,  8,  8, 4, 1, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 32}, { 1, 1, 1, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0, 128,  32,  32,  8,  8, 4, 1, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 32}, { 1, 1, 1, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0, 128,  32,  32,  8,  8, 4, 1, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 32}, { 1, 1, 1, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  32, 128,  32,  8,  8, 1, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 32}, { 1, 1, 4, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  32, 128,  32,  8,  8, 1, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 32}, { 1, 1, 4, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  32, 128,  32,  8,  8, 1, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 32}, { 1, 1, 4, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  32, 128,  32,  8,  8, 1, 4, 2, 2, 8, { 1, 1, 1, 8}, {  1,  4,  1, 32}, { 1, 1, 4, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64,  64,  64,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 4, 8}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64,  64,  64,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 4, 8}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64,  64,  64,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 4, 8}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64,  64,  64,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  8,  1, 16}, { 1, 1, 4, 8}, {  1,  8,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64,  64,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 32}, { 1, 1, 2, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64,  64,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 32}, { 1, 1, 2, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64,  64,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 32}, { 1, 1, 2, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64,  64,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 32}, { 1, 1, 2, 8}, {  1,  4,  1, 32}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  64,  32,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 16}, { 1, 1, 2, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  64,  32,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 16}, { 1, 1, 2, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  64,  32,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 16}, { 1, 1, 2, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  64,  32,  32,  8,  8, 4, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1, 16}, { 1, 1, 2, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  32,  64,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 16}, { 1, 1, 4, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  32,  64,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 16}, { 1, 1, 4, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  32,  64,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 16}, { 1, 1, 4, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  32,  64,  32,  8,  8, 2, 4, 2, 2, 8, { 1, 1, 1,16}, {  1,  4,  1, 16}, { 1, 1, 4, 8}, {  1,  4,  1, 16}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 1,  32,  32,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1,  8}, { 1, 1, 4, 8}, {  1,  4,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 1,  32,  32,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1,  8}, { 1, 1, 4, 8}, {  1,  4,  1,  8}},
        {"fwd", "nchwc_cyxkc", miopenHalf,  0, 0,  32,  32,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1,  8}, { 1, 1, 4, 8}, {  1,  4,  1,  8}},
        {"fwd", "nchwc_kcyxc", miopenHalf,  0, 0,  32,  32,  32,  8,  8, 2, 2, 2, 2, 8, { 1, 1, 1,32}, {  1,  4,  1,  8}, { 1, 1, 4, 8}, {  1,  4,  1,  8}},
    };
    // clang-format on
    return kernel_param_list;
}

static std::tuple<std::string, // kernel_name
                  size_t,      // block_size
                  size_t,      // grid_size
                  size_t>      // splits_4G
GetImplicitGemmGtcDynamicFwdDlopsNCHWCKernel(
    const ConvolutionContext& ctx, const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC& config)
{
    const auto& n     = ctx.batch_sz;
    const auto& k     = ctx.n_outputs * config.vector_c;
    const auto& ho    = ctx.out_height;
    const auto& wo    = ctx.out_width;
    const auto& group = ctx.group_counts;

    const auto& hi = ctx.in_height;
    const auto& wi = ctx.in_width;
    const auto& c  = ctx.n_inputs;

    auto splits_4G =
        igemm_split_batch_size(hi, wi, ho, wo, n, k, c, miopen::GetTypeSize(ctx.in_data_type));

    const auto gemm_m = k / group;
    const auto gemm_n = (n / splits_4G) * ho * wo;

    // printf("(gemm_m, gemm_n) = (%d, %d)", gemm_m, gemm_n);
    // printf("(config.gemm_m_per_block, config.gemm_n_per_block) = (%d, %d)",
    // config.gemm_m_per_block, config.gemm_n_per_block);

    size_t block_size = config.BlockSize();
    size_t grid_size  = group * integer_divide_ceil(gemm_m, config.gemm_m_per_block) *
                       integer_divide_ceil(gemm_n, config.gemm_n_per_block);
    std::string kernel_name = config.ToKernelName(ctx);
    return std::make_tuple(kernel_name, block_size, grid_size, splits_4G);
}

void PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC::HeuristicInit(const ConvolutionContext& ctx)
{

    static const std::vector<std::tuple<int, int, int>> tile_list_Halfx4 = {
        std::make_tuple(256, 128, 32), std::make_tuple(128, 256, 32), std::make_tuple(224, 128, 32),
        std::make_tuple(128, 224, 32), std::make_tuple(192, 128, 32), std::make_tuple(128, 192, 32),
        std::make_tuple(192, 64, 32),  std::make_tuple(64, 192, 32),  std::make_tuple(160, 128, 32),
        std::make_tuple(128, 160, 32), std::make_tuple(160, 64, 32),  std::make_tuple(64, 160, 32),
        std::make_tuple(128, 128, 32), std::make_tuple(128, 96, 32),  std::make_tuple(96, 128, 32),
        std::make_tuple(96, 96, 32),   std::make_tuple(128, 64, 32),  std::make_tuple(64, 128, 32),
        std::make_tuple(64, 64, 64),   std::make_tuple(64, 64, 32),   std::make_tuple(128, 32, 32),
        std::make_tuple(32, 128, 32),  std::make_tuple(64, 32, 32),   std::make_tuple(32, 64, 32),
        std::make_tuple(32, 32, 32),
    };

    static const std::vector<std::tuple<int, int, int>> tile_list_Halfx8 = {
        std::make_tuple(256, 128, 32), std::make_tuple(128, 256, 32), std::make_tuple(224, 128, 64),
        std::make_tuple(128, 224, 64), std::make_tuple(192, 128, 32), std::make_tuple(128, 192, 32),
        std::make_tuple(192, 64, 32),  std::make_tuple(64, 192, 32),  std::make_tuple(160, 128, 64),
        std::make_tuple(128, 160, 64), std::make_tuple(160, 64, 64),  std::make_tuple(64, 160, 64),
        std::make_tuple(128, 128, 32), std::make_tuple(128, 96, 64),  std::make_tuple(96, 128, 64),
        std::make_tuple(96, 96, 32),   std::make_tuple(128, 64, 32),  std::make_tuple(64, 128, 32),
        std::make_tuple(64, 64, 64),   std::make_tuple(64, 64, 32),   std::make_tuple(128, 32, 32),
        std::make_tuple(32, 128, 32),  std::make_tuple(64, 32, 32),   std::make_tuple(32, 64, 32),
        std::make_tuple(32, 32, 32),
    };

#ifdef DEBUG_IGEMM_ASM_FWD_NCHWC_CHECK_VALID_TILE_LIST
    const auto& c_list = GetFwdDlopsNCHWCConfigList();
    for(const auto& tile : tile_list_Halfx4)
    {
        int mp, np, kp;
        std::tie(mp, np, kp) = tile;
        bool found           = false;
        for(const auto& config : c_list)
        {
            if(config.precision == "Half" && config.vector_c == 8)
                continue;
            if(config.gemm_m_per_block == mp && config.gemm_n_per_block == np &&
               config.gemm_k_per_block == kp &&
               !(config.tensor_a_thread_lengths[1] == 1 && config.tensor_b_thread_lengths[1] == 1))
            {
                // pad c configs can't be used in tile list
                found = true;
                break;
            }
        }
        if(!found)
        {
            MIOPEN_LOG_E("Halfx4 list can't find " << mp << "x" << np << "x" << kp);
            MIOPEN_THROW(miopenStatusInternalError);
        }
    }

    for(const auto& tile : tile_list_Halfx8)
    {
        int mp, np, kp;
        std::tie(mp, np, kp) = tile;
        bool found           = false;
        for(const auto& config : c_list)
        {
            if(config.precision == "Half" && config.vector_c == 4)
                continue;
            if(config.gemm_m_per_block == mp && config.gemm_n_per_block == np &&
               config.gemm_k_per_block == kp &&
               !(config.tensor_a_thread_lengths[1] == 1 && config.tensor_b_thread_lengths[1] == 1))
            {
                // pad c configs can't be used in tile list
                found = true;
                break;
            }
        }
        if(!found)
        {
            MIOPEN_LOG_E("Halfx8 list can't find " << mp << "x" << np << "x" << kp);
            MIOPEN_THROW(miopenStatusInternalError);
        }
    }
#endif

    const auto& n     = ctx.batch_sz;
    const auto& c     = ctx.n_inputs;
    const auto& k     = ctx.n_outputs;
    const auto& ho    = ctx.out_height;
    const auto& wo    = ctx.out_width;
    const auto& y     = ctx.kernel_size_h;
    const auto& x     = ctx.kernel_size_w;
    const auto& group = ctx.group_counts;

    size_t gemm_m = n * ho * wo;
    size_t gemm_n = k / group;
    size_t gemm_k = (c / group) * y * x;

    int m_per_block, n_per_block, k_per_block;

    std::tie(m_per_block, n_per_block, k_per_block) = HeuristicInitMacroTileNoPadGemmK(
        gemm_m,
        gemm_n,
        gemm_k,
        (ctx.IsFp16() && ctx.vectorLength == 4) ? tile_list_Halfx4 : tile_list_Halfx8);

    auto find_with_gemm_k_pad = [&]() {
        const auto& config_list = GetFwdDlopsNCHWCConfigList();
        size_t min_pad_pixel    = std::numeric_limits<std::size_t>::max();
        size_t selected_index   = 0;
        for(size_t i = 0; i < config_list.size(); i++)
        {
            const auto& config = config_list[i];
            if(!(((ctx.IsFp16() && ctx.vectorLength == 4) && config.precision == "Halfx4") ||
                 ((ctx.IsFp16() && ctx.vectorLength == 8) && config.precision == "Halfx8")))
                continue;

            if(!((ctx.IsNCHWc_NCHWc() && config.tensor_layout == "nchwc_kcyxc") ||
                 (ctx.IsNCHWc_CHWNc() && config.tensor_layout == "nchwc_cyxkc")))
                continue;

            if(!(config.tensor_a_thread_lengths[1] == 1 && config.tensor_b_thread_lengths[1] == 1))
                continue;
            // If we go here, then this is our last hope.
            // This kind of kernel support any configs
            size_t cur_pad_pixel =
                ComputeMatrixPadSize(
                    gemm_m, config.gemm_m_per_block, gemm_k, config.gemm_k_per_block) +
                ComputeMatrixPadSize(
                    gemm_n, config.gemm_n_per_block, gemm_k, config.gemm_k_per_block) +
                ComputeMatrixPadSize(
                    gemm_m, config.gemm_m_per_block, gemm_n, config.gemm_n_per_block);
            if(cur_pad_pixel < min_pad_pixel)
            {
                min_pad_pixel  = cur_pad_pixel;
                selected_index = i;
            }
        }
        CopyParameters(config_list[selected_index]);
    };

    find_with_gemm_k_pad();
}

bool PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC::SetNextValue(
    const ConvolutionContext& /*config*/)
{
    if(use_spare_set)
    {
        const auto& config_list = GetFwdDlopsNCHWCConfigList();
        if(IsDefaultConstructed())
        {
            CopyParameters(config_list[index]);
        }
        else
        {
            index++;
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
bool PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC::IsValidValue() const
{
    if(IsDefaultConstructed())
        return true;
    const auto& config_list = GetFwdDlopsNCHWCConfigList();
    if(index >= config_list.size())
        return false;
    return *this == config_list[index];
}
bool PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC::IsValid(const ConvolutionContext& ctx) const
{
    if(IsDefaultConstructed())
        return false;

    if(!(((ctx.IsFp16() && ctx.vectorLength == 4) && precision == "Halfx4") ||
         ((ctx.IsFp16() && ctx.vectorLength == 8) && precision == "Halfx8")))
        return false;

    const auto& c         = ctx.n_inputs;
    const auto& k         = ctx.n_outputs;
    const auto& group     = ctx.group_counts;
    const auto stride_h   = ctx.out_height > 1 ? ctx.kernel_stride_h : 1;
    const auto stride_w   = ctx.out_width > 1 ? ctx.kernel_stride_w : 1;
    const auto dilation_h = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    const auto dilation_w = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    const auto& pad_h     = ctx.pad_h;
    const auto& pad_w     = ctx.pad_w;
    const auto& y         = ctx.kernel_size_h;
    const auto& x         = ctx.kernel_size_w;

    bool unit_conv = (x == 1) && (y == 1) && (stride_h == 1) && (stride_w == 1) &&
                     (dilation_h == 1) && (dilation_w == 1) && (pad_h == 0) && (pad_w == 0);

    // TODO : check logic here
    // tensor thread length[1]==1 means per thread read a dword
    // c, k should be integer multiples of vector_c
    if((c / group) % vector_c != 0 || (k / group) % vector_c != 0)
    {
        return false;
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

PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC::GetDefaultPerformanceConfig(
    const ConvolutionContext& params) const
{
    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC pp;
    pp.HeuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC::IsValidPerformanceConfig(
    const ConvolutionContext& problem,
    const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC& config) const
{
    return config.IsValidValue() && config.IsValid(problem);
}
PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC::Search(const ConvolutionContext& ctx,
                                                   const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

bool ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC::IsApplicable(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_DLOPS_NCHWC{}))
        return false;

    const auto device_name = ctx.GetStream().GetDeviceName();
    if((device_name != "gfx1030"))
        return false;

    if(!ctx.use_asm_kernels)
        return false;

    if(!ctx.direction.IsForward())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsLayoutNCHWC())
        return false;

    if(!(ctx.IsFp16() && ctx.vectorLength == 4) && !(ctx.IsFp16() && ctx.vectorLength == 8))
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false; // NOLINT (readability-simplify-boolean-expr)

    if(0 == igemm_split_batch_size(ctx.in_height,
                                   ctx.in_width,
                                   ctx.out_height,
                                   ctx.out_width,
                                   ctx.batch_sz,
                                   ctx.n_outputs,
                                   ctx.n_inputs,
                                   miopen::GetTypeSize(ctx.in_data_type)))
        return false;

    return true;
}

ConvSolution ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC& config) const
{
    ConvSolution result;
    KernelInfo kernel;

    std::string kernel_name;
    size_t block_size;
    size_t grid_size;

    int splits_4G;

    std::tie(kernel_name, block_size, grid_size, splits_4G) =
        GetImplicitGemmGtcDynamicFwdDlopsNCHWCKernel(ctx, config);
    kernel.kernel_file = kernel_name + ".s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(splits_4G);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    if(!ctx.IsLayoutNCHWC())
        MIOPEN_THROW("Tensor layout is not in vectorized");

    result.construction_params.push_back(kernel);
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    std::ostringstream opts_0(options.str(), std::ios_base::ate);
    result.construction_params[0].comp_options = opts_0.str();

    std::ostringstream msg;
    MIOPEN_LOG_I2(SolverDbId() << ": " << config.ToString() << msg.str());

    result.invoker_factory = conv::MakeImplGemmDynamicForwardDlopsNCHWCInvokerFactory(ctx, config);
    return result;
}

} // namespace solver
} // namespace miopen
