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

#include <miopen/stringutils.hpp>
#include <miopen/env.hpp>

#include "mdgraph_common.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_KERNELS)

int main()
{
    std::string pgm_name;
    std::string krn_name;
    std::string alg_name;

    auto&& h       = get_handle();
    auto this_arch = h.GetDeviceName();
    auto target    = h.GetTargetProperties();

    auto wino_supported_arch = {"gfx90a", "gfx908", "gfx906", "gfx900", "gfx803"};

    auto is_wino_support = !miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}) &&
                           !miopen::IsDisabled(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD{}) &&
                           std::any_of(wino_supported_arch.begin(),
                                       wino_supported_arch.end(),
                                       [&](std::string arch) { return arch == this_arch; });

    if(is_wino_support)
    {
        const auto name = get_handle().GetDeviceName();
        if(miopen::StartsWith(name, "gfx8"))
        {
            // Winograd because c, x and y satisfy criteria
            ConvAlgTest(
                {100, 32, 8, 8}, {64, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
            EXPECT(krn_name == "miopenSp3AsmConvRxSU_CBA");
            EXPECT(alg_name == "miopenConvolutionWinogradBiasActiv");

            // c is odd so winograd not supported and padding is zero
            ConvAlgTest(
                {100, 31, 8, 8}, {64, 31, 3, 3}, {0, 0, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
            EXPECT(krn_name != "miopenSp3AsmConvRxSU_CBA");
            EXPECT(alg_name != "miopenConvolutionWinogradBiasActiv");

            // c is less than 18 so winograd not supported and padding is zero
            ConvAlgTest(
                {100, 15, 8, 8}, {64, 15, 3, 3}, {0, 0, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
            EXPECT(krn_name != "miopenSp3AsmConvRxSU_CBA");
            EXPECT(alg_name != "miopenConvolutionWinogradBiasActiv");
        }
        else if(miopen::StartsWith(name, "gfx9") || miopen::StartsWith(name, "gfx10"))
        {
            std::string krn_name_ref = "miopenSp3AsmConv_v21_1_2_";
            if(miopen::StartsWith(name, "gfx9"))
            {
                krn_name_ref += "gfx9";
            }
            else // StartsWith(name, "gfx10")
            {
                krn_name_ref += "gfx10";
            }
            krn_name_ref += "_fp32_stride1";

            // Winograd because c, x and y satisfy criteria
            ConvAlgTest(
                {100, 32, 8, 8}, {64, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
            EXPECT(krn_name == krn_name_ref);
            EXPECT(alg_name == "miopenConvolutionWinogradBiasActiv");

            // Winograd because c, x and y satisfy criteria
            ConvAlgTest(
                {100, 31, 8, 8}, {64, 31, 3, 3}, {0, 0, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
            EXPECT(krn_name == krn_name_ref);
            EXPECT(alg_name == "miopenConvolutionWinogradBiasActiv");

            // Winograd because c, x and y satisfy criteria
            ConvAlgTest(
                {100, 15, 8, 8}, {64, 15, 3, 3}, {0, 0, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
            EXPECT(krn_name == krn_name_ref);
            EXPECT(alg_name == "miopenConvolutionWinogradBiasActiv");
        }
    }

    auto asm_supported_arch = {"gfx90a", "gfx908", "gfx906", "gfx900", "gfx803"};

    auto is_asm_support = !miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}) &&
                          std::any_of(asm_supported_arch.begin(),
                                      asm_supported_arch.end(),
                                      [&](std::string arch) { return arch == this_arch; });

    // the asm kernel is the fastest for 1x1 and padding
    if(is_asm_support)
    {
        ConvAlgTest(
            {100, 32, 8, 8}, {64, 32, 1, 1}, {0, 0, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
        EXPECT(pgm_name == "conv1x1u_bias_activ.s");
        EXPECT(krn_name == "miopenGcnAsmConv1x1U");
        EXPECT(alg_name == "miopenConvolutionDirectBiasActivAsm");

        ConvBiasAlgTest(
            {100, 32, 8, 8}, {64, 32, 1, 1}, {0, 0, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
        EXPECT(pgm_name == "conv1x1u_bias_activ.s");
        EXPECT(krn_name == "miopenGcnAsmConv1x1U");
        EXPECT(alg_name == "miopenConvolutionDirectBiasActivAsm");
    }
}
