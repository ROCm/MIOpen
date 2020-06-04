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

#include <miopen/readonlyramdb.hpp>
#include <miopen/logger.hpp>

#include <fstream>
#include <mutex>
#include <sstream>
#include <map>

namespace miopen {

const std::unordered_map<std::string, std::string>& FindRamDb::find_db_init(std::string arch_cu)
{
// use the path to switch between arches
#include "find_db_init.h"
#if 0
    return {{"3-32-32-3x3-32-30-30-100-0x0-1x1-1x1-0-NCHW-FP32-F",
             "miopenConvolutionFwdAlgoDirect:ConvOclDirectFwd,0.07792,0,"
             "miopenConvolutionFwdAlgoDirect,<unused>;miopenConvolutionFwdAlgoGEMM:gemm,2.736,"
             "97200,rocBlas,<unused>;miopenConvolutionFwdAlgoWinograd:ConvBinWinogradRxSf2x3,0."
             "07104,0,miopenConvolutionFwdAlgoWinograd,<unused>"},
            {"32-30-30-3x3-3-32-32-100-0x0-1x1-1x1-0-NCHW-FP32-B",
             "miopenConvolutionBwdDataAlgoGEMM:gemm,3.904,97200,rocBlas,<unused>;"
             "miopenConvolutionBwdDataAlgoWinograd:ConvBinWinogradRxS,0.13632,0,"
             "miopenConvolutionBwdDataAlgoWinograd,<unused>;miopenConvolutionBwdDataAlgoDirect:"
             "ConvOclDirectFwd,0.12496,0,miopenConvolutionBwdDataAlgoDirect,<unused>"},
            {"32-30-30-3x3-3-32-32-100-0x0-1x1-1x1-0-NCHW-FP32-W",
             "miopenConvolutionBwdWeightsAlgoWinograd:ConvWinograd3x3MultipassWrW<3-6>,0.31824,"
             "22424576,miopenConvolutionBwdWeightsAlgoWinograd,"
             "32x30x30x3x3x3x32x32x100xNCHWxFP32x0x0x1x1x1x1x1x0;"
             "miopenConvolutionBwdWeightsAlgoGEMM:gemm,13.712,97200,rocBlas,<unused>;"
             "miopenConvolutionBwdWeightsAlgoDirect:ConvOclBwdWrW53,0.0856,345600,"
             "miopenConvolutionBwdWeightsAlgoDirect,<unused>"},
            {"key0", "value0"},
            {"key1", "value1"}};
#endif
}

} // namespace miopen
