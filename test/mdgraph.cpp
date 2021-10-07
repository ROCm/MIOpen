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

#include <miopen/stringutils.hpp>
#include <miopen/env.hpp>

#include "mdgraph_common.hpp"

int main()
{
    std::string pgm_name;
    std::string krn_name;
    std::string alg_name;

    // opencl kernels supports odd sizes with padding <= 2
    for(auto idx : {5, 7, 9, 11})
    {
        ConvAlgTest(
            {100, 32, 8, 8}, {64, 32, idx, idx}, {0, 0, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
        EXPECT(pgm_name == "MIOpenConvDirBatchNormActiv.cl");
        EXPECT(krn_name == "MIOpenConvUniBatchNormActiv");
        EXPECT(alg_name == "miopenConvolutionDirectBiasActiv");

        ConvAlgTest(
            {100, 32, 8, 8}, {64, 32, idx, idx}, {2, 2, 1, 1, 1, 1}, pgm_name, krn_name, alg_name);
        EXPECT(pgm_name == "MIOpenConvDirBatchNormActiv.cl");
        EXPECT(krn_name == "MIOpenConvUniBatchNormActiv");
        EXPECT(alg_name == "miopenConvolutionDirectBiasActiv");
    }

    BNAlgTest({100, 32, 8, 8}, miopenBNSpatial, pgm_name, krn_name, alg_name);
    EXPECT(pgm_name == "MIOpenBatchNormActivInfer.cl");
    EXPECT(krn_name == "MIOpenBatchNormActivInferSpatialEst");
    EXPECT(alg_name == "MIOpenBatchNormActivInferSpatialEst");

    BNAlgTest({100, 32, 8, 8}, miopenBNPerActivation, pgm_name, krn_name, alg_name);
    EXPECT(pgm_name == "MIOpenBatchNormActivInfer.cl");
    EXPECT(krn_name == "MIOpenBatchNormActivInferPerActEst");
    EXPECT(alg_name == "MIOpenBatchNormActivInferPerActEst");
}
