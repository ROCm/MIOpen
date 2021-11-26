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
#ifndef CONVOLUTION_PROBLEM_DESCRIPTOR
#define CONVOLUTION_PROBLEM_DESCRIPTOR
#include "../../../composable_kernel/include/utility/data_type_enum.hpp"

namespace ck {
namespace driver {

struct ConvolutionProblemDescriptor
{
    ConvolutionProblemDescriptor() = default;

    ConvolutionProblemDescriptor(int N_,
                                 int K_,
                                 int C_,
                                 int Y_,
                                 int X_,
                                 int Hi_,
                                 int Wi_,
                                 int Ho_,
                                 int Wo_,
                                 int ConvStrideH_,
                                 int ConvStrideW_,
                                 int ConvDilationH_,
                                 int ConvDilationW_,
                                 int InLeftPadH_,
                                 int InLeftPadW_,
                                 int InRightPadH_,
                                 int InRightPadW_,
                                 ck::DataTypeEnum_t InDataTypeEnum_,
                                 ck::DataTypeEnum_t WeiDataTypeEnum_,
                                 ck::DataTypeEnum_t OutDataTypeEnum_)
        : N{N_},
          K{K_},
          C{C_},
          Y{Y_},
          X{X_},
          Hi{Hi_},
          Wi{Wi_},
          Ho{Ho_},
          Wo{Wo_},
          ConvStrideH{ConvStrideH_},
          ConvStrideW{ConvStrideW_},
          ConvDilationH{ConvDilationH_},
          ConvDilationW{ConvDilationW_},
          InLeftPadH{InLeftPadH_},
          InLeftPadW{InLeftPadW_},
          InRightPadH{InRightPadH_},
          InRightPadW{InRightPadW_},
          InDataTypeEnum{InDataTypeEnum_},
          WeiDataTypeEnum{WeiDataTypeEnum_},
          OutDataTypeEnum{OutDataTypeEnum_}
    {
    }

    int N;
    int K;
    int C;
    int Y;
    int X;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int ConvStrideH;
    int ConvStrideW;
    int ConvDilationH;
    int ConvDilationW;
    int InLeftPadH;
    int InLeftPadW;
    int InRightPadH;
    int InRightPadW;

    ck::DataTypeEnum_t InDataTypeEnum;
    ck::DataTypeEnum_t WeiDataTypeEnum;
    ck::DataTypeEnum_t OutDataTypeEnum;

    std::size_t CalculateFlop() const { return 2L * N * K * C * Y * X * Ho * Wo; }
};

} // namespace driver
} // namespace ck
#endif
