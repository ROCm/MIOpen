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
#ifndef GUARD_MIOPEN_CONV_ALGO_NAME_HPP
#define GUARD_MIOPEN_CONV_ALGO_NAME_HPP

#include <string>
#include <unordered_map>

namespace miopen {

inline int FwdAlgoResolver(const std::string& s)
{
    static std::unordered_map<std::string, int> data{
        {"miopenConvolutionFwdAlgoGEMM", 0},
        {"miopenConvolutionFwdAlgoDirect", 1},
        {"miopenConvolutionFwdAlgoFFT", 2},
        {"miopenConvolutionFwdAlgoWinograd", 3},
    };
    return data.at(s);
}

inline int BwdDataAlgoResolver(const std::string& s)
{
    static std::unordered_map<std::string, int> data{
        {"miopenConvolutionBwdDataAlgoGEMM", 0},
        {"miopenConvolutionBwdDataAlgoDirect", 1},
        {"miopenConvolutionBwdDataAlgoFFT", 2},
        {"miopenConvolutionBwdDataAlgoWinograd", 3},
        {"miopenTransposeBwdDataAlgoGEMM", 4},
    };
    return data.at(s);
}

inline int BwdWeightsAlgoResolver(const std::string& s)
{
    static std::unordered_map<std::string, int> data{
        {"miopenConvolutionBwdWeightsAlgoGEMM", 0},
        {"miopenConvolutionBwdWeightsAlgoDirect", 1},
        {"miopenConvolutionBwdWeightsAlgoWinograd", 3},
    };
    return data.at(s);
}

} // namespace miopen

#endif // GUARD_MIOPEN_CONV_ALGO_NAME_HPP
