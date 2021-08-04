/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>
#include <miopen/conv_algo_name.hpp>

#include <string>
#include <unordered_map>
namespace miopen {

miopenConvFwdAlgorithm_t StringToConvolutionFwdAlgo(const std::string& s)
{
    static const std::unordered_map<std::string, miopenConvFwdAlgorithm_t> data{
        {"miopenConvolutionFwdAlgoGEMM", miopenConvolutionFwdAlgoGEMM},
        {"miopenConvolutionFwdAlgoDirect", miopenConvolutionFwdAlgoDirect},
        {"miopenConvolutionFwdAlgoFFT", miopenConvolutionFwdAlgoFFT},
        {"miopenConvolutionFwdAlgoWinograd", miopenConvolutionFwdAlgoWinograd},
        {"miopenConvolutionFwdAlgoImplicitGEMM", miopenConvolutionFwdAlgoImplicitGEMM},
    };
    return data.at(s);
}

miopenConvBwdDataAlgorithm_t StringToConvolutionBwdDataAlgo(const std::string& s)
{
    static const std::unordered_map<std::string, miopenConvBwdDataAlgorithm_t> data{
        {"miopenConvolutionBwdDataAlgoGEMM", miopenConvolutionBwdDataAlgoGEMM},
        {"miopenConvolutionBwdDataAlgoDirect", miopenConvolutionBwdDataAlgoDirect},
        {"miopenConvolutionBwdDataAlgoFFT", miopenConvolutionBwdDataAlgoFFT},
        {"miopenConvolutionBwdDataAlgoWinograd", miopenConvolutionBwdDataAlgoWinograd},
        {"miopenTransposeBwdDataAlgoGEMM", miopenTransposeBwdDataAlgoGEMM},
        {"miopenConvolutionBwdDataAlgoImplicitGEMM", miopenConvolutionBwdDataAlgoImplicitGEMM},
    };
    return data.at(s);
}

miopenConvBwdWeightsAlgorithm_t StringToConvolutionBwdWeightsAlgo(const std::string& s)
{
    static const std::unordered_map<std::string, miopenConvBwdWeightsAlgorithm_t> data{
        {"miopenConvolutionBwdWeightsAlgoGEMM", miopenConvolutionBwdWeightsAlgoGEMM},
        {"miopenConvolutionBwdWeightsAlgoDirect", miopenConvolutionBwdWeightsAlgoDirect},
        {"miopenConvolutionBwdWeightsAlgoWinograd", miopenConvolutionBwdWeightsAlgoWinograd},
        {"miopenConvolutionBwdWeightsAlgoImplicitGEMM",
         miopenConvolutionBwdWeightsAlgoImplicitGEMM},
    };
    return data.at(s);
}

std::string ConvolutionAlgoToString(const miopenConvAlgorithm_t algo)
{
    switch(algo)
    {
    case miopenConvolutionAlgoGEMM: return "miopenConvolutionAlgoGEMM";
    case miopenConvolutionAlgoDirect: return "miopenConvolutionAlgoDirect";
    case miopenConvolutionAlgoFFT: return "miopenConvolutionAlgoFFT";
    case miopenConvolutionAlgoWinograd: return "miopenConvolutionAlgoWinograd";
    case miopenConvolutionAlgoImplicitGEMM: return "miopenConvolutionAlgoImplicitGEMM";
    }
    return "<invalid algorithm>";
}

std::string ConvolutionAlgoToDirectionalString(const miopenConvAlgorithm_t algo,
                                               conv::Direction dir)
{

    switch(dir)
    {
    case conv::Direction::Forward: {
        switch(algo)
        {
        case miopenConvolutionAlgoGEMM: return "miopenConvolutionFwdAlgoGEMM";
        case miopenConvolutionAlgoDirect: return "miopenConvolutionFwdAlgoDirect";
        case miopenConvolutionAlgoFFT: return "miopenConvolutionFwdAlgoFFT";
        case miopenConvolutionAlgoWinograd: return "miopenConvolutionFwdAlgoWinograd";
        case miopenConvolutionAlgoImplicitGEMM: return "miopenConvolutionFwdAlgoImplicitGEMM";
        }
        break;
    }
    case conv::Direction::BackwardData: {
        switch(algo)
        {
        case miopenConvolutionAlgoGEMM: return "miopenConvolutionBwdDataAlgoGEMM";
        case miopenConvolutionAlgoDirect: return "miopenConvolutionBwdDataAlgoDirect";
        case miopenConvolutionAlgoFFT: return "miopenConvolutionBwdDataAlgoFFT";
        case miopenConvolutionAlgoWinograd: return "miopenConvolutionBwdDataAlgoWinograd";
        case miopenConvolutionAlgoImplicitGEMM: return "miopenConvolutionBwdDataAlgoImplicitGEMM";
        }
        break;
    }
    case conv::Direction::BackwardWeights: {
        switch(algo)
        {
        case miopenConvolutionAlgoGEMM: return "miopenConvolutionBwdWeightsAlgoGEMM";
        case miopenConvolutionAlgoDirect: return "miopenConvolutionBwdWeightsAlgoDirect";
        case miopenConvolutionAlgoFFT: return "miopenConvolutionBwdWeightsAlgoFFT";
        case miopenConvolutionAlgoWinograd: return "miopenConvolutionBwdWeightsAlgoWinograd";
        case miopenConvolutionAlgoImplicitGEMM:
            return "miopenConvolutionBwdWeightsAlgoImplicitGEMM";
        }
        break;
    }
    }
    return "<invalid algorithm>";
}

// Interoperability of find-db (Find 1.0 API) and Immediate mode requires this:
static_assert(miopenConvolutionAlgoGEMM ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionFwdAlgoGEMM),
              "");
static_assert(miopenConvolutionAlgoGEMM ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdDataAlgoGEMM),
              "");
static_assert(miopenConvolutionAlgoGEMM ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdWeightsAlgoGEMM),
              "");

static_assert(miopenConvolutionAlgoDirect ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionFwdAlgoDirect),
              "");
static_assert(miopenConvolutionAlgoDirect ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdDataAlgoDirect),
              "");
static_assert(miopenConvolutionAlgoDirect ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdWeightsAlgoDirect),
              "");

static_assert(miopenConvolutionAlgoFFT ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionFwdAlgoFFT),
              "");
static_assert(miopenConvolutionAlgoFFT ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdDataAlgoFFT),
              "");

static_assert(miopenConvolutionAlgoWinograd ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionFwdAlgoWinograd),
              "");
static_assert(miopenConvolutionAlgoWinograd ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdDataAlgoWinograd),
              "");
static_assert(miopenConvolutionAlgoWinograd ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdWeightsAlgoWinograd),
              "");

static_assert(miopenConvolutionAlgoImplicitGEMM ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionFwdAlgoImplicitGEMM),
              "");
static_assert(miopenConvolutionAlgoImplicitGEMM ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdDataAlgoImplicitGEMM),
              "");
static_assert(miopenConvolutionAlgoImplicitGEMM ==
                  static_cast<miopenConvAlgorithm_t>(miopenConvolutionBwdWeightsAlgoImplicitGEMM),
              "");

} // namespace miopen
