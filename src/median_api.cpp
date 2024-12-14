/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <cstdint>
#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/median.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

inline void LogCmdMedian(const miopenTensorDescriptor_t inputDesc,
                         const miopenTensorDescriptor_t outputDesc,
                         const uint64_t dim,
                         const bool keepdim,
                         bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();

        if(dtype == miopenFloat)
        {
            ss << "medianfp32";
        }
        else if(dtype == miopenHalf)
        {
            ss << "medianfp16";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "medianbfp16";
        }

        ss << " -in_dims " << miopen::deref(inputDesc).GetLengths();
        ss << " -out_dims " << miopen::deref(outputDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        if(is_fwd)
        {
            ss << " -dim " << dim;
            ss << " -keepdim " << keepdim;
        }

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenMedianForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const void* input,
                                              const miopenTensorDescriptor_t outputDesc,
                                              void* output,
                                              const miopenTensorDescriptor_t indicesDesc,
                                              size_t* indices,
                                              const uint64_t dim,
                                              const bool keepdim)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputDesc, input, outputDesc, output, indicesDesc, indices, dim, keepdim);

    LogCmdMedian(inputDesc, outputDesc, dim, keepdim, true);

    return miopen::try_([&] {
        miopen::median::MedianForward(miopen::deref(handle),
                                      miopen::deref(inputDesc),
                                      DataCast(input),
                                      miopen::deref(outputDesc),
                                      DataCast(output),
                                      miopen::deref(indicesDesc),
                                      indices,
                                      dim,
                                      keepdim);
    });
};

extern "C" miopenStatus_t miopenMedianBackward(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t outputGradDesc,
                                               const void* outputGrad,
                                               const miopenTensorDescriptor_t indicesDesc,
                                               const size_t* indices,
                                               const miopenTensorDescriptor_t inputGradDesc,
                                               void* inputGrad,
                                               const uint64_t dim,
                                               const bool keepdim)
{
    MIOPEN_LOG_FUNCTION(handle,
                        outputGradDesc,
                        outputGrad,
                        indicesDesc,
                        indices,
                        inputGradDesc,
                        inputGrad,
                        dim,
                        keepdim);

    LogCmdMedian(inputGradDesc, outputGradDesc, dim, keepdim, false);

    return miopen::try_([&] {
        miopen::median::MedianBackward(miopen::deref(handle),
                                       miopen::deref(outputGradDesc),
                                       DataCast(outputGrad),
                                       miopen::deref(indicesDesc),
                                       indices,
                                       miopen::deref(inputGradDesc),
                                       DataCast(inputGrad),
                                       dim,
                                       keepdim);
    });
}
