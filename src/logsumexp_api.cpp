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

#include <miopen/logsumexp.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

#include <vector>

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

static void LogCmdLogSumExp(const miopenTensorDescriptor_t inputDesc,
                            const miopenTensorDescriptor_t outputDesc,
                            const int* dims,
                            const int num_dims,
                            bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenFloat)
        {
            ss << "logsumexpfp32";
        }
        else if(dtype == miopenHalf)
        {
            ss << "logsumexpfp16";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "logsumexpbfp16";
        }

        ss << " -iShape " << miopen::deref(inputDesc).GetLengths();
        ss << " -oShape " << miopen::deref(outputDesc).GetLengths();

        ss << " -dims ";
        for(int i = 0; i < num_dims; i++)
        {
            ss << dims[i] << " ";
        }

        ss << " -F " << ((is_fwd) ? "true" : "false");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
};

extern "C" miopenStatus_t miopenLogSumExpForward(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t inputDesc,
                                                 const void* input,
                                                 const miopenTensorDescriptor_t outputDesc,
                                                 void* output,
                                                 const int* dims,
                                                 const size_t num_dims)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, dims, num_dims);

    LogCmdLogSumExp(inputDesc, outputDesc, dims, num_dims, true);
    return miopen::try_([&] {
        miopen::LogSumExpForward(miopen::deref(handle),
                                 miopen::deref(inputDesc),
                                 DataCast(input),
                                 miopen::deref(outputDesc),
                                 DataCast(output),
                                 dims,
                                 num_dims);
    });
}

extern "C" miopenStatus_t miopenLogSumExpBackward(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  const void* output,
                                                  const miopenTensorDescriptor_t outputGradDesc,
                                                  const void* outputGrad,
                                                  const miopenTensorDescriptor_t inputGradDesc,
                                                  void* inputGrad,
                                                  const int* dims,
                                                  const size_t num_dims)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        outputDesc,
                        output,
                        outputGradDesc,
                        outputGrad,
                        inputGradDesc,
                        inputGrad,
                        dims,
                        num_dims);

    LogCmdLogSumExp(inputDesc, outputDesc, dims, num_dims, false);
    return miopen::try_([&] {
        miopen::LogSumExpBackward(miopen::deref(handle),
                                  miopen::deref(inputDesc),
                                  DataCast(input),
                                  miopen::deref(outputDesc),
                                  DataCast(output),
                                  miopen::deref(outputGradDesc),
                                  DataCast(outputGrad),
                                  miopen::deref(inputGradDesc),
                                  DataCast(inputGrad),
                                  dims,
                                  num_dims);
    });
}
