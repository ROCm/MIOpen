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

#include <miopen/fractionalmaxpool.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<uint64_t>& v)
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

inline void LogCmdFractionalMaxPool(const miopenTensorDescriptor_t xDesc,
                                    const miopenTensorDescriptor_t yDesc,
                                    const bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "fractionalmaxpoolfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "fractionalmaxpoolfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "fractionalmaxpoolbfp16";
        }
        ss << " -Xs ";
        ss << miopen::deref(xDesc).GetLengths();
        ss << " -Ys ";
        ss << miopen::deref(yDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenFractionalMaxPoolForward(miopenHandle_t handle,
                                                         const miopenTensorDescriptor_t inputDesc,
                                                         const void* input,
                                                         const miopenTensorDescriptor_t outputDesc,
                                                         void* output,
                                                         const miopenTensorDescriptor_t indicesDesc,
                                                         void* indices,
                                                         const int64_t KD,
                                                         const int64_t KH,
                                                         const int64_t KW)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputDesc, input, outputDesc, output, indicesDesc, indices, KD, KH, KW);

    LogCmdFractionalMaxPool(inputDesc, outputDesc, true);

    return miopen::try_([&] {
        miopen::fractionalmaxpool::FractionalMaxPoolForward(miopen::deref(handle),
                                                            miopen::deref(inputDesc),
                                                            DataCast(input),
                                                            miopen::deref(outputDesc),
                                                            DataCast(output),
                                                            miopen::deref(indicesDesc),
                                                            DataCast(indices),
                                                            KD,
                                                            KH,
                                                            KW);
    });
}

extern "C" miopenStatus_t
miopenFractionalMaxPoolBackward(miopenHandle_t handle,
                                const miopenTensorDescriptor_t indicesDesc,
                                const void* indices,
                                const miopenTensorDescriptor_t outputGradDesc,
                                const void* output_grad,
                                const miopenTensorDescriptor_t inputGradDesc,
                                void* input_grad)
{
    MIOPEN_LOG_FUNCTION(
        handle, indicesDesc, indices, outputGradDesc, output_grad, inputGradDesc, input_grad);

    LogCmdFractionalMaxPool(inputGradDesc, outputGradDesc, false);

    return miopen::try_([&] {
        miopen::fractionalmaxpool::FractionalMaxPoolBackward(miopen::deref(handle),
                                                             miopen::deref(indicesDesc),
                                                             DataCast(indices),
                                                             miopen::deref(outputGradDesc),
                                                             DataCast(output_grad),
                                                             miopen::deref(inputGradDesc),
                                                             DataCast(input_grad));
    });
}
