/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/lppool.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
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

static void LogCmdLPPool(const miopenTensorDescriptor_t iDesc,
                         const miopenTensorDescriptor_t oDesc,
                         const int64_t KD,
                         const int64_t KH,
                         const int64_t SD,
                         const int64_t SH,
                         const float norm_type,
                         const bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(iDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "lppoolfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "lppoolfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "lppoolbfp16";
        }

        MIOPEN_LOG_FUNCTION(iDesc, oDesc, KD, KH, SD, SH, norm_type, is_fwd);
        ss << " -Is " << miopen::deref(iDesc).GetLengths();
        ss << " -Os " << miopen::deref(oDesc).GetLengths();
        ss << " -Si " << miopen::deref(iDesc).GetStrides();
        ss << " -So " << miopen::deref(oDesc).GetStrides();
        ss << " -KD " << KD;
        ss << " -KH " << KH;
        ss << " -SD " << SD;
        ss << " -SH " << SH;
        ss << " -p " << norm_type;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenLPPoolForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const void* input,
                                              const miopenTensorDescriptor_t outputDesc,
                                              void* output,
                                              const int64_t KD,
                                              const int64_t KH,
                                              const int64_t SD,
                                              const int64_t SH,
                                              const float norm_type)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, KD, KH, SD, SH, norm_type);

    LogCmdLPPool(inputDesc, outputDesc, KD, KH, SD, SH, norm_type, true);

    return miopen::try_([&] {
        miopen::lppool::LPPoolForward(miopen::deref(handle),
                                      miopen::deref(inputDesc),
                                      DataCast(input),
                                      miopen::deref(outputDesc),
                                      DataCast(output),
                                      KD,
                                      KH,
                                      SD,
                                      SH,
                                      norm_type);
    });
}

extern "C" miopenStatus_t miopenLPPoolBackward(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t inputDesc,
                                               const void* input,
                                               const miopenTensorDescriptor_t outputDesc,
                                               const void* output,
                                               const miopenTensorDescriptor_t outputGradDesc,
                                               const void* output_grad,
                                               const miopenTensorDescriptor_t inputGradDesc,
                                               void* input_grad,
                                               const int64_t KD,
                                               const int64_t KH,
                                               const int64_t SD,
                                               const int64_t SH,
                                               const float norm_type)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        outputDesc,
                        output,
                        outputGradDesc,
                        output_grad,
                        inputGradDesc,
                        input_grad,
                        KD,
                        KH,
                        SD,
                        SH,
                        norm_type);

    LogCmdLPPool(inputGradDesc, outputGradDesc, KD, KH, SD, SH, norm_type, false);

    return miopen::try_([&] {
        miopen::lppool::LPPoolBackward(miopen::deref(handle),
                                       miopen::deref(inputDesc),
                                       DataCast(input),
                                       miopen::deref(outputDesc),
                                       DataCast(output),
                                       miopen::deref(outputGradDesc),
                                       DataCast(output_grad),
                                       miopen::deref(inputGradDesc),
                                       DataCast(input_grad),
                                       KD,
                                       KH,
                                       SD,
                                       SH,
                                       norm_type);
    });
}
