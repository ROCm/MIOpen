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

#include <miopen/avgpool.hpp>
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

static void LogCmdAvgPool(const miopenTensorDescriptor_t xDesc,
                          const miopenTensorDescriptor_t oDesc,
                          const bool count_include_pad,
                          const int32_t divisor_override,
                          const bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "avgpoolfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "avgpoolfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "avgpoolbfp16";
        }

        MIOPEN_LOG_FUNCTION(xDesc, oDesc, count_include_pad, divisor_override);
        ss << " -Is " << miopen::deref(xDesc).GetLengths();
        ss << " -Os " << miopen::deref(oDesc).GetLengths();
        ss << " -Si " << miopen::deref(xDesc).GetStrides();
        ss << " -So " << miopen::deref(oDesc).GetStrides();
        ss << " -Cp " << count_include_pad;
        ss << " -Do " << divisor_override;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenAvgPoolForward(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t inputDesc,
                                               const void* input,
                                               const miopenTensorDescriptor_t outputDesc,
                                               void* output,
                                               const miopenTensorDescriptor_t strideDesc,
                                               const void* stride,
                                               const miopenTensorDescriptor_t paddingDesc,
                                               const void* padding,
                                               const miopenTensorDescriptor_t kinforDesc,
                                               const void* kinfor,
                                               const bool count_include_pad,
                                               const int32_t divisor_override)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        outputDesc,
                        output,
                        strideDesc,
                        stride,
                        paddingDesc,
                        padding,
                        kinforDesc,
                        kinfor,
                        count_include_pad,
                        divisor_override);

    LogCmdAvgPool(inputDesc, outputDesc, count_include_pad, divisor_override, true);
    return miopen::try_([&] {
        miopen::AvgPoolForward(miopen::deref(handle),
                               miopen::deref(inputDesc),
                               DataCast(input),
                               miopen::deref(outputDesc),
                               DataCast(output),
                               miopen::deref(strideDesc),
                               DataCast(stride),
                               miopen::deref(paddingDesc),
                               DataCast(padding),
                               miopen::deref(kinforDesc),
                               DataCast(kinfor),
                               count_include_pad,
                               divisor_override);
    });
}

extern "C" miopenStatus_t miopenAvgPoolBackward(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t outputGradDesc,
                                                const void* output_grad,
                                                const miopenTensorDescriptor_t inputGradDesc,
                                                void* input_grad,
                                                const miopenTensorDescriptor_t strideDesc,
                                                const void* stride,
                                                const miopenTensorDescriptor_t paddingDesc,
                                                const void* padding,
                                                const miopenTensorDescriptor_t kinforDesc,
                                                const void* kinfor,
                                                const bool count_include_pad,
                                                const int32_t divisor_override)
{
    MIOPEN_LOG_FUNCTION(handle,
                        outputGradDesc,
                        output_grad,
                        inputGradDesc,
                        input_grad,
                        strideDesc,
                        stride,
                        paddingDesc,
                        padding,
                        kinforDesc,
                        kinfor,
                        count_include_pad,
                        divisor_override);

    LogCmdAvgPool(inputGradDesc, outputGradDesc, count_include_pad, divisor_override, false);
    return miopen::try_([&] {
        miopen::AvgPoolBackward(miopen::deref(handle),
                                miopen::deref(outputGradDesc),
                                DataCast(output_grad),
                                miopen::deref(inputGradDesc),
                                DataCast(input_grad),
                                miopen::deref(strideDesc),
                                DataCast(stride),
                                miopen::deref(paddingDesc),
                                DataCast(padding),
                                miopen::deref(kinforDesc),
                                DataCast(kinfor),
                                count_include_pad,
                                divisor_override);
    });
}
