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

#include <miopen/rope.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdRoPE(const miopenTensorDescriptor_t xDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "ropefp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "ropefp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "ropebfp16";
        }

        std::string input_sz;
        auto input = miopen::deref(xDesc).GetLengths();
        for(int32_t i = 0; i < input.size(); ++i)
        {
            input_sz += std::to_string(input[i]);
            if(i != input.size() - 1)
                input_sz += "x";
        }

        ss << " -input " << input_sz;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenRoPEForward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t cosDesc,
                                            const void* cos,
                                            const miopenTensorDescriptor_t sinDesc,
                                            const void* sin,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y)
{

    MIOPEN_LOG_FUNCTION(handle, xDesc, x, cosDesc, cos, sinDesc, sin, yDesc, y);

    LogCmdRoPE(xDesc, true);

    return miopen::try_([&] {
        miopen::RoPEForward(miopen::deref(handle),
                            miopen::deref(xDesc),
                            DataCast(x),
                            miopen::deref(cosDesc),
                            DataCast(cos),
                            miopen::deref(sinDesc),
                            DataCast(sin),
                            miopen::deref(yDesc),
                            DataCast(y));
    });
}

extern "C" miopenStatus_t miopenRoPEBackward(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const void* dy,
                                             const miopenTensorDescriptor_t cosDesc,
                                             const void* cos,
                                             const miopenTensorDescriptor_t sinDesc,
                                             const void* sin,
                                             const miopenTensorDescriptor_t dxDesc,
                                             void* dx)
{

    MIOPEN_LOG_FUNCTION(handle, dyDesc, dy, cosDesc, cos, sinDesc, sin, dxDesc, dx);

    LogCmdRoPE(dyDesc, true);

    return miopen::try_([&] {
        miopen::RoPEBackward(miopen::deref(handle),
                             miopen::deref(dyDesc),
                             DataCast(dy),
                             miopen::deref(cosDesc),
                             DataCast(cos),
                             miopen::deref(sinDesc),
                             DataCast(sin),
                             miopen::deref(dxDesc),
                             DataCast(dx));
    });
}
