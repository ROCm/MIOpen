/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/argmax.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdArgmax(const miopenTensorDescriptor_t xDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "argmaxfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "argmaxfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "argmaxbfp16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(xDesc, &size);
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0];
        if(size == 5)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1] << " -D "
               << miopen::deref(xDesc).GetLengths()[2] << " -H "
               << miopen::deref(xDesc).GetLengths()[3] << " -W "
               << miopen::deref(xDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1] << " -H "
               << miopen::deref(xDesc).GetLengths()[2] << " -W "
               << miopen::deref(xDesc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1] << " -W "
               << miopen::deref(xDesc).GetLengths()[2];
        }
        else if(size == 2)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1];
        }

        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenArgmaxForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const int32_t dim,
                                              const miopenTensorDescriptor_t yDesc,
                                              void* y)
{
    MIOPEN_LOG_FUNCTION(handle, xDesc, x, dim, yDesc, y);

    LogCmdArgmax(xDesc, true);
    return miopen::try_([&] {
        miopen::ArgmaxForward(miopen::deref(handle),
                              miopen::deref(xDesc),
                              DataCast(x),
                              miopen::deref(yDesc),
                              DataCast(y),
                              dim);
    });
}
