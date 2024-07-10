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

#include <miopen/outer.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdOuter(const miopenTensorDescriptor_t Desc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(Desc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "sumfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "sumfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "sumbfp16";
        }

        ss << " -N " << miopen::deref(Desc).GetLengths()[0] << " -M "
           << miopen::deref(Desc).GetLengths()[1];

        ss << " -F " << ((is_fwd) ? "1" : "0");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenOuterForward(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t x1Desc,
                                             const void* x1,
                                             const miopenTensorDescriptor_t x2Desc,
                                             const void* x2,
                                             const miopenTensorDescriptor_t yDesc,
                                             void* y)
{
    MIOPEN_LOG_FUNCTION(handle, x1Desc, x1, x2Desc, x2, yDesc, yDesc, y);

    LogCmdOuter(yDesc, true);

    return miopen::try_([&] {
        miopen::OuterForward(miopen::deref(handle),
                             miopen::deref(x1Desc),
                             DataCast(x1),
                             miopen::deref(x2Desc),
                             DataCast(x2),
                             miopen::deref(yDesc),
                             DataCast(y));
    });
}

extern "C" miopenStatus_t miopenOuterBackwardGrad1(miopenHandle_t handle,
                                                   const miopenTensorDescriptor_t x2Desc,
                                                   const void* x2,
                                                   const miopenTensorDescriptor_t x1GradDesc,
                                                   void* x1Grad,
                                                   const miopenTensorDescriptor_t yGradDesc,
                                                   const void* yGrad)
{
    MIOPEN_LOG_FUNCTION(handle, x2Desc, x2, x1GradDesc, x1Grad, yGradDesc, yGrad);

    LogCmdOuter(yGradDesc, false);

    return miopen::try_([&] {
        miopen::OuterBackwardGrad1(miopen::deref(handle),
                                   miopen::deref(x2Desc),
                                   DataCast(x2),
                                   miopen::deref(x1GradDesc),
                                   DataCast(x1Grad),
                                   miopen::deref(yGradDesc),
                                   DataCast(yGrad));
    });
}

extern "C" miopenStatus_t miopenOuterBackwardGrad2(miopenHandle_t handle,
                                                   const miopenTensorDescriptor_t x1Desc,
                                                   const void* x1,
                                                   const miopenTensorDescriptor_t x2GradDesc,
                                                   void* x2Grad,
                                                   const miopenTensorDescriptor_t yGradDesc,
                                                   const void* yGrad)
{
    MIOPEN_LOG_FUNCTION(handle, x1Desc, x1, x2GradDesc, x2Grad, yGradDesc, yGrad);

    LogCmdOuter(yGradDesc, true);

    return miopen::try_([&] {
        miopen::OuterBackwardGrad2(miopen::deref(handle),
                                   miopen::deref(x1Desc),
                                   DataCast(x1),
                                   miopen::deref(x2GradDesc),
                                   DataCast(x2Grad),
                                   miopen::deref(yGradDesc),
                                   DataCast(yGrad));
    });
}
