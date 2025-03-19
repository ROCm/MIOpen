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

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/pad_constant.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenPadConstantFwd(miopenHandle_t handle,
                                               miopenTensorDescriptor_t xDesc,
                                               miopenTensorDescriptor_t yDesc,
                                               const void* x,
                                               void* y,
                                               const int64_t* padding,
                                               const int padding_size,
                                               const float value)
{
    MIOPEN_LOG_FUNCTION(handle, xDesc, yDesc, x, y, padding, padding_size, value);

    return miopen::try_([&] {
        miopen::pad_constant::PadConstantForward(miopen::deref(handle),
                                                 miopen::deref(xDesc),
                                                 miopen::deref(yDesc),
                                                 DataCast(x),
                                                 DataCast(y),
                                                 padding,
                                                 padding_size,
                                                 value);
    });
}

extern "C" miopenStatus_t miopenPadConstantBwd(miopenHandle_t handle,
                                               miopenTensorDescriptor_t dxDesc,
                                               miopenTensorDescriptor_t dyDesc,
                                               void* dx,
                                               const void* dy,
                                               const int64_t* padding,
                                               const int padding_size)
{
    MIOPEN_LOG_FUNCTION(handle, dxDesc, dyDesc, dx, dy, padding, padding_size);

    return miopen::try_([&] {
        miopen::pad_constant::PadConstantBackward(miopen::deref(handle),
                                                  miopen::deref(dxDesc),
                                                  miopen::deref(dyDesc),
                                                  DataCast(dx),
                                                  DataCast(dy),
                                                  padding,
                                                  padding_size);
    });
}
