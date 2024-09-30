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

#include "miopen/miopen.h"
#include <cstdio>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/mseloss.hpp>

extern "C" miopenStatus_t miopenMSELossForward(miopenHandle_t handle,
                                               miopenTensorDescriptor_t xDesc,
                                               miopenTensorDescriptor_t yDesc,
                                               miopenTensorDescriptor_t zDesc,
                                               const void* x,
                                               const void* y,
                                               void* z,
                                               void* ws,
                                               const float divisor)
{
    MIOPEN_LOG_FUNCTION(xDesc, yDesc, x, y, z, divisor);

    return miopen::try_([&] {
        miopen::MSELossForward(miopen::deref(handle),
                               miopen::deref(xDesc),
                               miopen::deref(yDesc),
                               miopen::deref(zDesc),
                               DataCast(x),
                               DataCast(y),
                               DataCast(z),
                               DataCast(ws),
                               divisor);
    });
}

extern "C" miopenStatus_t miopenGetMSELossForwardWorkspaceSize(miopenHandle_t handle,
                                                               miopenTensorDescriptor_t xDesc,
                                                               miopenTensorDescriptor_t yDesc,
                                                               size_t* size)
{
    MIOPEN_LOG_FUNCTION(xDesc, yDesc, size);

    return miopen::try_([&] {
        miopen::deref(size) = miopen::MSELossForwardGetWorkspaceSize(
            miopen::deref(handle), miopen::deref(xDesc), miopen::deref(yDesc));
    });
}

extern "C" miopenStatus_t miopenMSELossBackward(miopenHandle_t handle,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t yDesc,
                                                miopenTensorDescriptor_t zDesc,
                                                miopenTensorDescriptor_t dxDesc,
                                                miopenTensorDescriptor_t dyDesc,
                                                const void* x,
                                                const void* y,
                                                const void* z,
                                                void* dx,
                                                void* dy,
                                                const float divisor)
{
    MIOPEN_LOG_FUNCTION(xDesc, yDesc, zDesc, dxDesc, dyDesc, x, y, z, dx, dy, divisor);

    return miopen::try_([&] {
        miopen::MSELossBackward(miopen::deref(handle),
                                miopen::deref(xDesc),
                                miopen::deref(yDesc),
                                miopen::deref(zDesc),
                                miopen::deref(dxDesc),
                                miopen::deref(dyDesc),
                                DataCast(x),
                                DataCast(y),
                                DataCast(z),
                                DataCast(dx),
                                DataCast(dy),
                                divisor);
    });
}
