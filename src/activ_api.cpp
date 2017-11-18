/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <array>
#include <initializer_list>
#include <miopen/activ.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>

extern "C" miopenStatus_t miopenCreateActivationDescriptor(miopenActivationDescriptor_t* activDesc)
{

    MIOPEN_LOG_FUNCTION(activDesc);
    return miopen::try_([&] { miopen::deref(activDesc) = new miopen::ActivationDescriptor(); });
}

extern "C" miopenStatus_t miopenSetActivationDescriptor(miopenActivationDescriptor_t activDesc,
                                                        miopenActivationMode_t mode,
                                                        double activAlpha,
                                                        double activBeta,
                                                        double activPower)
{

    MIOPEN_LOG_FUNCTION(activDesc, mode, activAlpha, activBeta, activPower);
    return miopen::try_([&] {
        std::initializer_list<double> parms = {activAlpha, activBeta, activPower};
        miopen::deref(activDesc)            = miopen::ActivationDescriptor(mode, parms.begin());
    });
}

extern "C" miopenStatus_t miopenGetActivationDescriptor(miopenActivationDescriptor_t activDesc,
                                                        miopenActivationMode_t* mode,
                                                        double* activAlpha,
                                                        double* activBeta,
                                                        double* activPower)
{

    MIOPEN_LOG_FUNCTION(activDesc, mode, activAlpha, activBeta, activPower);
    return miopen::try_([&] {
        *mode       = miopen::deref(activDesc).GetMode();
        *activAlpha = miopen::deref(activDesc).GetAlpha();
        *activBeta  = miopen::deref(activDesc).GetBeta();
        *activPower = miopen::deref(activDesc).GetPower();
    });
}

extern "C" miopenStatus_t miopenActivationForward(miopenHandle_t handle,
                                                  miopenActivationDescriptor_t activDesc,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y)
{

    MIOPEN_LOG_FUNCTION(activDesc, alpha, xDesc, x, beta, yDesc, y);
    return miopen::try_([&] {
        miopen::deref(activDesc).Forward(miopen::deref(handle),
                                         alpha,
                                         miopen::deref(xDesc),
                                         DataCast(x),
                                         beta,
                                         miopen::deref(yDesc),
                                         DataCast(y));
    });
}

extern "C" miopenStatus_t miopenActivationBackward(miopenHandle_t handle,
                                                   miopenActivationDescriptor_t activDesc,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx)
{
    MIOPEN_LOG_FUNCTION(activDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)

    return miopen::try_([&] {
        miopen::deref(activDesc).Backward(miopen::deref(handle),
                                          alpha,
                                          miopen::deref(yDesc),
                                          DataCast(y),
                                          miopen::deref(dyDesc),
                                          DataCast(dy),
                                          miopen::deref(xDesc),
                                          DataCast(x),
                                          beta,
                                          miopen::deref(dxDesc),
                                          DataCast(dx));
    });
}

extern "C" miopenStatus_t miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc)
{

    MIOPEN_LOG_FUNCTION(activDesc)
    return miopen::try_([&] { miopen_destroy_object(activDesc); });
}
