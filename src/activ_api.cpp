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
#include <miopen/activ.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>

#include <array>
#include <initializer_list>

extern "C" miopenStatus_t miopenCreateActivationDescriptor(miopenActivationDescriptor_t* activDesc)
{

    MIOPEN_LOG_FUNCTION(activDesc);
    return miopen::try_([&] { miopen::deref(activDesc) = new miopen::ActivationDescriptor(); });
}

extern "C" miopenStatus_t miopenSetActivationDescriptor(miopenActivationDescriptor_t activDesc,
                                                        miopenActivationMode_t mode,
                                                        double activAlpha,
                                                        double activBeta,
                                                        double activGamma)
{

    MIOPEN_LOG_FUNCTION(activDesc, mode, activAlpha, activBeta, activGamma);
    return miopen::try_([&] {
        std::initializer_list<double> parms = {activAlpha, activBeta, activGamma};
        miopen::deref(activDesc)            = miopen::ActivationDescriptor(mode, parms.begin());
    });
}

extern "C" miopenStatus_t miopenGetActivationDescriptor(miopenActivationDescriptor_t activDesc,
                                                        miopenActivationMode_t* mode,
                                                        double* activAlpha,
                                                        double* activBeta,
                                                        double* activGamma)
{

    MIOPEN_LOG_FUNCTION(activDesc, mode, activAlpha, activBeta, activGamma);
    return miopen::try_([&] {
        *mode       = miopen::deref(activDesc).GetMode();
        *activAlpha = miopen::deref(activDesc).GetAlpha();
        *activBeta  = miopen::deref(activDesc).GetBeta();
        *activGamma = miopen::deref(activDesc).GetGamma();
    });
}

static void LogCmdActivation(const miopenTensorDescriptor_t xDesc,
                             const miopenActivationDescriptor_t activDesc,
                             const bool Fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            ss << "activfp16";
        }
        else
        {
            ss << "activ";
        }
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0] << " -c "
           << miopen::deref(xDesc).GetLengths()[1] << " -H " << miopen::deref(xDesc).GetLengths()[2]
           << " -W " << miopen::deref(xDesc).GetLengths()[3] << " -m "
           << miopen::deref(activDesc).GetMode() << " --forw " << (Fwd ? "1" : "2") << " -A "
           << miopen::deref(activDesc).GetAlpha() << " -B " << miopen::deref(activDesc).GetBeta()
           << " -G " << miopen::deref(activDesc).GetGamma();
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
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

    MIOPEN_LOG_FUNCTION(handle, activDesc, alpha, xDesc, x, beta, yDesc, y);

    // bfloat16 not supported for activation operation
    if(miopen::deref(yDesc).GetType() == miopenBFloat16 ||
       miopen::deref(xDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdActivation(xDesc, activDesc, true);
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
    MIOPEN_LOG_FUNCTION(handle, activDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

    // bfloat16 not supported for activation operation
    if(miopen::deref(yDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dyDesc).GetType() == miopenBFloat16 ||
       miopen::deref(xDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dxDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    LogCmdActivation(xDesc, activDesc, false);
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

    MIOPEN_LOG_FUNCTION(activDesc);
    return miopen::try_([&] { miopen_destroy_object(activDesc); });
}
