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
#include <miopen/softmax.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdSoftmax(const miopenTensorDescriptor_t xDesc,
                          const miopenSoftmaxAlgorithm_t algo,
                          const miopenSoftmaxMode_t mode,
                          const void* alpha,
                          const void* beta,
                          bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenHalf)
            ss << "softmaxfp16";
        else
            ss << "softmax";
        // clang-format off
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0] 
           << " -c " << miopen::deref(xDesc).GetLengths()[1] 
           << " -H " << miopen::deref(xDesc).GetLengths()[2]
           << " -W " << miopen::deref(xDesc).GetLengths()[3]
           << " -F " << ((is_fwd) ? "1" : "2") 
           << " -a " << algo << " -m " << mode
           << " -A " << ((alpha == nullptr)?"1": std::to_string(*static_cast<const float*>(alpha)))
           << " -B " << ((beta == nullptr)?"0":std::to_string(*static_cast<const float*>(beta)));
        // clang-format on
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenSoftmaxForward(miopenHandle_t handle,
                                               const void* alpha,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const void* beta,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y)
{
    MIOPEN_LOG_FUNCTION(alpha, xDesc, x, beta, yDesc, y);

    // bfloat16 not supported for softmax operation
    if(miopen::deref(xDesc).GetType() == miopenBFloat16 ||
       miopen::deref(yDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdSoftmax(xDesc, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL, alpha, beta, true);
    return miopen::try_([&] {
        miopen::SoftmaxForward(miopen::deref(handle),
                               alpha,
                               beta,
                               miopen::deref(xDesc),
                               DataCast(x),
                               miopen::deref(yDesc),
                               DataCast(y),
                               miopenSoftmaxAlgorithm_t(1),
                               miopenSoftmaxMode_t(1),
                               0,
                               0);
    });
}

extern "C" miopenStatus_t miopenSoftmaxBackward(miopenHandle_t handle,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const void* beta,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx)
{

    MIOPEN_LOG_FUNCTION(alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);

    // bfloat16 not supported for softmax operation
    if(miopen::deref(dyDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dxDesc).GetType() == miopenBFloat16 ||
       miopen::deref(yDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdSoftmax(dxDesc, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL, alpha, beta, false);

    return miopen::try_([&] {
        miopen::SoftmaxBackward(miopen::deref(handle),
                                alpha,
                                miopen::deref(yDesc),
                                DataCast(y),
                                miopen::deref(dyDesc),
                                DataCast(dy),
                                beta,
                                miopen::deref(dxDesc),
                                DataCast(dx),
                                miopenSoftmaxAlgorithm_t(1),
                                miopenSoftmaxMode_t(1),
                                0,
                                0,
                                0);
    });
}

extern "C" miopenStatus_t miopenSoftmaxForward_V2(miopenHandle_t handle,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  miopenSoftmaxAlgorithm_t algorithm,
                                                  miopenSoftmaxMode_t mode)
{
    MIOPEN_LOG_FUNCTION(alpha, xDesc, x, beta, yDesc, y, algorithm, mode);
    // check for supported data types
    if(miopen::deref(xDesc).GetType() == miopenBFloat16 ||
       miopen::deref(yDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdSoftmax(xDesc, algorithm, mode, alpha, beta, true);
    return miopen::try_([&] {
        miopen::SoftmaxForward(miopen::deref(handle),
                               alpha,
                               beta,
                               miopen::deref(xDesc),
                               DataCast(x),
                               miopen::deref(yDesc),
                               DataCast(y),
                               algorithm,
                               mode,
                               0,
                               0);
    });
}

extern "C" miopenStatus_t miopenSoftmaxBackward_V2(miopenHandle_t handle,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx,
                                                   miopenSoftmaxAlgorithm_t algorithm,
                                                   miopenSoftmaxMode_t mode)
{

    MIOPEN_LOG_FUNCTION(alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx, algorithm, mode);
    if(miopen::deref(yDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dyDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dxDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdSoftmax(dxDesc, algorithm, mode, alpha, beta, false);
    return miopen::try_([&] {
        miopen::SoftmaxBackward(miopen::deref(handle),
                                alpha,
                                miopen::deref(yDesc),
                                DataCast(y),
                                miopen::deref(dyDesc),
                                DataCast(dy),
                                beta,
                                miopen::deref(dxDesc),
                                DataCast(dx),
                                algorithm,
                                mode,
                                0,
                                0,
                                0);
    });
}
