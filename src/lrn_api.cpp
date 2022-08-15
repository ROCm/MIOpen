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
#include <miopen/errors.hpp>
#include <miopen/lrn.hpp>
#include <miopen/logger.hpp>

extern "C" miopenStatus_t miopenCreateLRNDescriptor(miopenLRNDescriptor_t* lrnDesc)
{

    return miopen::try_([&] { miopen::deref(lrnDesc) = new miopen::LRNDescriptor(); });
}

extern "C" miopenStatus_t miopenSetLRNDescriptor(miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t mode,
                                                 unsigned int lrnN,
                                                 double lrnAlpha,
                                                 double lrnBeta,
                                                 double lrnK)
{
    MIOPEN_LOG_FUNCTION(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
    return miopen::try_([&] {
        std::initializer_list<double> parms = {lrnAlpha, lrnBeta, lrnK};
        miopen::deref(lrnDesc)              = miopen::LRNDescriptor(mode, lrnN, parms.begin());
    });
}

extern "C" miopenStatus_t miopenGetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t* mode,
                                                 unsigned int* lrnN,
                                                 double* lrnAlpha,
                                                 double* lrnBeta,
                                                 double* lrnK)
{

    MIOPEN_LOG_FUNCTION(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
    return miopen::try_([&] {
        *mode     = miopen::deref(lrnDesc).GetMode();
        *lrnN     = miopen::deref(lrnDesc).GetN();
        *lrnAlpha = miopen::deref(lrnDesc).GetAlpha();
        *lrnBeta  = miopen::deref(lrnDesc).GetBeta();
        *lrnK     = miopen::deref(lrnDesc).GetK();
    });
}

extern "C" miopenStatus_t miopenLRNGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                    size_t* workSpaceSize)
{

    // TODO: Supporting size 4 bytes only
    return miopen::try_([&] {
        miopen::deref(workSpaceSize) = miopen::deref(yDesc).GetLengths()[0] *
                                       miopen::deref(yDesc).GetStrides()[0] * sizeof(float);
    });
}

static void
LogCmdLRN(const miopenLRNDescriptor_t lrnDesc, const miopenTensorDescriptor_t xDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenFloat)
            ss << "lrn";
        else if(miopen::deref(xDesc).GetType() == miopenHalf)
            ss << "lrnfp16";
        // clang-format off
        ss << " -A " << miopen::deref(lrnDesc).GetAlpha() 
           << " -B " << miopen::deref(lrnDesc).GetBeta() 
           << " -F " << ((is_fwd) ? "1" : "2") 
           << " -H " << miopen::deref(xDesc).GetLengths()[2] 
           << " -K " << miopen::deref(lrnDesc).GetK()
           << " -N " << miopen::deref(lrnDesc).GetN()
           << " -W " << miopen::deref(xDesc).GetLengths()[3] 
           << " -c " << miopen::deref(xDesc).GetLengths()[1]
           << " -m " << miopen::deref(lrnDesc).GetMode() 
           << " -n " << miopen::deref(xDesc).GetLengths()[0];
        // clang-format on

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenLRNForward(miopenHandle_t handle,
                                           const miopenLRNDescriptor_t lrnDesc,
                                           const void* alpha,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const void* beta,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           bool do_backward,
                                           void* workSpace)
{

    MIOPEN_LOG_FUNCTION(handle, lrnDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace);

    // bfloat16 not supported for lrn operation
    if(miopen::deref(yDesc).GetType() == miopenBFloat16 ||
       miopen::deref(xDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    LogCmdLRN(lrnDesc, xDesc, true);

    return miopen::try_([&] {
        miopen::deref(lrnDesc).Forward(miopen::deref(handle),
                                       alpha,
                                       miopen::deref(xDesc),
                                       DataCast(x),
                                       beta,
                                       miopen::deref(yDesc),
                                       DataCast(y),
                                       do_backward,
                                       DataCast(workSpace));
    });
}

extern "C" miopenStatus_t miopenLRNBackward(miopenHandle_t handle,
                                            const miopenLRNDescriptor_t lrnDesc,
                                            const void* alpha,
                                            const miopenTensorDescriptor_t yDesc,
                                            const void* y,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const void* dy,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const void* beta,
                                            const miopenTensorDescriptor_t dxDesc,
                                            void* dx,
                                            const void* workSpace)
{

    MIOPEN_LOG_FUNCTION(
        handle, lrnDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);

    // bfloat16 not supported for lrn operation
    if(miopen::deref(yDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dyDesc).GetType() == miopenBFloat16 ||
       miopen::deref(xDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dxDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    LogCmdLRN(lrnDesc, xDesc, false);
    return miopen::try_([&] {
        miopen::deref(lrnDesc).Backward(miopen::deref(handle),
                                        alpha,
                                        miopen::deref(yDesc),
                                        DataCast(y),
                                        miopen::deref(dyDesc),
                                        DataCast(dy),
                                        miopen::deref(xDesc),
                                        DataCast(x),
                                        beta,
                                        miopen::deref(dxDesc),
                                        DataCast(dx),
                                        DataCast(workSpace));
    });
}

extern "C" miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc)
{
    MIOPEN_LOG_FUNCTION(lrnDesc);
    return miopen::try_([&] { miopen_destroy_object(lrnDesc); });
}
