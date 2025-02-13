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

#include <miopen/t5layernorm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void
LogCmdT5LayerNorm(const miopenTensorDescriptor_t Desc, const miopenNormMode_t mode, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(Desc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "t5layernormfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "t5layernormfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "t5layernormbfp16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(Desc, &size);
        ss << " -n " << miopen::deref(Desc).GetLengths()[0] << " -c "
           << miopen::deref(Desc).GetLengths()[1];
        if(size == 5)
        {
            ss << " -D " << miopen::deref(Desc).GetLengths()[2] << " -H "
               << miopen::deref(Desc).GetLengths()[3] << " -W "
               << miopen::deref(Desc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -H " << miopen::deref(Desc).GetLengths()[2] << " -W "
               << miopen::deref(Desc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " -W " << miopen::deref(Desc).GetLengths()[2];
        }

        ss << " -F " << ((is_fwd) ? "1" : "2") << " -m " << mode;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenT5LayerNormForward(miopenHandle_t handle,
                                                   miopenNormMode_t mode,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t weightDesc,
                                                   const void* weight,
                                                   const float epsilon,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t rstdDesc,
                                                   void* rstd)
{
    MIOPEN_LOG_FUNCTION(
        handle, mode, xDesc, x, weightDesc, weight, epsilon, yDesc, y, rstdDesc, rstd);

    LogCmdT5LayerNorm(xDesc, mode, true);
    return miopen::try_([&] {
        miopen::T5LayerNormForward(miopen::deref(handle),
                                   miopen::deref(xDesc),
                                   DataCast(x),
                                   miopen::deref(weightDesc),
                                   DataCast(weight),
                                   miopen::deref(yDesc),
                                   DataCast(y),
                                   miopen::deref(rstdDesc),
                                   DataCast(rstd),
                                   mode,
                                   epsilon);
    });
}

extern "C" miopenStatus_t
miopenGetT5LayerNormBackwardWorkspaceSize(miopenHandle_t handle,
                                          miopenNormMode_t mode,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const miopenTensorDescriptor_t xDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          const miopenTensorDescriptor_t rstdDesc,
                                          const miopenTensorDescriptor_t dxDesc,
                                          const miopenTensorDescriptor_t dwDesc,
                                          size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, mode, dyDesc, xDesc, weightDesc, rstdDesc, dxDesc, dwDesc);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetT5LayerNormBackwardWorkspaceSize(miopen::deref(handle),
                                                        miopen::deref(dyDesc),
                                                        miopen::deref(xDesc),
                                                        miopen::deref(weightDesc),
                                                        miopen::deref(rstdDesc),
                                                        miopen::deref(dxDesc),
                                                        miopen::deref(dwDesc),
                                                        mode);
    });
};

extern "C" miopenStatus_t miopenT5LayerNormBackward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    void* workspace,
                                                    size_t workspaceSizeInBytes,
                                                    const miopenTensorDescriptor_t dyDesc,
                                                    const void* dy,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t weightDesc,
                                                    const void* weight,
                                                    const miopenTensorDescriptor_t rstdDesc,
                                                    const void* rstd,
                                                    const miopenTensorDescriptor_t dxDesc,
                                                    void* dx,
                                                    const miopenTensorDescriptor_t dwDesc,
                                                    void* dw)
{
    MIOPEN_LOG_FUNCTION(handle,
                        mode,
                        workspace,
                        workspaceSizeInBytes,
                        dyDesc,
                        dy,
                        xDesc,
                        x,
                        weightDesc,
                        weight,
                        rstdDesc,
                        rstd,
                        dxDesc,
                        dx,
                        dw);

    LogCmdT5LayerNorm(dyDesc, mode, true);
    return miopen::try_([&] {
        miopen::T5LayerNormBackward(miopen::deref(handle),
                                    DataCast(workspace),
                                    workspaceSizeInBytes,
                                    miopen::deref(dyDesc),
                                    DataCast(dy),
                                    miopen::deref(xDesc),
                                    DataCast(x),
                                    miopen::deref(weightDesc),
                                    DataCast(weight),
                                    miopen::deref(rstdDesc),
                                    DataCast(rstd),
                                    miopen::deref(dxDesc),
                                    DataCast(dx),
                                    miopen::deref(dwDesc),
                                    DataCast(dw),
                                    mode);
    });
}
