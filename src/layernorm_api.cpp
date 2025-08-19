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

#include <miopen/layernorm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void
LogCmdLayerNorm(const miopenTensorDescriptor_t xDesc, const miopenNormMode_t mode, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "layernormfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "layernormfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "layernormbfp16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(xDesc, &size);
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0] << " -c "
           << miopen::deref(xDesc).GetLengths()[1];
        if(size == 5)
        {
            ss << " -D " << miopen::deref(xDesc).GetLengths()[2] << " -H "
               << miopen::deref(xDesc).GetLengths()[3] << " -W "
               << miopen::deref(xDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -H " << miopen::deref(xDesc).GetLengths()[2] << " -W "
               << miopen::deref(xDesc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " -W " << miopen::deref(xDesc).GetLengths()[2];
        }

        ss << " -F " << ((is_fwd) ? "1" : "2") << " -m " << mode;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenLayerNormForward(miopenHandle_t handle,
                                                 miopenNormMode_t mode,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const miopenTensorDescriptor_t weightDesc,
                                                 const void* weight,
                                                 const miopenTensorDescriptor_t biasDesc,
                                                 const void* bias,
                                                 const float epsilon,
                                                 const int32_t normalized_dim,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 const miopenTensorDescriptor_t meanDesc,
                                                 void* mean,
                                                 const miopenTensorDescriptor_t rstdDesc,
                                                 void* rstd)
{
    MIOPEN_LOG_FUNCTION(handle,
                        mode,
                        xDesc,
                        x,
                        weightDesc,
                        weight,
                        biasDesc,
                        bias,
                        epsilon,
                        normalized_dim,
                        yDesc,
                        y,
                        meanDesc,
                        mean,
                        rstdDesc,
                        rstd);

    LogCmdLayerNorm(xDesc, mode, true);
    return miopen::try_([&] {
        miopen::LayerNormForward(miopen::deref(handle),
                                 miopen::deref(xDesc),
                                 DataCast(x),
                                 miopen::deref(weightDesc),
                                 DataCast(weight),
                                 miopen::deref(biasDesc),
                                 DataCast(bias),
                                 miopen::deref(yDesc),
                                 DataCast(y),
                                 miopen::deref(meanDesc),
                                 DataCast(mean),
                                 miopen::deref(rstdDesc),
                                 DataCast(rstd),
                                 mode,
                                 epsilon,
                                 normalized_dim);
    });
}

extern "C" miopenStatus_t
miopenGetLayerNormBackwardWorkspaceSize(miopenHandle_t handle,
                                        miopenNormMode_t mode,
                                        const miopenTensorDescriptor_t dyDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenTensorDescriptor_t weightDesc,
                                        const miopenTensorDescriptor_t meanDesc,
                                        const miopenTensorDescriptor_t rstdDesc,
                                        const int32_t normalized_dim,
                                        const miopenTensorDescriptor_t dxDesc,
                                        const miopenTensorDescriptor_t dwDesc,
                                        const miopenTensorDescriptor_t dbDesc,
                                        size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle,
                        mode,
                        dyDesc,
                        xDesc,
                        weightDesc,
                        meanDesc,
                        rstdDesc,
                        normalized_dim,
                        dxDesc,
                        dwDesc,
                        dbDesc);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetLayerNormBackwardWorkspaceSize(miopen::deref(handle),
                                                      miopen::deref(dyDesc),
                                                      miopen::deref(xDesc),
                                                      miopen::deref(weightDesc),
                                                      miopen::deref(meanDesc),
                                                      miopen::deref(rstdDesc),
                                                      miopen::deref(dxDesc),
                                                      miopen::deref(dwDesc),
                                                      miopen::deref(dbDesc),
                                                      mode,
                                                      normalized_dim);
    });
}

extern "C" miopenStatus_t miopenLayerNormBackward(miopenHandle_t handle,
                                                  miopenNormMode_t mode,
                                                  void* workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const miopenTensorDescriptor_t dyDesc,
                                                  const void* dy,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const miopenTensorDescriptor_t weightDesc,
                                                  const void* weight,
                                                  const miopenTensorDescriptor_t meanDesc,
                                                  const void* mean,
                                                  const miopenTensorDescriptor_t rstdDesc,
                                                  const void* rstd,
                                                  const int32_t normalized_dim,
                                                  const miopenTensorDescriptor_t dxDesc,
                                                  void* dx,
                                                  const miopenTensorDescriptor_t dwDesc,
                                                  void* dw,
                                                  const miopenTensorDescriptor_t dbDesc,
                                                  void* db)
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
                        meanDesc,
                        mean,
                        rstdDesc,
                        rstd,
                        normalized_dim,
                        dxDesc,
                        dx,
                        dwDesc,
                        dw,
                        dbDesc,
                        db);

    LogCmdLayerNorm(dyDesc, mode, false);
    return miopen::try_([&] {
        miopen::LayerNormBackward(miopen::deref(handle),
                                  DataCast(workspace),
                                  workspaceSizeInBytes,
                                  miopen::deref(dyDesc),
                                  DataCast(dy),
                                  miopen::deref(xDesc),
                                  DataCast(x),
                                  miopen::deref(weightDesc),
                                  DataCast(weight),
                                  miopen::deref(meanDesc),
                                  DataCast(mean),
                                  miopen::deref(rstdDesc),
                                  DataCast(rstd),
                                  miopen::deref(dxDesc),
                                  DataCast(dx),
                                  miopen::deref(dwDesc),
                                  DataCast(dw),
                                  miopen::deref(dbDesc),
                                  DataCast(db),
                                  mode,
                                  normalized_dim);
    });
}
