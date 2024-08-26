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

#include <miopen/addlayernorm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void
LogCmdAddLayerNorm(const miopenTensorDescriptor_t xDesc, const miopenNormMode_t mode, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "addlayernormfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "addlayernormfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "addlayernormbfp16";
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

extern "C" miopenStatus_t miopenAddLayerNormForward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t x2Desc,
                                                    const void* x2,
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
                        x2Desc,
                        x2,
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

    LogCmdAddLayerNorm(xDesc, mode, true);
    return miopen::try_([&] {
        miopen::AddLayerNormForward(miopen::deref(handle),
                                    miopen::deref(xDesc),
                                    DataCast(x),
                                    miopen::deref(x2Desc),
                                    DataCast(x2),
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
