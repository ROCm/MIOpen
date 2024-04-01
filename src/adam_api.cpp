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
#include <miopen/adam.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdAdam(const miopenTensorDescriptor_t paramDesc)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(paramDesc).GetType();
        if(dtype == miopenFloat)
        {
            ss << "adam";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "adamf16";
        }

        std::string batch_sz;
        auto dims = miopen::deref(paramDesc).GetLengths();
        for(auto dim : dims)
        {
            batch_sz += std::to_string(dim);
            batch_sz += ",";
        }
        batch_sz.pop_back();
        ss << " -dims " << batch_sz;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenAdam(miopenHandle_t handle,
                                     const miopenTensorDescriptor_t paramInDesc,
                                     const void* paramIn,
                                     const miopenTensorDescriptor_t gradInDesc,
                                     const void* gradIn,
                                     const miopenTensorDescriptor_t expAvgInDesc,
                                     const void* expAvgIn,
                                     const miopenTensorDescriptor_t expAvgSqInDesc,
                                     const void* expAvgSqIn,
                                     const miopenTensorDescriptor_t maxExpAvgSqInDesc,
                                     const void* maxExpAvgSqIn,
                                     const int step,
                                     const float lr,
                                     const float beta1,
                                     const float beta2,
                                     const float weight_decay,
                                     const float eps,
                                     const bool amsgrad,
                                     const bool maximize,
                                     const miopenTensorDescriptor_t paramOutDesc,
                                     void* paramOut,
                                     const miopenTensorDescriptor_t expAvgOutDesc,
                                     void* expAvgOut,
                                     const miopenTensorDescriptor_t expAvgSqOutDesc,
                                     void* expAvgSqOut,
                                     const miopenTensorDescriptor_t maxExpAvgSqOutDesc,
                                     void* maxExpAvgSqOut)
{
    MIOPEN_LOG_FUNCTION(handle,
                        paramInDesc,
                        paramIn,
                        gradInDesc,
                        gradIn,
                        expAvgInDesc,
                        expAvgIn,
                        expAvgSqInDesc,
                        expAvgSqIn,
                        maxExpAvgSqInDesc,
                        maxExpAvgSqIn,
                        step,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        amsgrad,
                        maximize,
                        paramOutDesc,
                        paramOut,
                        expAvgOutDesc,
                        expAvgOut,
                        expAvgSqOutDesc,
                        expAvgSqOut,
                        maxExpAvgSqOutDesc,
                        maxExpAvgSqOut);

    LogCmdAdam(paramInDesc);
    auto maxExpAvgSqInDescPtr =
        (maxExpAvgSqInDesc != nullptr) ? &miopen::deref(maxExpAvgSqInDesc) : nullptr;
    auto maxExpAvgSqOutDescPtr =
        (maxExpAvgSqOutDesc != nullptr) ? &miopen::deref(maxExpAvgSqOutDesc) : nullptr;

    return miopen::try_([&] {
        miopen::Adam(miopen::deref(handle),
                     miopen::deref(paramInDesc),
                     DataCast(paramIn),
                     miopen::deref(paramOutDesc),
                     DataCast(paramOut),
                     miopen::deref(gradInDesc),
                     DataCast(gradIn),
                     miopen::deref(expAvgInDesc),
                     DataCast(expAvgIn),
                     miopen::deref(expAvgOutDesc),
                     DataCast(expAvgOut),
                     miopen::deref(expAvgSqInDesc),
                     DataCast(expAvgSqIn),
                     miopen::deref(expAvgSqOutDesc),
                     DataCast(expAvgSqOut),
                     maxExpAvgSqInDescPtr,
                     DataCast(maxExpAvgSqIn),
                     maxExpAvgSqOutDescPtr,
                     DataCast(maxExpAvgSqOut),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     step,
                     lr,
                     beta1,
                     beta2,
                     weight_decay,
                     eps,
                     amsgrad,
                     maximize);
    });
}

extern "C" miopenStatus_t miopenAmpAdam(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t paramInDesc,
                                        const void* paramIn,
                                        const miopenTensorDescriptor_t gradInDesc,
                                        const void* gradIn,
                                        const miopenTensorDescriptor_t expAvgInDesc,
                                        const void* expAvgIn,
                                        const miopenTensorDescriptor_t expAvgSqInDesc,
                                        const void* expAvgSqIn,
                                        const miopenTensorDescriptor_t maxExpAvgSqInDesc,
                                        const void* maxExpAvgSqIn,
                                        const miopenTensorDescriptor_t gradScaleDesc,
                                        const void* gradScale,
                                        const miopenTensorDescriptor_t foundInfDesc,
                                        const void* foundInf,
                                        const miopenTensorDescriptor_t stepInDesc,
                                        const void* stepIn,
                                        const float lr,
                                        const float beta1,
                                        const float beta2,
                                        const float weight_decay,
                                        const float eps,
                                        const bool amsgrad,
                                        const bool maximize,
                                        const miopenTensorDescriptor_t paramOutDesc,
                                        void* paramOut,
                                        const miopenTensorDescriptor_t expAvgOutDesc,
                                        void* expAvgOut,
                                        const miopenTensorDescriptor_t expAvgSqOutDesc,
                                        void* expAvgSqOut,
                                        const miopenTensorDescriptor_t maxExpAvgSqOutDesc,
                                        void* maxExpAvgSqOut,
                                        const miopenTensorDescriptor_t stepOutDesc,
                                        void* stepOut)
{
    MIOPEN_LOG_FUNCTION(handle,
                        paramInDesc,
                        paramIn,
                        gradInDesc,
                        gradIn,
                        expAvgInDesc,
                        expAvgIn,
                        expAvgSqInDesc,
                        expAvgSqIn,
                        maxExpAvgSqInDesc,
                        maxExpAvgSqIn,
                        gradScaleDesc,
                        gradScale,
                        foundInfDesc,
                        foundInf,
                        stepInDesc,
                        stepIn,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        amsgrad,
                        maximize,
                        paramOutDesc,
                        paramOut,
                        expAvgOutDesc,
                        expAvgOut,
                        expAvgSqOutDesc,
                        expAvgSqOut,
                        maxExpAvgSqOutDesc,
                        maxExpAvgSqOut,
                        stepOutDesc,
                        stepOut);

    LogCmdAdam(paramInDesc);
    auto maxExpAvgSqInDescPtr =
        (maxExpAvgSqInDesc != nullptr) ? &miopen::deref(maxExpAvgSqInDesc) : nullptr;
    auto maxExpAvgSqOutDescPtr =
        (maxExpAvgSqOutDesc != nullptr) ? &miopen::deref(maxExpAvgSqOutDesc) : nullptr;
    auto gradScaleDescPtr = (gradScaleDesc != nullptr) ? &miopen::deref(gradScaleDesc) : nullptr;
    auto foundInfDescPtr  = (foundInfDesc != nullptr) ? &miopen::deref(foundInfDesc) : nullptr;
    auto stepOutDescPtr   = (stepOutDesc != nullptr) ? &miopen::deref(stepOutDesc) : nullptr;

    return miopen::try_([&] {
        miopen::Adam(miopen::deref(handle),
                     miopen::deref(paramInDesc),
                     DataCast(paramIn),
                     miopen::deref(paramOutDesc),
                     DataCast(paramOut),
                     miopen::deref(gradInDesc),
                     DataCast(gradIn),
                     miopen::deref(expAvgInDesc),
                     DataCast(expAvgIn),
                     miopen::deref(expAvgOutDesc),
                     DataCast(expAvgOut),
                     miopen::deref(expAvgSqInDesc),
                     DataCast(expAvgSqIn),
                     miopen::deref(expAvgSqOutDesc),
                     DataCast(expAvgSqOut),
                     maxExpAvgSqInDescPtr,
                     DataCast(maxExpAvgSqIn),
                     maxExpAvgSqOutDescPtr,
                     DataCast(maxExpAvgSqOut),
                     gradScaleDescPtr,
                     DataCast(gradScale),
                     foundInfDescPtr,
                     DataCast(foundInf),
                     &miopen::deref(stepInDesc),
                     DataCast(stepIn),
                     stepOutDescPtr,
                     DataCast(stepOut),
                     -1,
                     lr,
                     beta1,
                     beta2,
                     weight_decay,
                     eps,
                     amsgrad,
                     maximize);
    });
}
