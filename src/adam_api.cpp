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
#include <miopen/adam.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdAdam(const miopenTensorDescriptor_t paramDesc,
                       const float lr,
                       const float beta1,
                       const float beta2,
                       const float weight_decay,
                       const float eps,
                       const bool amsgrad,
                       const bool maximize,
                       const bool adamw,
                       const bool is_amp)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(paramDesc).GetType();
        if(is_amp)
        {
            ss << "ampadam";
        }
        else
        {
            ss << "adam";
        }
        if(adamw)
        {
            ss << "w";
        }

        if(dtype == miopenHalf)
        {
            ss << "fp16";
        }

        std::string batch_sz;
        auto dims = miopen::deref(paramDesc).GetLengths();
        for(auto dim : dims)
        {
            batch_sz += std::to_string(dim);
            batch_sz += "x";
        }
        batch_sz.pop_back();
        ss << " -d " << batch_sz << " -l " << lr << " -1 " << beta1 << " -2 " << beta2 << " -e "
           << eps << " -W " << weight_decay << " -a " << amsgrad << " -m " << maximize;
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

#define DESC_PTR(desc) (((desc) != nullptr) ? miopen::deref(desc) : emptyDesc)

extern "C" miopenStatus_t miopenFusedAdam(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t paramDesc,
                                          void* param,
                                          const miopenTensorDescriptor_t gradDesc,
                                          const void* grad,
                                          const miopenTensorDescriptor_t expAvgDesc,
                                          void* expAvg,
                                          const miopenTensorDescriptor_t expAvgSqDesc,
                                          void* expAvgSq,
                                          const miopenTensorDescriptor_t maxExpAvgSqDesc,
                                          void* maxExpAvgSq,
                                          const miopenTensorDescriptor_t stateStepDesc,
                                          void* stateStep,
                                          const unsigned int state_step,
                                          const float lr,
                                          const float beta1,
                                          const float beta2,
                                          const float weight_decay,
                                          const float eps,
                                          const bool amsgrad,
                                          const bool maximize,
                                          const bool adamw,
                                          const miopenTensorDescriptor_t gradScaleDesc,
                                          const void* gradScale,
                                          const miopenTensorDescriptor_t foundInfDesc,
                                          const void* foundInf)
{
    MIOPEN_LOG_FUNCTION(handle,
                        paramDesc,
                        param,
                        gradDesc,
                        grad,
                        expAvgDesc,
                        expAvg,
                        expAvgSqDesc,
                        expAvgSq,
                        maxExpAvgSqDesc,
                        maxExpAvgSq,
                        stateStepDesc,
                        stateStep,
                        state_step,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        amsgrad,
                        maximize,
                        adamw,
                        gradScaleDesc,
                        gradScale,
                        foundInfDesc,
                        foundInf);

    const miopen::TensorDescriptor emptyDesc;
    bool is_amp = (foundInfDesc != nullptr || gradScaleDesc != nullptr);

    LogCmdAdam(paramDesc, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, adamw, is_amp);

    return miopen::try_([&] {
        miopen::Adam(miopen::deref(handle),
                     miopen::deref(paramDesc),
                     DataCast(param),
                     miopen::deref(paramDesc),
                     DataCast(param),
                     emptyDesc,
                     nullptr,
                     miopen::deref(gradDesc),
                     DataCast(grad),
                     miopen::deref(expAvgDesc),
                     DataCast(expAvg),
                     miopen::deref(expAvgDesc),
                     DataCast(expAvg),
                     miopen::deref(expAvgSqDesc),
                     DataCast(expAvgSq),
                     miopen::deref(expAvgSqDesc),
                     DataCast(expAvgSq),
                     DESC_PTR(maxExpAvgSqDesc),
                     DataCast(maxExpAvgSq),
                     DESC_PTR(maxExpAvgSqDesc),
                     DataCast(maxExpAvgSq),
                     DESC_PTR(gradScaleDesc),
                     DataCast(gradScale),
                     DESC_PTR(foundInfDesc),
                     DataCast(foundInf),
                     DESC_PTR(stateStepDesc),
                     DataCast(stateStep),
                     DESC_PTR(stateStepDesc),
                     DataCast(stateStep),
                     state_step,
                     lr,
                     beta1,
                     beta2,
                     weight_decay,
                     eps,
                     amsgrad,
                     maximize,
                     adamw,
                     is_amp);
    });
}

extern "C" miopenStatus_t
miopenFusedAdamWithOutput(miopenHandle_t handle,
                          const miopenTensorDescriptor_t paramInDesc,
                          void* paramIn,
                          const miopenTensorDescriptor_t paramOutDesc,
                          void* paramOut,
                          const miopenTensorDescriptor_t paramOutFloat16Desc,
                          void* paramOutFloat16,
                          const miopenTensorDescriptor_t gradInDesc,
                          const void* gradIn,
                          const miopenTensorDescriptor_t expAvgInDesc,
                          void* expAvgIn,
                          const miopenTensorDescriptor_t expAvgOutDesc,
                          void* expAvgOut,
                          const miopenTensorDescriptor_t expAvgSqInDesc,
                          void* expAvgSqIn,
                          const miopenTensorDescriptor_t expAvgSqOutDesc,
                          void* expAvgSqOut,
                          const miopenTensorDescriptor_t maxExpAvgSqInDesc,
                          void* maxExpAvgSqIn,
                          const miopenTensorDescriptor_t maxExpAvgSqOutDesc,
                          void* maxExpAvgSqOut,
                          const miopenTensorDescriptor_t stateStepInDesc,
                          void* stateStepIn,
                          const miopenTensorDescriptor_t stateStepOutDesc,
                          void* stateStepOut,
                          const unsigned int state_step,
                          const float lr,
                          const float beta1,
                          const float beta2,
                          const float weight_decay,
                          const float eps,
                          const bool amsgrad,
                          const bool maximize,
                          const bool adamw,
                          const miopenTensorDescriptor_t gradScaleDesc,
                          const void* gradScale,
                          const miopenTensorDescriptor_t foundInfDesc,
                          const void* foundInf)
{
    MIOPEN_LOG_FUNCTION(handle,
                        paramInDesc,
                        paramIn,
                        paramOutDesc,
                        paramOut,
                        gradInDesc,
                        gradIn,
                        expAvgInDesc,
                        expAvgIn,
                        expAvgOutDesc,
                        expAvgOut,
                        expAvgSqInDesc,
                        expAvgSqIn,
                        expAvgSqOutDesc,
                        expAvgSqOut,
                        maxExpAvgSqInDesc,
                        maxExpAvgSqIn,
                        maxExpAvgSqOutDesc,
                        maxExpAvgSqOut,
                        stateStepInDesc,
                        stateStepIn,
                        stateStepOutDesc,
                        stateStepOut,
                        state_step,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        amsgrad,
                        maximize,
                        adamw,
                        gradScaleDesc,
                        gradScale,
                        foundInfDesc,
                        foundInf);

    const miopen::TensorDescriptor emptyDesc;
    bool is_amp = (foundInfDesc != nullptr || gradScaleDesc != nullptr);

    LogCmdAdam(paramInDesc, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, adamw, is_amp);

    return miopen::try_([&] {
        miopen::Adam(miopen::deref(handle),
                     miopen::deref(paramInDesc),
                     DataCast(paramIn),
                     miopen::deref(paramOutDesc),
                     DataCast(paramOut),
                     DESC_PTR(paramOutFloat16Desc),
                     DataCast(paramOutFloat16),
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
                     DESC_PTR(maxExpAvgSqInDesc),
                     DataCast(maxExpAvgSqIn),
                     DESC_PTR(maxExpAvgSqOutDesc),
                     DataCast(maxExpAvgSqOut),
                     DESC_PTR(gradScaleDesc),
                     DataCast(gradScale),
                     DESC_PTR(foundInfDesc),
                     DataCast(foundInf),
                     DESC_PTR(stateStepInDesc),
                     DataCast(stateStepIn),
                     DESC_PTR(stateStepOutDesc),
                     DataCast(stateStepOut),
                     state_step,
                     lr,
                     beta1,
                     beta2,
                     weight_decay,
                     eps,
                     amsgrad,
                     maximize,
                     adamw,
                     is_amp);
    });
}
