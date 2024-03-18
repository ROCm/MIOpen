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
                                     const miopenTensorDescriptor_t paramDesc,
                                     void* param,
                                     const miopenTensorDescriptor_t gradDesc,
                                     const void* grad,
                                     const miopenTensorDescriptor_t expAvgDesc,
                                     void* expAvg,
                                     const miopenTensorDescriptor_t expAvgSqDesc,
                                     void* expAvgSq,
                                     const miopenTensorDescriptor_t stepDesc,
                                     void* step,
                                     const double lr,
                                     const double beta1,
                                     const double beta2,
                                     const double weight_decay,
                                     const double eps,
                                     const bool amsgrad,
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
                        stepDesc,
                        step,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        eps,
                        amsgrad,
                        gradScaleDesc,
                        gradScale,
                        foundInfDesc,
                        foundInf);
    LogCmdAdam(paramDesc);
    auto gradScaleDescPtr = (gradScaleDesc != nullptr) ? &miopen::deref(gradScaleDesc) : nullptr;
    auto foundInfDescPtr  = (foundInfDesc != nullptr) ? &miopen::deref(foundInfDesc) : nullptr;
    return miopen::try_([&] {
        miopen::Adam(miopen::deref(handle),
                     miopen::deref(paramDesc),
                     DataCast(param),
                     miopen::deref(gradDesc),
                     DataCast(grad),
                     miopen::deref(expAvgDesc),
                     DataCast(expAvg),
                     miopen::deref(expAvgSqDesc),
                     DataCast(expAvgSq),
                     miopen::deref(stepDesc),
                     DataCast(step),
                     lr,
                     beta1,
                     beta2,
                     weight_decay,
                     eps,
                     amsgrad,
                     gradScaleDescPtr,
                     DataCast(gradScale),
                     foundInfDescPtr,
                     DataCast(foundInf));
    });
}
