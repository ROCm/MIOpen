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

#include <miopen/nllloss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdNLLLoss(const miopenTensorDescriptor_t xDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "nlllossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "nllloss";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "nlllossbfp16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(xDesc, &size);
        ss << " -N " << miopen::deref(xDesc).GetLengths()[0];
        ss << " -C " << miopen::deref(xDesc).GetLengths()[1] << " -d "
           << miopen::deref(xDesc).GetLengths()[2] << " -D "
           << miopen::deref(xDesc).GetLengths()[3];

        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenNLLLossForward(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t inputDesc,
                                               const void* input,
                                               const miopenTensorDescriptor_t targetDesc,
                                               const void* target,
                                               const miopenTensorDescriptor_t weightDesc,
                                               const void* weight,
                                               const miopenTensorDescriptor_t outputDesc,
                                               void* output,
                                               int ignore_index)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        weightDesc,
                        weight,
                        outputDesc,
                        output,
                        ignore_index);

    LogCmdNLLLoss(inputDesc, true);
    return miopen::try_([&] {
        miopen::NLLLossForward(miopen::deref(handle),
                               miopen::deref(inputDesc),
                               DataCast(input),
                               miopen::deref(targetDesc),
                               DataCast(target),
                               miopen::deref(weightDesc),
                               DataCast(weight),
                               miopen::deref(outputDesc),
                               DataCast(output),
                               ignore_index);
    });
}
