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

#include <miopen/miopen.h>
#include <miopen/sigmoid_focal_loss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

static void LogCmdSigmoidFocalLoss(const miopenTensorDescriptor_t inputDesc,
                                   const miopenTensorDescriptor_t targetDesc,
                                   bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "sigmoidFocalLossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "sigmoidFocalLossfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "sigmoidFocalLossbfp16";
        }

        MIOPEN_LOG_FUNCTION(inputDesc, targetDesc);
        ss << " -n " << miopen::deref(inputDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(inputDesc).GetLengths();
        ss << " -Si " << miopen::deref(inputDesc).GetStrides();
        ss << " -St " << miopen::deref(targetDesc).GetStrides();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetSigmoidFocalLossForwardWorkspaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const miopenTensorDescriptor_t targetDesc,
                                              const miopenTensorDescriptor_t outputDesc,
                                              miopenLossReductionMode_t reduction,
                                              size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, inputDesc, targetDesc, outputDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetSigmoidFocalLossForwardWorkspaceSize(miopen::deref(handle),
                                                            miopen::deref(inputDesc),
                                                            miopen::deref(targetDesc),
                                                            miopen::deref(outputDesc),
                                                            reduction);
    });
}

extern "C" miopenStatus_t miopenSigmoidFocalLossForward(miopenHandle_t handle,
                                                        void* workspace,
                                                        size_t workspaceSizeInBytes,
                                                        const miopenTensorDescriptor_t inputDesc,
                                                        const void* input,
                                                        const miopenTensorDescriptor_t targetDesc,
                                                        const void* target,
                                                        const miopenTensorDescriptor_t outputDesc,
                                                        void* output,
                                                        const float alpha,
                                                        const float gamma,
                                                        const miopenLossReductionMode_t reduction)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputDesc,
                        output,
                        alpha,
                        gamma,
                        reduction);

    LogCmdSigmoidFocalLoss(inputDesc, targetDesc, true);

    return miopen::try_([&] {
        miopen::SigmoidFocalLossForward(miopen::deref(handle),
                                        DataCast(workspace),
                                        workspaceSizeInBytes,
                                        miopen::deref(inputDesc),
                                        DataCast(input),
                                        miopen::deref(targetDesc),
                                        DataCast(target),
                                        miopen::deref(outputDesc),
                                        DataCast(output),
                                        alpha,
                                        gamma,
                                        reduction);
    });
}

extern "C" miopenStatus_t miopenSigmoidFocalLossBackward(miopenHandle_t handle,
                                                         miopenTensorDescriptor_t inputDesc,
                                                         const void* input,
                                                         miopenTensorDescriptor_t targetDesc,
                                                         const void* target,
                                                         miopenTensorDescriptor_t doutputDesc,
                                                         const void* doutput,
                                                         miopenTensorDescriptor_t dinputDesc,
                                                         void* dinput,
                                                         miopenTensorDescriptor_t dtargetDesc,
                                                         void* dtarget,
                                                         float alpha,
                                                         float gamma,
                                                         const miopenLossReductionMode_t reduction)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        doutputDesc,
                        doutput,
                        dinputDesc,
                        dinput,
                        dtargetDesc,
                        dtarget,
                        alpha,
                        gamma,
                        reduction);

    LogCmdSigmoidFocalLoss(inputDesc, targetDesc, false);

    return miopen::try_([&] {
        miopen::SigmoidFocalLossBackward(miopen::deref(handle),
                                         miopen::deref(inputDesc),
                                         DataCast(input),
                                         miopen::deref(targetDesc),
                                         DataCast(target),
                                         miopen::deref(doutputDesc),
                                         DataCast(doutput),
                                         miopen::deref(dinputDesc),
                                         DataCast(dinput),
                                         miopen::deref(dtargetDesc),
                                         DataCast(dtarget),
                                         alpha,
                                         gamma,
                                         reduction);
    });
}
