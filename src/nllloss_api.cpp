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

static void LogCmdNLLLoss(const miopenTensorDescriptor_t xDesc,
                          const miopenTensorDescriptor_t tDesc,
                          const miopenTensorDescriptor_t wDesc,
                          bool is_fwd)
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
            ss << "nlllossfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "nlllossbfp16";
        }

        MIOPEN_LOG_FUNCTION(xDesc, tDesc, wDesc);
        ss << " -N " << miopen::deref(xDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(xDesc).GetLengths();
        ss << " -Si " << miopen::deref(xDesc).GetStrides();
        ss << " -St " << miopen::deref(tDesc).GetStrides();
        ss << " -Sw " << miopen::deref(wDesc).GetStrides();

        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenNLLLossUnreduceForward(miopenHandle_t handle,
                                                       const miopenTensorDescriptor_t inputDesc,
                                                       const void* input,
                                                       const miopenTensorDescriptor_t targetDesc,
                                                       const void* target,
                                                       const miopenTensorDescriptor_t weightDesc,
                                                       const void* weight,
                                                       const miopenTensorDescriptor_t outputDesc,
                                                       void* output,
                                                       int32_t ignore_index)
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

    LogCmdNLLLoss(inputDesc, targetDesc, weightDesc, true);
    return miopen::try_([&] {
        miopen::NLLLossUnreduceForward(miopen::deref(handle),
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

extern "C" miopenStatus_t
miopenGetNLLLossReduceForwardWorkspaceSize(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t inputDesc,
                                           const miopenTensorDescriptor_t targetDesc,
                                           const miopenTensorDescriptor_t weightDesc,
                                           const miopenTensorDescriptor_t outputDesc,
                                           size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, inputDesc, targetDesc, weightDesc, outputDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetNLLLossReduceForwardWorkspaceSize(miopen::deref(handle),
                                                         miopen::deref(inputDesc),
                                                         miopen::deref(targetDesc),
                                                         miopen::deref(weightDesc),
                                                         miopen::deref(outputDesc));
    });
}

extern "C" miopenStatus_t miopenNLLLossReduceForward(miopenHandle_t handle,
                                                     void* workspace,
                                                     size_t workspaceSizeInBytes,
                                                     const miopenTensorDescriptor_t inputDesc,
                                                     const void* input,
                                                     const miopenTensorDescriptor_t targetDesc,
                                                     const void* target,
                                                     const miopenTensorDescriptor_t weightDesc,
                                                     const void* weight,
                                                     const miopenTensorDescriptor_t outputDesc,
                                                     void* output,
                                                     const int32_t ignore_index,
                                                     const float divisor)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        weightDesc,
                        weight,
                        outputDesc,
                        output,
                        ignore_index,
                        divisor);

    LogCmdNLLLoss(inputDesc, targetDesc, weightDesc, true);
    return miopen::try_([&] {
        miopen::NLLLossReduceForward(miopen::deref(handle),
                                     DataCast(workspace),
                                     workspaceSizeInBytes,
                                     miopen::deref(inputDesc),
                                     DataCast(input),
                                     miopen::deref(targetDesc),
                                     DataCast(target),
                                     miopen::deref(weightDesc),
                                     DataCast(weight),
                                     miopen::deref(outputDesc),
                                     DataCast(output),
                                     ignore_index,
                                     divisor);
    });
}

extern "C" miopenStatus_t
miopenNLLLossUnreduceBackward(miopenHandle_t handle,
                              const miopenTensorDescriptor_t inputGradDesc,
                              void* input_grad,
                              const miopenTensorDescriptor_t targetDesc,
                              const void* target,
                              const miopenTensorDescriptor_t weightDesc,
                              const void* weight,
                              const miopenTensorDescriptor_t outputGradDesc,
                              void* output_grad,
                              int32_t ignore_index)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputGradDesc,
                        input_grad,
                        targetDesc,
                        target,
                        weightDesc,
                        weight,
                        outputGradDesc,
                        output_grad,
                        ignore_index);

    LogCmdNLLLoss(inputGradDesc, targetDesc, weightDesc, false);
    return miopen::try_([&] {
        miopen::NLLLossUnreduceBackward(miopen::deref(handle),
                                        miopen::deref(inputGradDesc),
                                        DataCast(input_grad),
                                        miopen::deref(targetDesc),
                                        DataCast(target),
                                        miopen::deref(weightDesc),
                                        DataCast(weight),
                                        miopen::deref(outputGradDesc),
                                        DataCast(output_grad),
                                        ignore_index);
    });
}

extern "C" miopenStatus_t miopenNLLLossReduceBackward(miopenHandle_t handle,
                                                      const miopenTensorDescriptor_t inputGradDesc,
                                                      void* input_grad,
                                                      const miopenTensorDescriptor_t targetDesc,
                                                      const void* target,
                                                      const miopenTensorDescriptor_t weightDesc,
                                                      const void* weight,
                                                      const miopenTensorDescriptor_t outputGradDesc,
                                                      void* output_grad,
                                                      const int32_t ignore_index,
                                                      const float divisor)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputGradDesc,
                        input_grad,
                        targetDesc,
                        target,
                        weightDesc,
                        weight,
                        outputGradDesc,
                        output_grad,
                        ignore_index,
                        divisor);

    LogCmdNLLLoss(inputGradDesc, targetDesc, weightDesc, false);
    return miopen::try_([&] {
        miopen::NLLLossReduceBackward(miopen::deref(handle),
                                      miopen::deref(inputGradDesc),
                                      DataCast(input_grad),
                                      miopen::deref(targetDesc),
                                      DataCast(target),
                                      miopen::deref(weightDesc),
                                      DataCast(weight),
                                      miopen::deref(outputGradDesc),
                                      DataCast(output_grad),
                                      ignore_index,
                                      divisor);
    });
}
