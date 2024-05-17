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

#include <miopen/kldivloss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdKLDivLoss(const miopenTensorDescriptor_t xDesc, bool is_fwd)
{
    // if(miopen::IsLoggingCmd())
    // {
    //     std::stringstream ss;
    //     auto dtype = miopen::deref(xDesc).GetType();
    //     if(dtype == miopenHalf)
    //     {
    //         ss << "nlllossfp16";
    //     }
    //     else if(dtype == miopenFloat)
    //     {
    //         ss << "nllloss";
    //     }
    //     else if(dtype == miopenBFloat16)
    //     {
    //         ss << "nlllossbfp16";
    //     }

    //     int32_t size = {0};
    //     miopenGetTensorDescriptorSize(xDesc, &size);
    //     ss << " -N " << miopen::deref(xDesc).GetLengths()[0];
    //     ss << " -C " << miopen::deref(xDesc).GetLengths()[1] << " -d "
    //        << miopen::deref(xDesc).GetLengths()[2] << " -D "
    //        << miopen::deref(xDesc).GetLengths()[3];

    //     ss << " -F " << ((is_fwd) ? "1" : "2");

    //     MIOPEN_LOG_DRIVER_CMD(ss.str());
    // }
}

extern "C" miopenStatus_t miopenKLDivLossUnreducedForward(miopenHandle_t handle,
                                                          const miopenTensorDescriptor_t inputDesc,
                                                          const void* input,
                                                          const miopenTensorDescriptor_t targetDesc,
                                                          const void* target,
                                                          const miopenTensorDescriptor_t outputDesc,
                                                          void* output,
                                                          bool log_target)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputDesc, input, targetDesc, target, outputDesc, output, log_target);

    LogCmdKLDivLoss(inputDesc, true);
    return miopen::try_([&] {
        miopen::KLDivLossUnreducedForward(miopen::deref(handle),
                                          miopen::deref(inputDesc),
                                          DataCast(input),
                                          miopen::deref(targetDesc),
                                          DataCast(target),
                                          miopen::deref(outputDesc),
                                          DataCast(output),
                                          log_target);
    });
}

// extern "C" miopenStatus_t
// miopenGetNLLLossReducedForwardWorkspaceSize(miopenHandle_t handle,
//                                            const miopenTensorDescriptor_t inputDesc,
//                                            const miopenTensorDescriptor_t targetDesc,
//                                            const miopenTensorDescriptor_t weightDesc,
//                                            const miopenTensorDescriptor_t outputDesc,
//                                            size_t* sizeInBytes)
// {

//     MIOPEN_LOG_FUNCTION(handle, inputDesc, targetDesc, weightDesc, outputDesc, sizeInBytes);

//     return miopen::try_([&] {
//         miopen::deref(sizeInBytes) =
//             miopen::GetNLLLossReducedForwardWorkspaceSize(miopen::deref(handle),
//                                                          miopen::deref(inputDesc),
//                                                          miopen::deref(targetDesc),
//                                                          miopen::deref(weightDesc),
//                                                          miopen::deref(outputDesc));
//     });
// }

// extern "C" miopenStatus_t miopenNLLLossReducedForward(miopenHandle_t handle,
//                                                      void* workspace,
//                                                      size_t workspaceSizeInBytes,
//                                                      const miopenTensorDescriptor_t inputDesc,
//                                                      const void* input,
//                                                      const miopenTensorDescriptor_t targetDesc,
//                                                      const void* target,
//                                                      const miopenTensorDescriptor_t weightDesc,
//                                                      const void* weight,
//                                                      const miopenTensorDescriptor_t outputDesc,
//                                                      void* output,
//                                                      const int32_t ignore_index,
//                                                      const float divisor)
// {
//     MIOPEN_LOG_FUNCTION(handle,
//                         workspace,
//                         workspaceSizeInBytes,
//                         inputDesc,
//                         input,
//                         targetDesc,
//                         target,
//                         weightDesc,
//                         weight,
//                         outputDesc,
//                         output,
//                         ignore_index,
//                         divisor);

//     LogCmdNLLLoss(inputDesc, true);
//     return miopen::try_([&] {
//         miopen::NLLLossReducedForward(miopen::deref(handle),
//                                      DataCast(workspace),
//                                      workspaceSizeInBytes,
//                                      miopen::deref(inputDesc),
//                                      DataCast(input),
//                                      miopen::deref(targetDesc),
//                                      DataCast(target),
//                                      miopen::deref(weightDesc),
//                                      DataCast(weight),
//                                      miopen::deref(outputDesc),
//                                      DataCast(output),
//                                      ignore_index,
//                                      divisor);
//     });
// }

extern "C" miopenStatus_t
miopenKLDivLossUnreducedBackward(miopenHandle_t handle,
                                 const miopenTensorDescriptor_t inputDesc,
                                 const void* input,
                                 const miopenTensorDescriptor_t targetDesc,
                                 const void* target,
                                 const miopenTensorDescriptor_t outputGradDesc,
                                 const void* output_grad,
                                 const miopenTensorDescriptor_t inputGradDesc,
                                 void* input_grad,
                                 const miopenTensorDescriptor_t targetGradDesc,
                                 void* target_grad,
                                 bool log_target)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputGradDesc,
                        output_grad,
                        inputGradDesc,
                        input_grad,
                        targetGradDesc,
                        target_grad,
                        log_target);

    LogCmdKLDivLoss(inputGradDesc, false);
    return miopen::try_([&] {
        miopen::KLDivLossUnreducedBackward(miopen::deref(handle),
                                           miopen::deref(inputDesc),
                                           DataCast(input),
                                           miopen::deref(targetDesc),
                                           DataCast(target),
                                           miopen::deref(outputGradDesc),
                                           DataCast(output_grad),
                                           miopen::deref(inputGradDesc),
                                           DataCast(input_grad),
                                           miopen::deref(targetGradDesc),
                                           DataCast(target_grad),
                                           log_target);
    });
}

extern "C" miopenStatus_t
miopenKLDivLossReducedBackward(miopenHandle_t handle,
                                 const miopenTensorDescriptor_t inputDesc,
                                 const void* input,
                                 const miopenTensorDescriptor_t targetDesc,
                                 const void* target,
                                 const miopenTensorDescriptor_t outputGradDesc,
                                 const void* output_grad,
                                 const miopenTensorDescriptor_t inputGradDesc,
                                 void* input_grad,
                                 const miopenTensorDescriptor_t targetGradDesc,
                                 void* target_grad,
                                 float divisor,
                                 bool log_target)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputGradDesc,
                        output_grad,
                        inputGradDesc,
                        input_grad,
                        targetGradDesc,
                        target_grad,
                        divisor,
                        log_target);

    LogCmdKLDivLoss(inputGradDesc, false);
    return miopen::try_([&] {
        miopen::KLDivLossReducedBackward(miopen::deref(handle),
                                           miopen::deref(inputDesc),
                                           DataCast(input),
                                           miopen::deref(targetDesc),
                                           DataCast(target),
                                           miopen::deref(outputGradDesc),
                                           DataCast(output_grad),
                                           miopen::deref(inputGradDesc),
                                           DataCast(input_grad),
                                           miopen::deref(targetGradDesc),
                                           DataCast(target_grad),
                                           divisor,
                                           log_target);
    });
}
