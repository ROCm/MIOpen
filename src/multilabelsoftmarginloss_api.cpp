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
#include "miopen/miopen.h"
#include <miopen/multilabelsoftmarginloss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t
miopenGetMultilabelSoftMarginLossForwardWorkspaceSize(miopenHandle_t handle,
                                                      const miopenTensorDescriptor_t inputDesc,
                                                      const miopenTensorDescriptor_t targetDesc,
                                                      const miopenTensorDescriptor_t weightDesc,
                                                      const miopenTensorDescriptor_t outputDesc,
                                                      const miopenLossReductionMode_t reduction,
                                                      size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, targetDesc, weightDesc, outputDesc, reduction);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetMultilabelSoftMarginLossForwardWorkspaceSize(miopen::deref(handle),
                                                                    miopen::deref(inputDesc),
                                                                    miopen::deref(targetDesc),
                                                                    miopen::deref(weightDesc),
                                                                    miopen::deref(outputDesc),
                                                                    reduction);
    });
}

extern "C" miopenStatus_t
miopenMultilabelSoftMarginLossForward(miopenHandle_t handle,
                                      const miopenTensorDescriptor_t inputDesc,
                                      const void* input,
                                      const miopenTensorDescriptor_t targetDesc,
                                      const void* target,
                                      const miopenTensorDescriptor_t weightDesc,
                                      const void* weight,
                                      const miopenTensorDescriptor_t outputDesc,
                                      void* output,
                                      const miopenLossReductionMode_t reduction,
                                      void* workspace,
                                      const size_t workspaceSizeInBytes)
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
                        reduction);

    return miopen::try_([&] {
        miopen::MultilabelSoftMarginLossForward(miopen::deref(handle),
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
                                                reduction);
    });
}
