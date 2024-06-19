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

#include "miopen/tensor.hpp"
#include <miopen/where.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

#define CHECK_DESC_EXIST(desc) (((desc) != nullptr) ? miopen::deref(desc) : dummyDesc)

extern "C" miopenStatus_t miopenWhereBackward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t outputGradDesc,
                                              void* outputGrad,
                                              const miopenTensorDescriptor_t conditionDesc,
                                              void* condition,
                                              const miopenTensorDescriptor_t inputGradDesc,
                                              void* inputGrad,
                                              const miopenTensorDescriptor_t otherGradDesc,
                                              void* otherGrad)
{
    MIOPEN_LOG_FUNCTION(handle,
                        outputGradDesc,
                        outputGrad,
                        conditionDesc,
                        condition,
                        inputGradDesc,
                        inputGrad,
                        otherGradDesc,
                        otherGrad);
    const miopen::TensorDescriptor dummyDesc;

    return miopen::try_([&] {
        miopen::WhereBackward(miopen::deref(handle),
                              miopen::deref(outputGradDesc),
                              DataCast(outputGrad),
                              miopen::deref(conditionDesc),
                              DataCast(condition),
                              CHECK_DESC_EXIST(inputGradDesc),
                              DataCast(inputGrad),
                              CHECK_DESC_EXIST(otherGradDesc),
                              DataCast(otherGrad));
    });
}
