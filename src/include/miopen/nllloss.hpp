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
#ifndef MIOPEN_NLLLOSS_HPP_
#define MIOPEN_NLLLOSS_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

miopenStatus_t NLLLossUnreduceForward(Handle& handle,
                                      const TensorDescriptor& inputDesc,
                                      ConstData_t input,
                                      const TensorDescriptor& targetDesc,
                                      ConstData_t target,
                                      const TensorDescriptor& weightDesc,
                                      ConstData_t weight,
                                      const TensorDescriptor& outputDesc,
                                      Data_t output,
                                      int32_t ignore_index);

size_t GetNLLLossReduceForwardWorkspaceSize(Handle& handle,
                                            const TensorDescriptor& inputDesc,
                                            const TensorDescriptor& targetDesc,
                                            const TensorDescriptor& weightDesc,
                                            const TensorDescriptor& outputDesc);

miopenStatus_t NLLLossReduceForward(Handle& handle,
                                    Data_t workspace,
                                    size_t workspaceSizeInBytes,
                                    const TensorDescriptor& inputDesc,
                                    ConstData_t input,
                                    const TensorDescriptor& targetDesc,
                                    ConstData_t target,
                                    const TensorDescriptor& weightDesc,
                                    ConstData_t weight,
                                    const TensorDescriptor& outputDesc,
                                    Data_t output,
                                    int32_t ignore_index,
                                    float divisor);

miopenStatus_t NLLLossUnreduceBackward(Handle& handle,
                                       const TensorDescriptor& inputGradDesc,
                                       Data_t input_grad,
                                       const TensorDescriptor& targetDesc,
                                       ConstData_t target,
                                       const TensorDescriptor& weightDesc,
                                       ConstData_t weight,
                                       const TensorDescriptor& outputGradDesc,
                                       Data_t output_grad,
                                       int32_t ignore_index);

miopenStatus_t NLLLossReduceBackward(Handle& handle,
                                       const TensorDescriptor& inputGradDesc,
                                       Data_t input_grad,
                                       const TensorDescriptor& targetDesc,
                                       ConstData_t target,
                                       const TensorDescriptor& weightDesc,
                                       ConstData_t weight,
                                       const TensorDescriptor& outputGradDesc,
                                       Data_t output_grad,
                                       int32_t ignore_index,
                                       float divisor);
} // namespace miopen
#endif // _MIOPEN_NLLLOSS_HPP_
