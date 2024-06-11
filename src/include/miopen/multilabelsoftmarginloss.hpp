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
#ifndef MIOPEN_MULTILABELSOFTMARGINLOSS_HPP_
#define MIOPEN_MULTILABELSOFTMARGINLOSS_HPP_

#include "miopen/miopen.h"
#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

std::size_t GetMultilabelSoftMarginLossForwardWorkspaceSize(Handle& handle,
                                                            const TensorDescriptor& iDesc,
                                                            const TensorDescriptor& tDesc,
                                                            const TensorDescriptor& wDesc,
                                                            const TensorDescriptor& oDesc,
                                                            miopenLossReductionMode_t reduction);

miopenStatus_t MultilabelSoftMarginLossUnreducedForward(Handle& handle,
                                                        const TensorDescriptor& iDesc,
                                                        ConstData_t i,
                                                        const TensorDescriptor& tDesc,
                                                        ConstData_t t,
                                                        const TensorDescriptor& wDesc,
                                                        ConstData_t w,
                                                        const TensorDescriptor& oDesc,
                                                        Data_t o);

miopenStatus_t MultilabelSoftMarginLossForward(Handle& handle,
                                               Data_t workspace,
                                               size_t workspaceSizeInBytes,
                                               const TensorDescriptor& iDesc,
                                               ConstData_t i,
                                               const TensorDescriptor& tDesc,
                                               ConstData_t t,
                                               const TensorDescriptor& wDesc,
                                               ConstData_t w,
                                               const TensorDescriptor& oDesc,
                                               Data_t o,
                                               miopenLossReductionMode_t reduction);

miopenStatus_t MultilabelSoftMarginLossUnreducedBackward(Handle& handle,
                                                         const TensorDescriptor& iDesc,
                                                         ConstData_t i,
                                                         const TensorDescriptor& tDesc,
                                                         ConstData_t t,
                                                         const TensorDescriptor& wDesc,
                                                         ConstData_t w,
                                                         const TensorDescriptor& dODesc,
                                                         ConstData_t dO,
                                                         const TensorDescriptor& dIDesc,
                                                         Data_t dI,
                                                         const TensorDescriptor& dWDesc,
                                                         Data_t dW);

miopenStatus_t MultilabelSoftMarginLossBackward(Handle& handle,
                                                const TensorDescriptor& iDesc,
                                                ConstData_t i,
                                                const TensorDescriptor& tDesc,
                                                ConstData_t t,
                                                const TensorDescriptor& wDesc,
                                                ConstData_t w,
                                                const TensorDescriptor& dODesc,
                                                ConstData_t dO,
                                                const TensorDescriptor& dIDesc,
                                                Data_t dI,
                                                const TensorDescriptor& dWDesc,
                                                Data_t dW,
                                                miopenLossReductionMode_t reduction);

} // namespace miopen
#endif // _MIOPEN_MULTILABELSOFTMARGINLOSS_HPP_
