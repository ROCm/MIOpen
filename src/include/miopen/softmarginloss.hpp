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
#ifndef MIOPEN_SOFTMARGINLOSS_HPP_
#define MIOPEN_SOFTMARGINLOSS_HPP_

#include "miopen/miopen.h"
#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

MIOPEN_INTERNALS_EXPORT std::size_t
GetSoftMarginLossForwardWorkspaceSize(Handle& handle,
                                      const TensorDescriptor& iDesc,
                                      const TensorDescriptor& tDesc,
                                      const TensorDescriptor& oDesc,
                                      miopenLossReductionMode_t reduction);

MIOPEN_INTERNALS_EXPORT miopenStatus_t SoftMarginLossForward(Handle& handle,
                                                             Data_t workspace,
                                                             size_t workspaceSizeInBytes,
                                                             const TensorDescriptor& iDesc,
                                                             ConstData_t i,
                                                             const TensorDescriptor& tDesc,
                                                             ConstData_t t,
                                                             const TensorDescriptor& oDesc,
                                                             Data_t o,
                                                             miopenLossReductionMode_t reduction);

MIOPEN_INTERNALS_EXPORT miopenStatus_t SoftMarginLossBackward(Handle& handle,
                                                              const TensorDescriptor& iDesc,
                                                              ConstData_t i,
                                                              const TensorDescriptor& tDesc,
                                                              ConstData_t t,
                                                              const TensorDescriptor& dODesc,
                                                              ConstData_t dO,
                                                              const TensorDescriptor& dIDesc,
                                                              Data_t dI,
                                                              miopenLossReductionMode_t reduction);

} // namespace miopen
#endif // _MIOPEN_SOFTMARGINLOSS_HPP_
