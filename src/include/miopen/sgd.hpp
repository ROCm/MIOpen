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
#pragma once

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

namespace SGD {

MIOPEN_INTERNALS_EXPORT miopenStatus_t SGDForward(Handle& handle,
                                                  const TensorDescriptor& paramInDesc,
                                                  ConstData_t paramIn,
                                                  const TensorDescriptor& paramOutDesc,
                                                  Data_t paramOut,
                                                  const TensorDescriptor& gradDesc,
                                                  ConstData_t grad,
                                                  const TensorDescriptor& momentumBufferInDesc,
                                                  ConstData_t momentumBufferIn,
                                                  const TensorDescriptor& momentumBufferOutDesc,
                                                  Data_t momentumBufferOut,
                                                  double lr,
                                                  double momentum,
                                                  double dampening,
                                                  double weightDecay,
                                                  bool nesterov,
                                                  bool momentum_initialized);

} // namespace SGD

} // namespace miopen
