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
#ifndef MIOPEN_AVGPOOL_HPP_
#define MIOPEN_AVGPOOL_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

MIOPEN_INTERNALS_EXPORT miopenStatus_t AvgPoolForward(Handle& handle,
                                                      const TensorDescriptor& inputDesc,
                                                      ConstData_t input,
                                                      const TensorDescriptor& outputDesc,
                                                      Data_t output,
                                                      int64_t KD,
                                                      int64_t KH,
                                                      int64_t KW,
                                                      int64_t SD,
                                                      int64_t SH,
                                                      int64_t SW,
                                                      int64_t PD,
                                                      int64_t PH,
                                                      int64_t PW,
                                                      bool count_include_pad,
                                                      int64_t divisor_override);

MIOPEN_INTERNALS_EXPORT miopenStatus_t AvgPoolBackward(Handle& handle,
                                                       const TensorDescriptor& outputGradDesc,
                                                       ConstData_t output_grad,
                                                       const TensorDescriptor& inputGradDesc,
                                                       Data_t input_grad,
                                                       int64_t KD,
                                                       int64_t KH,
                                                       int64_t KW,
                                                       int64_t SD,
                                                       int64_t SH,
                                                       int64_t SW,
                                                       int64_t PD,
                                                       int64_t PH,
                                                       int64_t PW,
                                                       bool count_include_pad,
                                                       int64_t divisor_override);
} // namespace miopen
#endif // _MIOPEN_AVGPOOL_HPP_
