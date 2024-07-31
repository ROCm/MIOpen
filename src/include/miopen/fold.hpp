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
#ifndef MIOPEN_FOLD_HPP_
#define MIOPEN_FOLD_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

MIOPEN_INTERNALS_EXPORT miopenStatus_t UnfoldForward(Handle& handle,
                             const TensorDescriptor& inputDesc,
                             ConstData_t input,
                             const TensorDescriptor& outputDesc,
                             Data_t output,
                             const int32_t* kernel_size,
                             int32_t kernel_size_size,
                             const int32_t* stride,
                             int32_t stride_size,
                             const int32_t* padding,
                             int32_t padding_size,
                             const int32_t* dilation,
                             int32_t dilation_size);

MIOPEN_INTERNALS_EXPORT miopenStatus_t UnfoldBackward(Handle& handle,
                              const TensorDescriptor& dinputDesc,
                              Data_t dinput,
                              const TensorDescriptor& doutputDesc,
                              ConstData_t doutput,
                              const int32_t* kernel_size,
                              int32_t kernel_size_size,
                              const int32_t* stride,
                              int32_t stride_size,
                              const int32_t* padding,
                              int32_t padding_size,
                              const int32_t* dilation,
                              int32_t dilation_size);

MIOPEN_INTERNALS_EXPORT miopenStatus_t FoldForward(Handle& handle,
                           const TensorDescriptor& inputDesc,
                           ConstData_t input,
                           const TensorDescriptor& outputDesc,
                           Data_t output,
                           const int32_t* kernel_size,
                           int32_t kernel_size_size,
                           const int32_t* stride,
                           int32_t stride_size,
                           const int32_t* padding,
                           int32_t padding_size,
                           const int32_t* dilation,
                           int32_t dilation_size);

MIOPEN_INTERNALS_EXPORT miopenStatus_t FoldBackward(Handle& handle,
                            const TensorDescriptor& dinputDesc,
                            Data_t dinput,
                            const TensorDescriptor& doutputDesc,
                            ConstData_t doutput,
                            const int32_t* kernel_size,
                            int32_t kernel_size_size,
                            const int32_t* stride,
                            int32_t stride_size,
                            const int32_t* padding,
                            int32_t padding_size,
                            const int32_t* dilation,
                            int32_t dilation_size);
} // namespace miopen
#endif // MIOPEN_FOLD_HPP_
