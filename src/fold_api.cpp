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
#include <miopen/fold.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenUnfoldForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const void* input,
                                              const miopenTensorDescriptor_t outputDesc,
                                              void* output,
                                              const int64_t* kernel_size,
                                              const int64_t kernel_size_size,
                                              const int64_t* stride,
                                              const int64_t stride_size,
                                              const int64_t* padding,
                                              const int64_t padding_size,
                                              const int64_t* dilation,
                                              const int64_t dilation_size)
{
    return miopen::try_([&] {
        miopen::fold::UnfoldForward(miopen::deref(handle),
                                    miopen::deref(inputDesc),
                                    DataCast(input),
                                    miopen::deref(outputDesc),
                                    DataCast(output),
                                    kernel_size,
                                    kernel_size_size,
                                    stride,
                                    stride_size,
                                    padding,
                                    padding_size,
                                    dilation,
                                    dilation_size);
    });
}

extern "C" miopenStatus_t miopenUnfoldBackward(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t dinputDesc,
                                               void* dinput,
                                               const miopenTensorDescriptor_t doutputDesc,
                                               const void* doutput,
                                               const int64_t* kernel_size,
                                               const int64_t kernel_size_size,
                                               const int64_t* stride,
                                               const int64_t stride_size,
                                               const int64_t* padding,
                                               const int64_t padding_size,
                                               const int64_t* dilation,
                                               const int64_t dilation_size)
{
    return miopen::try_([&] {
        miopen::fold::UnfoldBackward(miopen::deref(handle),
                                     miopen::deref(dinputDesc),
                                     DataCast(dinput),
                                     miopen::deref(doutputDesc),
                                     DataCast(doutput),
                                     kernel_size,
                                     kernel_size_size,
                                     stride,
                                     stride_size,
                                     padding,
                                     padding_size,
                                     dilation,
                                     dilation_size);
    });
}

extern "C" miopenStatus_t miopenFoldForward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t inputDesc,
                                            const void* input,
                                            const miopenTensorDescriptor_t outputDesc,
                                            void* output,
                                            const int64_t* kernel_size,
                                            const int64_t kernel_size_size,
                                            const int64_t* stride,
                                            const int64_t stride_size,
                                            const int64_t* padding,
                                            const int64_t padding_size,
                                            const int64_t* dilation,
                                            const int64_t dilation_size)
{
    return miopen::try_([&] {
        miopen::fold::FoldForward(miopen::deref(handle),
                                  miopen::deref(inputDesc),
                                  DataCast(input),
                                  miopen::deref(outputDesc),
                                  DataCast(output),
                                  kernel_size,
                                  kernel_size_size,
                                  stride,
                                  stride_size,
                                  padding,
                                  padding_size,
                                  dilation,
                                  dilation_size);
    });
}

extern "C" miopenStatus_t miopenFoldBackward(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dinputDesc,
                                             void* dinput,
                                             const miopenTensorDescriptor_t doutputDesc,
                                             const void* doutput,
                                             const int64_t* kernel_size,
                                             const int64_t kernel_size_size,
                                             const int64_t* stride,
                                             const int64_t stride_size,
                                             const int64_t* padding,
                                             const int64_t padding_size,
                                             const int64_t* dilation,
                                             const int64_t dilation_size)
{
    return miopen::try_([&] {
        miopen::fold::FoldBackward(miopen::deref(handle),
                                   miopen::deref(dinputDesc),
                                   DataCast(dinput),
                                   miopen::deref(doutputDesc),
                                   DataCast(doutput),
                                   kernel_size,
                                   kernel_size_size,
                                   stride,
                                   stride_size,
                                   padding,
                                   padding_size,
                                   dilation,
                                   dilation_size);
    });
}
