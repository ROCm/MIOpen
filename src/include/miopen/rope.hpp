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
#ifndef MIOPEN_ROPE_HPP_
#define MIOPEN_ROPE_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

MIOPEN_INTERNALS_EXPORT miopenStatus_t RoPEForward(Handle& handle,
                                                   const TensorDescriptor& xDesc,
                                                   ConstData_t x,
                                                   const TensorDescriptor& cosDesc,
                                                   ConstData_t cos,
                                                   const TensorDescriptor& sinDesc,
                                                   ConstData_t sin,
                                                   const TensorDescriptor& yDesc,
                                                   Data_t y);

MIOPEN_INTERNALS_EXPORT miopenStatus_t RoPEBackward(Handle& handle,
                                                    const TensorDescriptor& dyDesc,
                                                    ConstData_t dy,
                                                    const TensorDescriptor& cosDesc,
                                                    ConstData_t cos,
                                                    const TensorDescriptor& sinDesc,
                                                    ConstData_t sin,
                                                    const TensorDescriptor& dxDesc,
                                                    Data_t dx);

} // namespace miopen
#endif // MIOPEN_ROPE_HPP_
