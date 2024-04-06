/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_GETITEM_HPP_
#define MIOPEN_GETITEM_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

std::size_t GetGetitemWorkspaceSize(Handle& handle,
                                    int32_t indexCount,
                                    const TensorDescriptor* const* indexDescs,
                                    int32_t dimCount,
                                    int32_t* dims);

miopenStatus_t GetitemBackward(Handle& handle,
                               const TensorDescriptor& dyDesc,
                               ConstData_t dy,
                               const TensorDescriptor& xDesc,
                               ConstData_t x,
                               int32_t indexCount,
                               const TensorDescriptor* const* indexDescs,
                               ConstData_t* indexs,
                               const TensorDescriptor& yDesc,
                               ConstData_t y,
                               const TensorDescriptor& dxDesc,
                               Data_t dx,
                               int32_t dimCount,
                               int32_t* dims,
                               int32_t sliceCount,
                               int32_t* slices,
                               int32_t offset);

} // namespace miopen
#endif // _MIOPEN_GETITEM_HPP_
