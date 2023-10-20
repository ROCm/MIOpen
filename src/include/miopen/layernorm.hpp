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
#include <miopen/miopen.h>
#ifdef MIOPEN_BETA_API
#ifndef MIOPEN_LAYERNORM_HPP_
#define MIOPEN_LAYERNORM_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

miopenStatus_t LayerNormForward(const Handle& handle,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& weightDesc,
                                ConstData_t weight,
                                const TensorDescriptor& biasDesc,
                                ConstData_t bias,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                const TensorDescriptor& meanDesc,
                                Data_t mean,
                                const TensorDescriptor& rstdDesc,
                                Data_t rstd,
                                miopenLayerNormMode_t mode,
                                float epsilon,
                                int32_t normalized_dim);

} // namespace miopen
#endif // _MIOPEN_LAYERNORM_HPP_
#endif
