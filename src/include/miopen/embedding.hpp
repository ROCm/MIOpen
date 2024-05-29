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
#ifndef MIOPEN_EMBEDDING_HPP_
#define MIOPEN_EMBEDDING_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

miopenStatus_t EmbeddingForward(Handle& handle,
                                const TensorDescriptor& inputDesc,
                                ConstData_t input,
                                const TensorDescriptor& outputDesc,
                                Data_t output,
                                const TensorDescriptor& weightDesc,
                                Data_t weight,
                                const TensorDescriptor& errorDesc,
                                Data_t error,
                                bool has_max_norm,
                                float max_norm,
                                float norm_type);

miopenStatus_t EmbeddingBackward(Handle& handle,
                                 Data_t workspace,
                                 size_t workspaceSizeInBytes,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& outputGradDesc,
                                 ConstData_t outputGrad,
                                 const TensorDescriptor& weightGradDesc,
                                 Data_t weightGrad,
                                 const TensorDescriptor& errorDesc,
                                 Data_t error,
                                 long long padding_idx,
                                 bool scale_grad_by_freq,
                                 bool deterministic_mode);
} // namespace miopen
#endif // _MIOPEN_EMBEDDING_HPP_
