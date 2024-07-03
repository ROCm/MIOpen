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
#ifndef MIOPEN_ADAM_HPP_
#define MIOPEN_ADAM_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

MIOPEN_INTERNALS_EXPORT miopenStatus_t Adam(Handle& handle,
                                            const TensorDescriptor& paramInDesc,
                                            ConstData_t paramIn,
                                            const TensorDescriptor& paramOutDesc,
                                            Data_t paramOut,
                                            const TensorDescriptor& paramOutFloat16Desc,
                                            Data_t paramOutFloat16,
                                            const TensorDescriptor& gradInDesc,
                                            ConstData_t gradIn,
                                            const TensorDescriptor& expAvgInDesc,
                                            ConstData_t expAvgIn,
                                            const TensorDescriptor& expAvgOutDesc,
                                            Data_t expAvgOut,
                                            const TensorDescriptor& expAvgSqInDesc,
                                            ConstData_t expAvgSqIn,
                                            const TensorDescriptor& expAvgSqOutDesc,
                                            Data_t expAvgSqOut,
                                            const TensorDescriptor& maxExpAvgSqInDesc,
                                            ConstData_t maxExpAvgSqIn,
                                            const TensorDescriptor& maxExpAvgSqOutDesc,
                                            Data_t maxExpAvgSqOut,
                                            const TensorDescriptor& gradScaleDescPtr,
                                            ConstData_t gradScale,
                                            const TensorDescriptor& foundInfDescPtr,
                                            ConstData_t foundInf,
                                            const TensorDescriptor& stepInDesc,
                                            ConstData_t stepIn,
                                            const TensorDescriptor& stepOutDesc,
                                            Data_t stepOut,
                                            uint32_t step,
                                            float lr,
                                            float beta1,
                                            float beta2,
                                            float weight_decay,
                                            float eps,
                                            bool amsgrad,
                                            bool maximize,
                                            bool adamw,
                                            bool is_amp);

MIOPEN_INTERNALS_EXPORT miopenStatus_t
TransformersAdamW(Handle& handle,
                  const TensorDescriptor& paramInDesc,
                  ConstData_t paramIn,
                  const TensorDescriptor& paramOutDesc,
                  Data_t paramOut,
                  const TensorDescriptor& paramOutFloat16Desc,
                  Data_t paramOutFloat16,
                  const TensorDescriptor& gradInDesc,
                  ConstData_t gradIn,
                  const TensorDescriptor& expAvgInDesc,
                  ConstData_t expAvgIn,
                  const TensorDescriptor& expAvgOutDesc,
                  Data_t expAvgOut,
                  const TensorDescriptor& expAvgSqInDesc,
                  ConstData_t expAvgSqIn,
                  const TensorDescriptor& expAvgSqOutDesc,
                  Data_t expAvgSqOut,
                  const TensorDescriptor& gradScaleDescPtr,
                  ConstData_t gradScale,
                  const TensorDescriptor& foundInfDescPtr,
                  ConstData_t foundInf,
                  const TensorDescriptor& stepInDesc,
                  ConstData_t stepIn,
                  const TensorDescriptor& stepOutDesc,
                  Data_t stepOut,
                  uint32_t step,
                  float lr,
                  float beta1,
                  float beta2,
                  float eps,
                  float lr_weight_decay,
                  float step_size,
                  bool correct_bias,
                  bool is_amp);
} // namespace miopen
#endif // _MIOPEN_ADAM_HPP_
