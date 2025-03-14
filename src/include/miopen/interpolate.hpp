/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

namespace interpolate {

MIOPEN_INTERNALS_EXPORT miopenStatus_t
InterpolateNearestForward(Handle& handle,
                          const TensorDescriptor& inputDesc,
                          ConstData_t input,
                          const TensorDescriptor& outputDesc,
                          Data_t output,
                          const TensorDescriptor& scaleFactorsDesc,
                          ConstData_t scale_factors,
                          miopenInterpolateMode_t mode);

MIOPEN_INTERNALS_EXPORT miopenStatus_t
InterpolateLinearCubicForward(Handle& handle,
                              const TensorDescriptor& inputDesc,
                              ConstData_t input,
                              const TensorDescriptor& outputDesc,
                              Data_t output,
                              const TensorDescriptor& scaleFactorsDesc,
                              ConstData_t scale_factors,
                              miopenInterpolateMode_t mode,
                              bool align_corners);

MIOPEN_INTERNALS_EXPORT size_t
GetInterpolateBicubicBackwardWorkspaceSize(Handle& handle,
                                           const TensorDescriptor& outputGradDesc,
                                           const TensorDescriptor& inputGradDesc,
                                           const TensorDescriptor& scaleFactorsDesc,
                                           miopenInterpolateMode_t mode,
                                           bool align_corners);

MIOPEN_INTERNALS_EXPORT miopenStatus_t
InterpolateBicubicBackward(Handle& handle,
                           Data_t workspace,
                           size_t workspaceSizeInBytes,
                           const TensorDescriptor& inputGradDesc,
                           Data_t input_grad,
                           const TensorDescriptor& outputGradDesc,
                           ConstData_t output_grad,
                           const TensorDescriptor& scaleFactorsDesc,
                           ConstData_t scale_factors,
                           miopenInterpolateMode_t mode,
                           bool align_corners);

MIOPEN_INTERNALS_EXPORT miopenStatus_t
InterpolateNearestBackward(Handle& handle,
                           const TensorDescriptor& inputGradDesc,
                           Data_t input_grad,
                           const TensorDescriptor& outputGradDesc,
                           ConstData_t output_grad,
                           const TensorDescriptor& scaleFactorsDesc,
                           ConstData_t scale_factors,
                           miopenInterpolateMode_t mode);

MIOPEN_INTERNALS_EXPORT miopenStatus_t
InterpolateLinearBackward(Handle& handle,
                          const TensorDescriptor& inputGradDesc,
                          Data_t input_grad,
                          const TensorDescriptor& outputGradDesc,
                          ConstData_t output_grad,
                          const TensorDescriptor& scaleFactorsDesc,
                          ConstData_t scale_factors,
                          miopenInterpolateMode_t mode,
                          bool align_corners);

} // namespace interpolate

} // namespace miopen
