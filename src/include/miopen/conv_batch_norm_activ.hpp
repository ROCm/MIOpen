/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_CONV_BATCHNORMALIZATION_ACTIV_HPP_
#define GUARD_MIOPEN_CONV_BATCHNORMALIZATION_ACTIV_HPP_

#include <chrono>
#include <cmath>
#include <miopen/common.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

#define MIO_BN_CPP_PROF 0
#define MIOPEN_BN_CPP_DEBUG 0
#define MIO_BN_STATIC_WGSIZE 256
#define MIO_BN_TIME_EVERYTHING 0

namespace miopen {

void DirectConvBNActivInference(Handle& handle,
                                const void* alpha,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& wDesc,
                                ConstData_t w,
                                const void* beta,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                int pad_h,
                                int pad_w,
                                int u,
                                int v,
                                int dilation_h,
                                int dilation_w,
                                int bias_mode,
                                ConstData_t bias,
                                miopenBatchNormMode_t bn_mode,
                                ConstData_t bnScale,
                                ConstData_t bnBias,
                                ConstData_t estimatedMean,
                                ConstData_t estimatedVariance,
                                double epsilon,
                                miopenActivationMode_t activ_mode,
                                double activ_alpha,
                                double activ_beta,
                                double activ_gama);
}

#endif // GUARD_MIOPEN_BATCHNORMALIZATION_HPP_
