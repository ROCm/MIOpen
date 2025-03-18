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
#pragma once

#include <miopen/miopen.h>

#include <miopen/convolution.hpp>
#include <miopen/errors.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

#include <algorithm>

namespace miopen {
namespace debug {

int GetFusionMode(const miopenFusionPlanDescriptor_t& fusePlanDesc);

enum class ConvDirection
{
    Fwd = 1,
    Bwd = 2,
    WrW = 4
};

enum BatchNormDirection_t
{
    ForwardInference,
    ForwardTraining,
    Backward
};

miopenProblemDirection_t CmdArgToDirection(ConvDirection direction);

std::string ConvArgsForMIOpenDriver(const miopen::TensorDescriptor& xDesc,
                                    const miopen::TensorDescriptor& wDesc,
                                    const miopen::ConvolutionDescriptor& convDesc,
                                    const miopen::TensorDescriptor& yDesc,
                                    const miopenProblemDirection_t& conv_dir,
                                    std::optional<uint64_t> immediate_mode_solver_id,
                                    bool print_for_conv_driver = true);

std::string BnormArgsForMIOpenDriver(miopenTensorDescriptor_t xDesc,
                                     miopenTensorDescriptor_t yDesc,
                                     miopenTensorDescriptor_t scaleDesc,
                                     miopenTensorDescriptor_t biasDesc,
                                     miopenTensorDescriptor_t saveMeanDesc,
                                     miopenBatchNormMode_t bn_mode,
                                     const void* resultRunningMean,
                                     const void* resultRunningVariance,
                                     const void* resultSaveMean,
                                     const void* resultSaveInvVariance,
                                     const BatchNormDirection_t& dir,
                                     bool print_for_bn_driver = true);
} // namespace debug
} // namespace miopen
