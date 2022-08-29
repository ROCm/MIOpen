/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>

#include "miopen/logger.hpp"
#include <miopen/convolution.hpp>

#if MIOPEN_BACKEND_OPENCL
#define BKEND "OpenCL"
#elif MIOPEN_BACKEND_HIP
#define BKEND "HIP"
#endif

static const std::string& logConv =
    "MIOpen(" BKEND
    "): Command [LogCmdConvolution] ./bin/MIOpenDriver conv -n 128 -c 3 -H 32 -W 32 -k "
    "64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1";
static const std::string& logFindConv =
    "MIOpen(" BKEND
    "): Command [LogCmdFindConvolution] ./bin/MIOpenDriver conv -n 128 -c 3 -H 32 -W 32 "
    "-k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1";

static const std::string& envConv     = "MIOPEN_ENABLE_LOGGING_CMD";
static const std::string& envFindConv = "MIOPEN_ENABLE_LOGGING_CMD_FIND";

void LogCmdConvolution(const miopenTensorDescriptor_t& xDesc,
                       const miopenTensorDescriptor_t& wDesc,
                       const miopenConvolutionDescriptor_t& convDesc,
                       const miopenTensorDescriptor_t& yDesc,
                       const miopen::ConvDirection& conv_dir,
                       bool is_immediate);

void LogCmdFindConvolution(const miopenTensorDescriptor_t& xDesc,
                           const miopenTensorDescriptor_t& wDesc,
                           const miopenConvolutionDescriptor_t& convDesc,
                           const miopenTensorDescriptor_t& yDesc,
                           const miopen::ConvDirection& conv_dir,
                           bool is_immediate);

void TestLogFun(std::function<void(const miopenTensorDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const miopenConvolutionDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const miopen::ConvDirection&,
                                   bool)> const& func,
                std::string env_var,
                std::string sub_str,
                bool set_env);
