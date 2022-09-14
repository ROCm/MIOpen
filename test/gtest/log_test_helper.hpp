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

#include <miopen/config.h>
#include <miopen/convolution.hpp>

#if MIOPEN_BACKEND_OPENCL
#define BKEND "OpenCL"
#elif MIOPEN_BACKEND_HIP
#define BKEND "HIP"
#endif

extern std::string const logConv;
extern std::string const logFindConv;

extern std::string const envConv;
extern std::string const envFindConv;

// Copy of struct that is in miopen.
// This is for testing purpose only.
enum class ConvDirection
{
    Fwd = 1,
    Bwd = 2,
    WrW = 4
};

// Copy of function declaration that is in miopen.
// This is for testing purpose only.
void LogCmdConvolution(const miopenTensorDescriptor_t& xDesc,
                       const miopenTensorDescriptor_t& wDesc,
                       const miopenConvolutionDescriptor_t& convDesc,
                       const miopenTensorDescriptor_t& yDesc,
                       const ConvDirection& conv_dir,
                       bool is_immediate);
// Copy of function declaration that is in miopen.
// This is for testing purpose only.
void LogCmdFindConvolution(const miopenTensorDescriptor_t& xDesc,
                           const miopenTensorDescriptor_t& wDesc,
                           const miopenConvolutionDescriptor_t& convDesc,
                           const miopenTensorDescriptor_t& yDesc,
                           const ConvDirection& conv_dir,
                           bool is_immediate);

// Function that is used in multiple test cases.
void TestLogFun(std::function<void(const miopenTensorDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const miopenConvolutionDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const ConvDirection&,
                                   bool)> const& func,
                std::string env_var,
                std::string sub_str,
                bool set_env);
