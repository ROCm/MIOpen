/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/invoker.hpp>
#include <miopen/kernel.hpp>
#include <miopen/conv/context.hpp>

#include <vector>

namespace miopen {
namespace conv {

// Beside used in invoker, currently this function is only called in RunAndMeasure() of dynamic
// igemm solver
// Remove this in the future when invoker is fully re-factored.
float CallImplicitGemmDynamic(const miopen::Handle& handle,
                              const ConvolutionContext& ctx,
                              ConstData_t src,
                              Data_t dst,
                              ConstData_t wei,
                              const std::vector<KernelInvoke>& kernels);

float CallImplicitGemmWrwDynamic(const miopen::Handle& handle,
                                 const ConvolutionContext& ctx,
                                 ConstData_t src,
                                 ConstData_t dst,
                                 Data_t wei,
                                 Data_t wei_workspace,
                                 const std::vector<KernelInvoke>& kernels);

InvokerFactory MakeImplGemmDynamicDataInvokerFactory(const ConvolutionContext& ctx);

} // namespace conv
} // namespace miopen
