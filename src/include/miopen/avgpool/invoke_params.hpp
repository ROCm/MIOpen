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

#pragma once

#include "miopen/common.hpp"
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace avgpool {

struct FwdInvokeParams : public miopen::InvokeParams
{

    FwdInvokeParams() = default;

    const TensorDescriptor* inputDesc   = nullptr;
    const TensorDescriptor* outputDesc  = nullptr;
    const TensorDescriptor* strideDesc  = nullptr;
    const TensorDescriptor* paddingDesc = nullptr;
    const TensorDescriptor* kinfor      = nullptr;

    ConstData_t input   = nullptr;
    Data_t output       = nullptr;
    ConstData_t stride  = nullptr;
    ConstData_t padding = nullptr;
    ConstData_t kinfo   = nullptr;

    const bool count_include_pad   = false;
    const int32_t divisor_override = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct BwdInvokeParams : public miopen::InvokeParams
{

    BwdInvokeParams() = default;

    const TensorDescriptor* outputGradDesc = nullptr;
    const TensorDescriptor* inputGradDesc  = nullptr;
    const TensorDescriptor* strideDesc     = nullptr;
    const TensorDescriptor* paddingDesc    = nullptr;
    const TensorDescriptor* kinfor         = nullptr;

    ConstData_t output_grad = nullptr;
    Data_t input_grad       = nullptr;
    ConstData_t stride      = nullptr;
    ConstData_t padding     = nullptr;
    ConstData_t kinfo       = nullptr;

    const bool count_include_pad   = false;
    const int32_t divisor_override = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace avgpool
} // namespace miopen
