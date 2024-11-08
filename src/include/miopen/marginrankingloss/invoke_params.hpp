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

#include <miopen/common.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace marginrankingloss {

struct FwdInvokeParams : public miopen::InvokeParams
{
    FwdInvokeParams() = default;

    const TensorDescriptor* input1Desc = nullptr;
    const TensorDescriptor* input2Desc = nullptr;
    const TensorDescriptor* targetDesc = nullptr;
    const TensorDescriptor* outputDesc = nullptr;

    ConstData_t input1 = nullptr;
    ConstData_t input2 = nullptr;
    ConstData_t target = nullptr;
    Data_t output      = nullptr;
    float margin       = 0;
    miopenMarginRakningLossReductionMode_t reduction_mode;

    size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct BwdInvokeParams : public miopen::InvokeParams
{
    BwdInvokeParams() = default;

    const TensorDescriptor* input1Desc  = nullptr;
    const TensorDescriptor* input2Desc  = nullptr;
    const TensorDescriptor* targetDesc  = nullptr;
    const TensorDescriptor* outGradDesc = nullptr;
    const TensorDescriptor* in1GradDesc = nullptr;
    const TensorDescriptor* in2GradDesc = nullptr;

    ConstData_t input1  = nullptr;
    ConstData_t input2  = nullptr;
    ConstData_t target  = nullptr;
    ConstData_t outGrad = nullptr;
    Data_t in1Grad      = nullptr;
    Data_t in2Grad      = nullptr;
    float margin        = 0;
    miopenMarginRakningLossReductionMode_t reduction_mode;

    size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace marginrankingloss
} // namespace miopen
