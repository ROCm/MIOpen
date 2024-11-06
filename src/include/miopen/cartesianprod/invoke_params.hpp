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

namespace cartesianprod {

struct FwdInvokeParams : public miopen::InvokeParams
{

    FwdInvokeParams() = default;

    const TensorDescriptor* const* inputDescs = nullptr;
    const TensorDescriptor* outputDesc        = nullptr;

    uint64_t inputCount = 0;
    ConstData_t* inputs = nullptr;
    Data_t output       = nullptr;

    const void* GetInput(uint64_t inputIndex) const
    {
        return inputIndex < inputCount ? inputs[inputIndex] : nullptr;
    }

    const TensorDescriptor* GetInputDesc(uint64_t inputIndex) const
    {
        return inputIndex < inputCount ? inputDescs[inputIndex] : nullptr;
    }

    std::uint64_t workspaceSize = 0;
    Data_t workspace            = nullptr;

    std::uint64_t GetWorkspaceSize() const { return workspaceSize; }
    Data_t GetWorkspace() const { return workspace; }
};

struct BwdInvokeParams : public miopen::InvokeParams
{

    BwdInvokeParams() = default;

    const TensorDescriptor* outputGradDesc        = nullptr;
    const TensorDescriptor* const* inputGradDescs = nullptr;

    uint64_t inputCount     = 0;
    ConstData_t output_grad = nullptr;
    Data_t* input_grads     = nullptr;

    void* GetInputGrad(uint64_t inputIndex) const
    {
        return inputIndex < inputCount ? input_grads[inputIndex] : nullptr;
    }

    const TensorDescriptor* GetInputGradDesc(uint64_t inputIndex) const
    {
        return inputIndex < inputCount ? inputGradDescs[inputIndex] : nullptr;
    }

    std::uint64_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace cartesianprod

} // namespace miopen
