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
#include "miopen/mlo_internal.hpp"
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace mseloss {
namespace forward {
struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* zDesc;

    ConstData_t x    = nullptr;
    ConstData_t y    = nullptr;
    Data_t workspace = nullptr;
    Data_t output    = nullptr;

    float divisor = 1.0f;

    size_t workspace_size = 0;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

} // namespace forward

namespace forward_unreduced {
struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* zDesc;

    ConstData_t x;
    ConstData_t y;
    Data_t z;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};
} // namespace forward_unreduced

namespace backward {
struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* zDesc;
    const TensorDescriptor* dxDesc;
    const TensorDescriptor* dyDesc;

    ConstData_t x;
    ConstData_t y;
    ConstData_t z;
    Data_t dx;
    Data_t dy;

    float divisor = 1.0f;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};
} // namespace backward
namespace backward_unreduced {
struct InvokeParams : public miopen::InvokeParams
{

    InvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* zDesc;
    const TensorDescriptor* dxDesc;
    const TensorDescriptor* dyDesc;

    ConstData_t x;
    ConstData_t y;
    ConstData_t z;
    Data_t dx;
    Data_t dy;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};
} // namespace backward_unreduced
} // namespace mseloss
} // namespace miopen
