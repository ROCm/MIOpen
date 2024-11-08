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

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace reduce {

struct ExtremeInvokeParams : public miopen::InvokeParams
{
    ExtremeInvokeParams() = default;

    const TensorDescriptor* xDesc      = nullptr;
    const TensorDescriptor* yDesc      = nullptr;
    const TensorDescriptor* indiceDesc = nullptr;

    ConstData_t x              = nullptr;
    Data_t y                   = nullptr;
    Data_t indice              = nullptr;
    std::size_t workspace_size = 0;
    int32_t dim                = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct CalculationInvokeParams : public miopen::InvokeParams
{
    CalculationInvokeParams() = default;

    const TensorDescriptor* xDesc = nullptr;
    const TensorDescriptor* yDesc = nullptr;

    ConstData_t x              = nullptr;
    Data_t y                   = nullptr;
    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    int32_t dim                = 0;
    miopenReduceCalculationNanPropagation_t nanPropagation =
        MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

} // namespace reduce

} // namespace miopen
