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

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace adam {

struct AdamInvokeParams : public miopen::InvokeParams
{
    AdamInvokeParams() = default;

    const TensorDescriptor* paramDesc = nullptr;
    const TensorDescriptor* gradDesc  = nullptr;

    ConstData_t paramIn       = nullptr;
    Data_t paramOut           = nullptr;
    Data_t paramOutFloat16    = nullptr;
    ConstData_t gradIn        = nullptr;
    ConstData_t expAvgIn      = nullptr;
    Data_t expAvgOut          = nullptr;
    ConstData_t expAvgSqIn    = nullptr;
    Data_t expAvgSqOut        = nullptr;
    ConstData_t maxExpAvgSqIn = nullptr;
    Data_t maxExpAvgSqOut     = nullptr;
    ConstData_t gradScale     = nullptr;
    ConstData_t foundInf      = nullptr;
    ConstData_t stepIn        = nullptr;
    Data_t stepOut            = nullptr;

    uint32_t step      = 0;
    float lr           = 0.0;
    float beta1        = 0.0;
    float beta2        = 0.0;
    float weight_decay = 0.0;
    float eps          = 0.0;
    bool amsgrad       = false;
    bool maximize      = false;
    bool adamw         = false;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct TransformersAdamWInvokeParams : public miopen::InvokeParams
{
    TransformersAdamWInvokeParams() = default;

    const TensorDescriptor* paramDesc = nullptr;
    const TensorDescriptor* gradDesc  = nullptr;

    ConstData_t paramIn    = nullptr;
    Data_t paramOut        = nullptr;
    Data_t paramOutFloat16 = nullptr;
    ConstData_t gradIn     = nullptr;
    ConstData_t expAvgIn   = nullptr;
    Data_t expAvgOut       = nullptr;
    ConstData_t expAvgSqIn = nullptr;
    Data_t expAvgSqOut     = nullptr;
    ConstData_t gradScale  = nullptr;
    ConstData_t foundInf   = nullptr;
    ConstData_t stepIn     = nullptr;
    Data_t stepOut         = nullptr;

    uint32_t step      = 0;
    float lr           = 0.0;
    float beta1        = 0.0;
    float beta2        = 0.0;
    float eps          = 0.0;
    float weight_decay = 0.0;
    float step_size    = 0.0;
    bool correct_bias  = true;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace adam
} // namespace miopen
