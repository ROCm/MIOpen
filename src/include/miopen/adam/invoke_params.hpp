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
namespace adam {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams() = default;

    const TensorDescriptor* paramDesc     = nullptr;
    const TensorDescriptor* gradDesc      = nullptr;
    const TensorDescriptor* expAvgDesc    = nullptr;
    const TensorDescriptor* expAvgSqDesc  = nullptr;
    const TensorDescriptor* stepDesc      = nullptr;
    const TensorDescriptor* gradScaleDesc = nullptr;
    const TensorDescriptor* foundInfDesc  = nullptr;

    Data_t param          = nullptr;
    ConstData_t grad      = nullptr;
    Data_t expAvg         = nullptr;
    Data_t expAvgSq       = nullptr;
    Data_t step           = nullptr;
    ConstData_t gradScale = nullptr;
    ConstData_t foundInf  = nullptr;

    double lr           = 0;
    double beta1        = 0;
    double beta2        = 0;
    double weight_decay = 0;
    double eps          = 0;
    bool amsgrad        = false;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace adam
} // namespace miopen
