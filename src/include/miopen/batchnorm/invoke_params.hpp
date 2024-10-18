/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
namespace batchnorm {

struct FwdTrainInvokeParams : public miopen::InvokeParams
{
    FwdTrainInvokeParams() = default;

    ConstData_t x                = nullptr;
    Data_t y                     = nullptr;
    ConstData_t bnScale          = nullptr;
    ConstData_t bnBias           = nullptr;
    double expAvgFactor          = 0;
    Data_t resultRunningMean     = nullptr;
    Data_t resultRunningVariance = nullptr;
    double epsilon               = 0;
    Data_t resultSaveMean        = nullptr;
    Data_t resultSaveInvVariance = nullptr;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct BwdInvokeParams : public miopen::InvokeParams
{
    BwdInvokeParams() = default;

    ConstData_t x                = nullptr;
    ConstData_t dy               = nullptr;
    Data_t dx                    = nullptr;
    ConstData_t bnScale          = nullptr;
    Data_t resultBnScaleDiff     = nullptr;
    Data_t resultBnBiasDiff      = nullptr;
    double epsilon               = 0;
    ConstData_t savedMean        = nullptr;
    ConstData_t savedInvVariance = nullptr;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct InfInvokeParams : public miopen::InvokeParams
{
    InfInvokeParams() = default;

    const TensorDescriptor* xDesc = nullptr;

    ConstData_t x                 = nullptr;
    Data_t y                      = nullptr;
    ConstData_t bnScale           = nullptr;
    ConstData_t bnBias            = nullptr;
    ConstData_t estimatedMean     = nullptr;
    ConstData_t estimatedVariance = nullptr;
    double epsilon                = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace batchnorm

} // namespace miopen
