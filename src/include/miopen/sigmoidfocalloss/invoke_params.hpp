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
#include <miopen/miopen.h>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace sigmoidfocalloss {

struct SigmoidFocalLossInvokeParams : public miopen::InvokeParams
{
    SigmoidFocalLossInvokeParams() = default;

    const TensorDescriptor* inputDesc  = nullptr;
    const TensorDescriptor* targetDesc = nullptr;

    ConstData_t input                   = nullptr;
    ConstData_t target                  = nullptr;
    Data_t workspace                    = nullptr;
    std::size_t workspace_size          = 0;
    float alpha                         = 0.25;
    float gamma                         = 2.0f;
    miopenLossReductionMode_t reduction = MIOPEN_LOSS_REDUCTION_NONE;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

struct FwdInvokeParams : SigmoidFocalLossInvokeParams
{
    FwdInvokeParams() = default;

    const TensorDescriptor* outputDesc = nullptr;
    Data_t output                      = nullptr;
};

struct BwdInvokeParams : SigmoidFocalLossInvokeParams
{
    BwdInvokeParams() = default;

    const TensorDescriptor* doutputDesc = nullptr;
    const TensorDescriptor* dinputDesc  = nullptr;
    const TensorDescriptor* dtargetDesc = nullptr;

    ConstData_t doutput = nullptr;
    ConstData_t dinput  = nullptr;
    ConstData_t dtarget = nullptr;
};

} // namespace sigmoidfocalloss

} // namespace miopen
