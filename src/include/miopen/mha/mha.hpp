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

#include <miopen/tensor.hpp>

namespace miopen {
namespace mha {

struct MhaInputDescsForward
{
    // input tensors
    TensorDescriptor kDesc;
    TensorDescriptor qDesc;
    TensorDescriptor vDesc;

    // input scaling tensors
    TensorDescriptor descaleKDesc;
    TensorDescriptor descaleQDesc;
    TensorDescriptor descaleVDesc;
    TensorDescriptor descaleSDesc;
    TensorDescriptor scaleSDesc;
    TensorDescriptor scaleODesc;

    // input scalars
    float scale;

    // input dropout tensors
    TensorDescriptor dropoutProbabilityDesc;
    TensorDescriptor dropoutSeedDesc;
    TensorDescriptor dropoutOffsetDesc;

    TensorDescriptor biasDesc;

    // output tensors
    TensorDescriptor oDesc;
    TensorDescriptor amaxODesc;
    TensorDescriptor amaxSDesc;

    // output tensors for training only
    TensorDescriptor mDesc;
    TensorDescriptor zInvDesc;
};

struct MhaInputDescsBackward
{
    // input tensors
    TensorDescriptor kDesc;
    TensorDescriptor qDesc;
    TensorDescriptor vDesc;

    TensorDescriptor oDesc;
    TensorDescriptor doDesc;

    // input tensors from fwd pass
    TensorDescriptor mDesc;
    TensorDescriptor zInvDesc;

    // input scaling tensors
    TensorDescriptor descaleKDesc;
    TensorDescriptor descaleQDesc;
    TensorDescriptor descaleVDesc;
    TensorDescriptor descaleSDesc;
    TensorDescriptor descaleODesc;
    TensorDescriptor descaleDODesc;
    TensorDescriptor descaleDSDesc;

    TensorDescriptor scaleSDesc;
    TensorDescriptor scaleDSDesc;
    TensorDescriptor scaleDQDesc;
    TensorDescriptor scaleDKDesc;
    TensorDescriptor scaleDVDesc;

    // input scalars
    float scale;

    TensorDescriptor dropoutProbabilityDesc;
    TensorDescriptor dropoutSeedDesc;
    TensorDescriptor dropoutOffsetDesc;

    // output tensors
    TensorDescriptor dqDesc;
    TensorDescriptor dkDesc;
    TensorDescriptor dvDesc;
    TensorDescriptor amaxDQDesc;
    TensorDescriptor amaxDKDesc;
    TensorDescriptor amaxDVDesc;
    TensorDescriptor amaxDSDesc;
};

struct MhaDataForward
{
    // input tensors
    ConstData_t kData;
    ConstData_t qData;
    ConstData_t vData;

    // input scaling tensors
    ConstData_t descaleKData;
    ConstData_t descaleQData;
    ConstData_t descaleVData;
    ConstData_t descaleSData;
    ConstData_t scaleSData;
    ConstData_t scaleOData;

    ConstData_t dropoutProbabilityData;
    ConstData_t dropoutSeedData;
    ConstData_t dropoutOffsetData;

    ConstData_t biasData;
    miopenMhaMask_t mask;

    // output tensors
    Data_t oData;
    Data_t amaxOData;
    Data_t amaxSData;

    // output tensors for training only
    Data_t mData;
    Data_t zInvData;
};

struct MhaDataBackward
{
    // input tensors
    ConstData_t kData;
    ConstData_t qData;
    ConstData_t vData;

    ConstData_t oData;
    ConstData_t doData;

    // input tensors from fwd pass
    ConstData_t mData;
    ConstData_t zInvData;

    // input scaling tensors
    ConstData_t descaleKData;
    ConstData_t descaleQData;
    ConstData_t descaleVData;
    ConstData_t descaleSData;
    ConstData_t descaleOData;
    ConstData_t descaleDOData;
    ConstData_t descaleDSData;
    ConstData_t scaleSData;
    ConstData_t scaleDSData;
    ConstData_t scaleDQData;
    ConstData_t scaleDKData;
    ConstData_t scaleDVData;

    ConstData_t dropoutProbabilityData;
    ConstData_t dropoutSeedData;
    ConstData_t dropoutOffsetData;

    // output tensors
    Data_t dqData;
    Data_t dkData;
    Data_t dvData;
    Data_t amaxDQData;
    Data_t amaxDKData;
    Data_t amaxDVData;
    Data_t amaxDSData;
};

} // namespace mha
} // namespace miopen
