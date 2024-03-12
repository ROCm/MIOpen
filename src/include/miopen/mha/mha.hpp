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

// Are we going to use backend descriptors? If yes just chnage the typedef
typedef TensorDescriptor MHATensorDescriptor;


struct MHAInputDescsForward
{
    //input tensors
    MHATensorDescriptor kDesc;
    MHATensorDescriptor qDesc;
    MHATensorDescriptor vDesc;

    //input scaling tensors
    MHATensorDescriptor descaleKDesc;
    MHATensorDescriptor descaleQDesc;
    MHATensorDescriptor descaleVDesc;
    MHATensorDescriptor descaleSDesc;
    MHATensorDescriptor scaleSDesc;
    MHATensorDescriptor scaleODesc;

    //input scalars
    float scale;
    float dropoutProbability;
    MHATensorDescriptor dropoutSeed;
    MHATensorDescriptor dropoutOffset;
 
    //output tensors
    MHATensorDescriptor oDesc;
    MHATensorDescriptor amaxODesc;
    MHATensorDescriptor amaxSDesc;

    //output tensors for training only
    MHATensorDescriptor mDesc;
    MHATensorDescriptor zInvDesc;
};

struct MHADataForward 
{
    //input tensors
    ConstData_t kData;
    ConstData_t qData;
    ConstData_t vData;

    //input scaling tensors
    ConstData_t descaleKData;
    ConstData_t descaleQData;
    ConstData_t descaleVData;
    ConstData_t descaleSData;
    ConstData_t scaleSData;
    ConstData_t scaleOData;

    ConstData_t dropoutSeedData;
    ConstData_t dropoutOffsetData;
 
    //output tensors
    Data_t oData;
    Data_t amaxOData;
    Data_t amaxSData;

    //output tensors for training only
    Data_t mData;
    Data_t zInvData;
};


} // mha
} // miopen
