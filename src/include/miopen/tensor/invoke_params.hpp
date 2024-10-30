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

namespace tensor {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams(miopenTensorOp_t tensorOp_,
                 const void* alpha0_,
                 const TensorDescriptor& aTensorDesc_,
                 ConstData_t ATensor_,
                 const void* alpha1_,
                 const TensorDescriptor& bTensorDesc_,
                 ConstData_t BTensor_,
                 const void* beta_,
                 const TensorDescriptor& cTensorDesc_,
                 Data_t CTensor_,
                 const size_t Aoffset_,
                 const size_t Boffset_,
                 const size_t Coffset_,
                 const bool nonStandardSquash_)
        : alpha0(alpha0_),
          alpha1(alpha1_),
          beta(beta_),
          tensorOp(tensorOp_),
          aTensorDesc(aTensorDesc_),
          ATensor(ATensor_),
          bTensorDesc(bTensorDesc_),
          BTensor(BTensor_),
          cTensorDesc(cTensorDesc_),
          CTensor(CTensor_),
          Aoffset(Aoffset_),
          Boffset(Boffset_),
          Coffset(Coffset_),
          nonStandardSquash(nonStandardSquash_)
    {
    }

    size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }

public:
    const void* alpha0;
    const void* alpha1;
    const void* beta;

    miopenTensorOp_t tensorOp;

    TensorDescriptor aTensorDesc;
    ConstData_t ATensor;

    TensorDescriptor bTensorDesc;
    ConstData_t BTensor;

    TensorDescriptor cTensorDesc;
    Data_t CTensor;

    size_t Aoffset;
    size_t Boffset;
    size_t Coffset;

    bool nonStandardSquash;
};

} // namespace tensor

} // namespace miopen
