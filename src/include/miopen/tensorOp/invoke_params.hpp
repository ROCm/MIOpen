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

namespace tensorOp {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams(const void* alpha0_,
                 ConstData_t ATensor_,
                 const void* alpha1_,
                 ConstData_t BTensor_,
                 const void* beta_,
                 Data_t CTensor_,
                 const size_t Aoffset_,
                 const size_t Boffset_,
                 const size_t Coffset_)
        : alpha0(alpha0_),
          alpha1(alpha1_),
          beta(beta_),
          ATensor(ATensor_),
          BTensor(BTensor_),
          CTensor(CTensor_),
          Aoffset(Aoffset_),
          Boffset(Boffset_),
          Coffset(Coffset_)
    {
    }

    size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }

public:
    const void* alpha0;
    const void* alpha1;
    const void* beta;

    ConstData_t ATensor;
    ConstData_t BTensor;
    Data_t CTensor;

    size_t Aoffset;
    size_t Boffset;
    size_t Coffset;
};

} // namespace tensorOp

} // namespace miopen
