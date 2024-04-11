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

#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace nllloss {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& targetDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& outputDesc_,
                       long ignore_index_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          weightDesc(weightDesc_),
          outputDesc(outputDesc_),
          ignore_index(ignore_index_),
          N_total(outputDesc_.GetElementSize()),
          C(inputDesc_.GetLengths()[1]),
          D1(inputDesc_.GetLengths()[2]), 
          D2(inputDesc_.GetLengths()[3])
    {
        
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    long GetIgnoreIndex() const { return ignore_index; }
    long GetNtotal() const { return N_total; }
    long GetC() const { return C; }
    long GetD1() const { return D1; }
    long GetD2() const { return D2; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor outputDesc;
    
    long ignore_index;
    long N_total;
    long C;
    long D1;
    long D2;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace nllloss

} // namespace miopen
