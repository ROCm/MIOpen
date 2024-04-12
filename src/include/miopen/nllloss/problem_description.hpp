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
                       int ignore_index_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          weightDesc(weightDesc_),
          outputDesc(outputDesc_),
          ignore_index(ignore_index_),
          N_total(outputDesc_.GetElementSize()),
          N(inputDesc_.GetLengths()[0]),
          C(inputDesc_.GetLengths()[1]),
          D1(inputDesc_.GetLengths()[2]), 
          D2(inputDesc_.GetLengths()[3])
    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    int GetIgnoreIndex() const { return ignore_index; }
    size_t GetNtotal() const { return N_total; }
    size_t GetC() const { return C; }
    size_t GetD1() const { return D1; }
    size_t GetD2() const { return D2; }

    /* input(input): [N, C, D1, D2], target(target): [N, D1, D2],
    * weight(weight): [C], output(output): [N, D1, D2] */
    bool IsRightDim() const
    { 
        if (outputDesc.GetLengths()[0] != N || outputDesc.GetLengths()[1] != D1 || outputDesc.GetLengths()[2] != D2 ||
            targetDesc.GetLengths()[0] != N || targetDesc.GetLengths()[1] != D1 || targetDesc.GetLengths()[2] != D2 ||
            weightDesc.GetLengths()[0] != C)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(
                miopenStatusBadParm,
                "NLLLoss: Tensors dimension do not match.");
#else
            return false;
#endif
        }
        return true; 
    }
    
    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Tensor types of Input and Output do not match.");
#else
            return false;
#endif
        }
        if(outputDesc.GetType() != weightDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Tensor types of Output and Weight do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inputDesc.IsPacked() && targetDesc.IsPacked() && weightDesc.IsPacked() && outputDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor outputDesc;
    
    int ignore_index;
    size_t N_total;
    size_t N;
    size_t C;
    size_t D1;
    size_t D2;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace nllloss

} // namespace miopen
