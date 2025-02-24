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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <sstream>
#include <vector>

namespace miopen {

struct NetworkConfig;

namespace logsumexp {

struct ProblemDescriptionForward : ProblemDescriptionBase
{
    ProblemDescriptionForward(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& outputDesc_,
                              const std::vector<int>& dims_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), dims(dims_)
    {
        EliminateNegativeDims();
        IsSameType();
        IsValidDims();
        IsValidOutputSize();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsSameType() const
    {
        if(!(inputDesc.GetType() == outputDesc.GetType()))
            MIOPEN_THROW(miopenStatusBadParm,
                         "LogCumSumExp: Input and Output tensor type do not match.");
        return true;
    }

    void EliminateNegativeDims()
    {
        for(auto& dim : dims)
            if(dim < 0)
                dim += inputDesc.GetNumDims();
    }

    bool IsValidDims() const
    {
        for(auto& dim : dims)
        {
            if(dim + inputDesc.GetNumDims() < 1 || dim > inputDesc.GetNumDims())
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    (std::stringstream() << "LogCumSumExp: Invalid reduce dimension=" << dim << ".")
                        .str());
        }
        return true;
    }

    bool IsValidOutputSize() const
    {
        auto expected_output_dims_1 = inputDesc.GetLengths();
        auto expected_output_dims_2 = expected_output_dims_1;
        for(auto dim : dims)
            expected_output_dims_1[dim] = expected_output_dims_2[dim] = 0;
        for(int i = inputDesc.GetNumDims() - 1; i >= 0; --i)
        {
            if(expected_output_dims_1[i] == 0)
                expected_output_dims_1[i] = 1;
            if(expected_output_dims_2[i] == 0)
                expected_output_dims_2.erase(expected_output_dims_2.begin() + i);
        }
        if(expected_output_dims_2.size() == 0)
            expected_output_dims_2 = {1};
        if(expected_output_dims_1 != outputDesc.GetLengths() &&
           expected_output_dims_2 != outputDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm, "LogCumSumExp: Invalid Output tensor size.");
        return true;
    }

    bool IsAllPacked() const { return inputDesc.IsPacked() && outputDesc.IsPacked(); }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;

    std::vector<int> dims;
};

struct ProblemDescriptionBackward : ProblemDescriptionForward
{
    ProblemDescriptionBackward(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& outputDesc_,
                               const TensorDescriptor& outputGradDesc_,
                               const TensorDescriptor& inputGradDesc_,
                               const std::vector<int>& dims_)
        : ProblemDescriptionForward(inputDesc_, outputDesc_, dims_),
          outputGradDesc(outputGradDesc_),
          inputGradDesc(inputGradDesc_)
    {
        IsSameType();
        IsSameSize();
        IsValidDims();
        IsValidOutputSize();
    }

    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    bool IsSameType() const
    {
        ProblemDescriptionForward::IsSameType();
        if(!(inputDesc.GetType() == inputGradDesc.GetType()))
            MIOPEN_THROW(miopenStatusBadParm,
                         "LogCumSumExp: Input and Input Gradient tensor type do not match.");
        if(!(outputDesc.GetType() == outputGradDesc.GetType()))
            MIOPEN_THROW(miopenStatusBadParm,
                         "LogCumSumExp: Output and Output Gradient tensor type do not match.");
        return true;
    }

    bool IsSameSize() const
    {
        if(!(inputDesc.GetLengths() == inputGradDesc.GetLengths()))
            MIOPEN_THROW(miopenStatusBadParm,
                         "LogCumSumExp: Input and Input Gradient tensor size do not match.");
        if(!(outputDesc.GetLengths() == outputGradDesc.GetLengths()))
            MIOPEN_THROW(miopenStatusBadParm,
                         "LogCumSumExp: Output and Output Gradient tensor size do not match.");
        return true;
    }

    bool IsAllPacked() const
    {
        return ProblemDescriptionForward::IsAllPacked() && inputGradDesc.IsPacked() &&
               outputGradDesc.IsPacked();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
};

} // namespace logsumexp

} // namespace miopen
