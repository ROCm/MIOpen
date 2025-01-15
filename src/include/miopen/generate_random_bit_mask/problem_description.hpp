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

#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace generate_random_bit_mask {

struct PStateProblemDescription : ProblemDescriptionBase
{
    PStateProblemDescription(const size_t stateSizeInBytes_) : stateSizeInBytes(stateSizeInBytes_)
    {
        if(stateSizeInBytes == 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GenerateRandomBitMask: State size in bytes must be greater than 0");
        }
    }

    size_t GetStateSizeInBytes() const { return stateSizeInBytes; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    size_t stateSizeInBytes;
};

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& pstateDesc_,
                       const TensorDescriptor& maskDesc_,
                       const float p_)
        : pstateDesc(pstateDesc_), maskDesc(maskDesc_), p(p_)
    {

        IsValidProbValue();
        IsRightType();
    }

    const TensorDescriptor& GetMaskDesc() const { return maskDesc; }
    float GetProb() const { return p; }
    float GetStateSizeInBytes() const { return pstateDesc.GetNumBytes(); }

    bool IsValidProbValue() const
    {
        if(p < 0 || p > 1)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "GenerateRandomBitMask: Probability value must be in the range [0, 1], but got " +
                    std::to_string(p));
        }

        return true;
    }

    bool IsRightType() const
    {
        if(maskDesc.GetType() != miopenInt8)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GenerateRandomBitMask: Mask tensor must be of type byte/uint8/int8");
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& pstateDesc;
    const TensorDescriptor& maskDesc;
    float p;
};

} // namespace generate_random_bit_mask
} // namespace miopen
