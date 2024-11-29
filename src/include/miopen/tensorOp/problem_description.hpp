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

namespace tensorOp {

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const miopenTensorOp_t tensorOp_,
                       const void* beta_,
                       const TensorDescriptor& aTensorDesc_,
                       const TensorDescriptor& bTensorDesc_,
                       const TensorDescriptor& cTensorDesc_,
                       const bool nonStandardSquash_)
        : tensorOp(tensorOp_),
          aTensorDesc(aTensorDesc_),
          bTensorDesc(bTensorDesc_),
          cTensorDesc(cTensorDesc_),
          nonStandardSquash(nonStandardSquash_)
    {
        if(beta_ == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Beta value is nullptr");
        }

        beta = *(static_cast<const float*>(beta_));

        if(aTensorDesc.GetElementSize() != cTensorDesc.GetElementSize())
        {
            MIOPEN_THROW("A and C Tensors do not match");
        }

        if(bTensorDesc.GetType() != cTensorDesc.GetType())
        {
            MIOPEN_THROW("Datatypes for B and C tensors do not match !");
        }

        const auto& blens = bTensorDesc.GetLengths();
        const auto& clens = cTensorDesc.GetLengths();

        if(clens.size() > 5)
        {
            MIOPEN_THROW("Tensor dimension larger than 5: " + std::to_string(clens.size()));
        }

        if(blens.size() != clens.size())
        {
            MIOPEN_THROW("Number of dims in B and C Tensors do not match: " +
                         std::to_string(blens.size()) + ", " + std::to_string(clens.size()));
        }

        if(!nonStandardSquash)
        {
            constexpr auto comparator = [](size_t c, size_t b) { return b == 1 || b == c; };
            const auto [c_diff, b_diff] =
                std::mismatch(clens.begin(), clens.end(), blens.begin(), comparator);
            if(c_diff != clens.end())
                MIOPEN_THROW("BTensor dim != 1 && BTensor dim != CTensor dim:" +
                             std::to_string(std::distance(clens.begin(), c_diff)));
        }
        else
        {
            // non standard behavior because blens[1] can be not equalt to clens[1]
            if(!(clens.size() == 3 && blens[0] == 1 && clens[0] == 1 && blens[2] == clens[2]))
            {
                MIOPEN_THROW(
                    "Non standard squashed operation supported only for 3d tensors and for "
                    "the specific configuration");
            }
        }
    }

    miopenTensorOp_t GetTensorOp() const { return tensorOp; }

    float GetBeta() const { return beta; }

    const TensorDescriptor& GetATensorDesc() const { return aTensorDesc; }
    const TensorDescriptor& GetBTensorDesc() const { return bTensorDesc; }
    const TensorDescriptor& GetCTensorDesc() const { return cTensorDesc; }

    bool GetNonStandardSquash() const { return nonStandardSquash; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const miopenTensorOp_t tensorOp;

    float beta;

    TensorDescriptor aTensorDesc;
    TensorDescriptor bTensorDesc;
    TensorDescriptor cTensorDesc;

    const bool nonStandardSquash;
};

} // namespace tensorOp

} // namespace miopen
