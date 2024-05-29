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
#include <string>

namespace miopen {

struct NetworkConfig;

namespace softmarginloss {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& iDesc_,
                              const TensorDescriptor& tDesc_,
                              const TensorDescriptor& oDesc_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_)
    {
    }

    const TensorDescriptor& GetiDesc() const { return iDesc; }
    const TensorDescriptor& GettDesc() const { return tDesc; }
    const TensorDescriptor& GetoDesc() const { return oDesc; }

    bool IsSameType() const
    {
        if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != oDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "SoftMarginLoss: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightLength() const
    {
        if(iDesc.GetLengths() != tDesc.GetLengths() || iDesc.GetLengths() != oDesc.GetLengths())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftMarginLoss: Tensor dimension lengths do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor oDesc;
};

// struct BackwardProblemDescription : ProblemDescriptionBase
// {
//     BackwardProblemDescription(const TensorDescriptor& iDesc_,
//                                const TensorDescriptor& tDesc_,
//                                const TensorDescriptor& dODesc_,
//                                const TensorDescriptor& dIDesc_)
//         : iDesc(iDesc_), tDesc(tDesc_), dODesc(dODesc_), dIDesc(dIDesc_)
//     {
//     }

//     const TensorDescriptor& GetiDesc() const { return iDesc; }
//     const TensorDescriptor& GettDesc() const { return tDesc; }
//     const TensorDescriptor& GetdODesc() const { return dODesc; }
//     const TensorDescriptor& GetdIDesc() const { return dIDesc; }

//     bool IsSameType() const
//     {
//         if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != dODesc.GetType() ||
//            iDesc.GetType() != dIDesc.GetType())
//         {
// #if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
//             MIOPEN_THROW(miopenStatusBadParm, "SoftMarginLoss: Tensor types do not match.");
// #else
//             return false;
// #endif
//         }
//         return true;
//     }

//     bool IsRightLength() const
//     {
//         if(iDesc.GetLengths() != tDesc.GetLengths() || iDesc.GetLengths() != dODesc.GetLengths()
//         ||
//            iDesc.GetLengths() != dIDesc.GetLengths())
//         {
// #if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
//             MIOPEN_THROW(miopenStatusBadParm,
//                          "SoftMarginLoss: Tensor dimension lengths do not match.");
// #else
//             return false;
// #endif
//         }
//         return true;
//     }

//     NetworkConfig MakeNetworkConfig() const override;

// private:
//     TensorDescriptor iDesc;
//     TensorDescriptor tDesc;
//     TensorDescriptor dODesc;
//     TensorDescriptor dIDesc;
// };

} // namespace softmarginloss

} // namespace miopen
