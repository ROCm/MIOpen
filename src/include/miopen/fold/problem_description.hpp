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

#include "miopen/errors.hpp"
#include "miopen/miopen.h"
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace fold {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

// struct FoldFwdProblemDescription : ProblemDescriptionBase
// {
//     FoldFwdProblemDescription(const TensorDescriptor& inputDesc_,
//                                 const TensorDescriptor& outputDesc_,
//                                 const int32_t* kernel_size_,
//                                 const int kernel_size_size_,
//                                 const int32_t* stride_,
//                                 const int stride_size_,
//                                 const int32_t* padding_,
//                                 const int padding_size_,
//                                 const int32_t* dilation_,
//                                 const int dilation_size_)
//         : inputDesc(inputDesc_),
//           outputDesc(outputDesc_),
//             kernel_size(kernel_size_),
//             kernel_size_size(kernel_size_size_),
//             stride(stride_),
//             stride_size(stride_size_),
//             padding(padding_),
//             padding_size(padding_size_),
//             dilation(dilation_),
//             dilation_size(dilation_size_)
//     {
//         // IsValidSize();
//     }

// //     bool IsValidSize() const
// //     {
// //         if(inputDesc.GetSize() < 2 || inputDesc.GetSize() > 5)
// //         {
// // #if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
// //             MIOPEN_THROW(miopenStatusBadParm,
// //                          "Instance Norm: The input tensor dimension should be in range [2,
// 5].");
// // #else
// //             return false;
// // #endif
// //         }
// //         return true;
// //     }

//     const TensorDescriptor& GetInputDesc() const { return inputDesc; }
//     const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

//     NetworkConfig MakeNetworkConfig() const override;

// public:
//     TensorDescriptor inputDesc;
//     TensorDescriptor outputDesc;
//     const int32_t* kernel_size;
//     const int kernel_size_size;
//     const int32_t* stride;
//     const int stride_size;
//     const int32_t* padding;
//     const int padding_size;
//     const int32_t* dilation;
//     const int dilation_size;
// };

struct UnfoldFwdProblemDescription : ProblemDescriptionBase
{
    UnfoldFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                const TensorDescriptor& outputDesc_,
                                const int32_t* kernel_size_,
                                const int kernel_size_size_,
                                const int32_t* stride_,
                                const int stride_size_,
                                const int32_t* padding_,
                                const int padding_size_,
                                const int32_t* dilation_,
                                const int dilation_size_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          kernel_size(kernel_size_),
          kernel_size_size(kernel_size_size_),
          stride(stride_),
          stride_size(stride_size_),
          padding(padding_),
          padding_size(padding_size_),
          dilation(dilation_),
          dilation_size(dilation_size_)
    {
        // IsValidSize();
    }

    //     bool IsValidSize() const
    //     {
    //         if(inputDesc.GetSize() < 2 || inputDesc.GetSize() > 5)
    //         {
    // #if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
    //             MIOPEN_THROW(miopenStatusBadParm,
    //                          "Instance Norm: The input tensor dimension should be in range [2,
    //                          5].");
    // #else
    //             return false;
    // #endif
    //         }
    //         return true;
    //     }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    const int32_t* kernel_size;
    const int kernel_size_size;
    const int32_t* stride;
    const int stride_size;
    const int32_t* padding;
    const int padding_size;
    const int32_t* dilation;
    const int dilation_size;
};

} // namespace fold

} // namespace miopen
