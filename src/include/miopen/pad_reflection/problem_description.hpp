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

namespace pad_reflection {

struct PadReflectionFwdProblemDescription : ProblemDescriptionBase
{
    PadReflectionFwdProblemDescription(const TensorDescriptor& xDesc_,
                                       const TensorDescriptor& yDesc_,
                                       const int64_t* padding_,
                                       const size_t num_padding_)
        : xDesc(xDesc_), yDesc(yDesc_), padding(padding_), num_padding(num_padding_)
    {
        IsSameType();
        IsSameShape();
        IsValidNumDim();
        IsValidPadding();
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameShape() const
    {
        if(xDesc.GetNumDims() != yDesc.GetNumDims())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionFwd: Input and output tensors' dimensions don't match");
        return true;
    }

    bool IsValidNumDim() const
    {
        if(xDesc.GetNumDims() != 3)
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionFwd: Only support for Pad Reflect 1D (input_num_dims = 3)");

        return true;
    }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionFwd: Input and output tensors' types don't match");

        return true;
    }

    bool IsValidPadding() const
    {
        auto input_dims = xDesc.GetLengths();
        if((input_dims.size() == 3 && num_padding != 2))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionFwd: input_grad_dims_size and padding size "
                         "mismatch/invalid.");
        }

        std::vector<size_t> input_with_padding_dims(input_dims);
        for(uint64_t i = 0; i < num_padding / 2; i++)
        {
            int idx = input_dims.size() - i - 1;
            if(padding[i * 2] >= static_cast<int64_t>(input_dims[idx]) ||
               padding[i * 2 + 1] >= static_cast<int64_t>(input_dims[idx]))
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "PadReflectionFwd: Padding size should be less than the corresponding "
                             "input_grad dimension.");
            }

            input_with_padding_dims[idx] += padding[i * 2] + padding[i * 2 + 1];
        }

        if(input_with_padding_dims != yDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionFwd: Input + padding tensor and output tensor do not match");

        return true;
    }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const int64_t* padding;
    const size_t num_padding;
};

struct PadReflectionBwdProblemDescription : ProblemDescriptionBase
{
    PadReflectionBwdProblemDescription(const TensorDescriptor& dxDesc_,
                                       const TensorDescriptor& dyDesc_,
                                       const int64_t* padding_,
                                       const size_t num_padding_)
        : dxDesc(dxDesc_), dyDesc(dyDesc_), padding(padding_), num_padding(num_padding_)
    {
        IsSameType();
        IsValidNumDim();
        IsValidPadding();
    }

    const TensorDescriptor& GetdXDesc() const { return dxDesc; }
    const TensorDescriptor& GetdYDesc() const { return dyDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameType() const
    {
        if(dxDesc.GetType() != dyDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionBwd: Input grad and output grad tensors' types don't match");

        return true;
    }

    bool IsSameShape() const
    {
        if(dxDesc.GetNumDims() != dyDesc.GetNumDims())
            MIOPEN_THROW(
                miopenStatusBadParm,
                "PadReflectionBwd: Input grad and output grad tensors' dimensions don't match");
        return true;
    }

    bool IsValidNumDim() const
    {
        if(dxDesc.GetNumDims() != 3)
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionBwd: Only support for Pad Reflect 1D (input_num_dims = 3)");

        return true;
    }

    bool IsValidPadding() const
    {
        auto input_grad_dims = dxDesc.GetLengths();

        if(num_padding != 2)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "PadReflectionBwd: Only support for 3D input tensor and num_padding_elements = 2");
        }

        std::vector<size_t> input_with_padding_dims(input_grad_dims);
        for(uint64_t i = 0; i < num_padding / 2; i++)
        {
            int idx = input_grad_dims.size() - i - 1;
            if(padding[i * 2] >= static_cast<int64_t>(input_grad_dims[idx]) ||
               padding[i * 2 + 1] >= static_cast<int64_t>(input_grad_dims[idx]))
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "PadReflectionBwd: Padding size should be less than the corresponding "
                             "input_grad dimension.");
            }
            input_with_padding_dims[idx] += padding[i * 2] + padding[i * 2 + 1];
        }

        if(input_with_padding_dims != dyDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadReflectionBwd: Input + padding tensor and output tensor do not match");

        return true;
    }

    bool IsContiguous() const { return dxDesc.IsContiguous() && dyDesc.IsContiguous(); }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& dxDesc;
    const TensorDescriptor& dyDesc;
    const int64_t* padding;
    const size_t num_padding;
};

} // namespace pad_reflection
} // namespace miopen
