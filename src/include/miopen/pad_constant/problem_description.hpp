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
namespace pad_constant {
struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& xDesc_,
                          const TensorDescriptor& yDesc_,
                          const int64_t* padding_,
                          const int padding_size_ = 0)
        : xDesc(xDesc_), yDesc(yDesc_), padding(padding_), padding_size(padding_size_)
    {
        IsSameType();
        IsSameShape();
        IsValidPadding();
        IsValidIODims();
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    int64_t GetPaddingSize() const { return padding_size; }
    std::vector<int64_t> GetPadding() const { return {padding, padding + padding_size}; }

    NetworkConfig MakeNetworkConfig() const override;

    bool IsSameShape() const
    {
        if(xDesc.GetNumDims() != yDesc.GetNumDims())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadConstantFwd: Input and output tensors' dimensions don't match");
        return true;
    }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadConstantFwd: Input and output tensors' types don't match");

        return true;
    }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    bool IsValidPadding() const
    {
        if(padding_size % 2 != 0)
            MIOPEN_THROW(miopenStatusBadParm, "PadConstantFwd: Padding size must be even");

        if(xDesc.GetNumDims() < padding_size / 2)
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadConstantFwd: Padding size is larger than input's dimensions");

        return true;
    }

    bool IsValidIODims() const
    {
        auto input_dims = xDesc.GetLengths();
        std::vector<size_t> input_with_padding_dims(input_dims);
        for(uint64_t i = 0; i < padding_size / 2; i++)
        {
            int idx = input_dims.size() - i - 1;
            input_with_padding_dims[idx] += padding[i * 2] + padding[i * 2 + 1];
        }

        if(input_with_padding_dims != yDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadConstantFwd: Input + padding tensor and output tensor do not match");

        return true;
    }

    bool IsPadFirstDim() const
    {
        if(padding_size / 2 != xDesc.GetNumDims())
            return false;

        auto padding_vec = GetPadding();

        return padding_vec[padding_size - 1] != 0 || padding_vec[padding_size - 2] != 0;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const int64_t* padding;
    const int padding_size;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& dxDesc_,
                          const TensorDescriptor& dyDesc_,
                          const int64_t* padding_,
                          const int padding_size_ = 0)
        : dxDesc(dxDesc_), dyDesc(dyDesc_), padding(padding_), padding_size(padding_size_)
    {

        IsSameType();
        IsSameShape();
        IsValidPadding();
        IsValidIODims();
    }

    const TensorDescriptor& GetdXDesc() const { return dxDesc; }
    const TensorDescriptor& GetdYDesc() const { return dyDesc; }
    int64_t GetPaddingSize() const { return padding_size; }
    std::vector<int64_t> GetPadding() const { return {padding, padding + padding_size}; }

    NetworkConfig MakeNetworkConfig() const override;

    bool IsSameType() const
    {
        if(dyDesc.GetType() != dxDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadConstanhtBwd: Input grad and output grad tensors' types don't match");

        return true;
    }

    bool IsSameShape() const
    {
        if(dxDesc.GetNumDims() != dyDesc.GetNumDims())
            MIOPEN_THROW(
                miopenStatusBadParm,
                "PadConstantBwd: Input grad and output grad tensors' dimensions don't match: " +
                    std::to_string(dxDesc.GetNumDims()) + " vs " +
                    std::to_string(dyDesc.GetNumDims()));

        return true;
    }

    bool IsContiguous() const { return dyDesc.IsContiguous() && dxDesc.IsContiguous(); }

    bool IsValidPadding() const
    {
        if(padding_size % 2 != 0)
            MIOPEN_THROW(miopenStatusBadParm, "PadConstantBwd: Padding size must be even");

        if(dxDesc.GetNumDims() < padding_size / 2)
            MIOPEN_THROW(miopenStatusBadParm,
                         "PadConstantBwd: Padding size is larger than input grad's dimensions");

        return true;
    }

    bool IsValidIODims() const
    {
        auto input_dims = dxDesc.GetLengths();
        std::vector<size_t> input_with_padding_dims(input_dims);
        for(uint64_t i = 0; i < padding_size / 2; i++)
        {
            int idx = input_dims.size() - i - 1;
            input_with_padding_dims[idx] += padding[i * 2] + padding[i * 2 + 1];
        }

        if(input_with_padding_dims != dyDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "ConstantPadBwd: Input + padding tensor and output tensor do not match");

        return true;
    }

    bool IsOnlyPadFirstDim() const
    {
        if(padding_size / 2 != dxDesc.GetNumDims())
            return false;

        auto padding_vec = GetPadding();

        bool is_pad_first_dim =
            padding_vec[padding_size - 1] != 0 || padding_vec[padding_size - 2] != 0;
        bool is_remaining_zeros =
            std::all_of(padding_vec.begin(), padding_vec.end() - 2, [](int x) { return x == 0; });

        return is_pad_first_dim && is_remaining_zeros;
    }

private:
    const TensorDescriptor& dxDesc;
    const TensorDescriptor& dyDesc;
    const int64_t* padding;
    const int padding_size;
};

} // namespace pad_constant
} // namespace miopen
