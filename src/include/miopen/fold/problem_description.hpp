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

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace fold {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

struct UnfoldFwdProblemDescription : ProblemDescriptionBase
{
    UnfoldFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                const TensorDescriptor& outputDesc_,
                                const uint64_t* kernel_size_,
                                const uint64_t kernel_size_size_,
                                const uint64_t* stride_,
                                const uint64_t stride_size_,
                                const uint64_t* padding_,
                                const uint64_t padding_size_,
                                const uint64_t* dilation_,
                                const uint64_t dilation_size_)
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
        IsValidSize();
        IsValidType();
    }

    bool IsValidSize() const
    {
        if(inputDesc.GetNumDims() != 4)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Unfold: The input tensor should be 4D.");
#else
            return false;
#endif
        }
        uint64_t spatial_dim_size = inputDesc.GetNumDims() - 2;
        if(kernel_size_size != spatial_dim_size || stride_size != spatial_dim_size ||
           padding_size != spatial_dim_size || dilation_size != spatial_dim_size)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Unfold: Argument length should be 2D");
        }
        auto input_dims  = inputDesc.GetLengths();
        const uint64_t N = static_cast<uint64_t>(input_dims[0]);
        const uint64_t C = static_cast<uint64_t>(input_dims[1]);
        uint64_t P = 1, L = 1;
        std::vector<uint64_t> ls;
        for(uint64_t i = 0; i < spatial_dim_size; ++i)
        {
            P *= kernel_size[i];
            uint64_t l = (static_cast<uint64_t>(input_dims[i + 2]) + 2 * padding[i] -
                          dilation[i] * (kernel_size[i] - 1) - 1) /
                             stride[i] +
                         1;
            L *= l;
            ls.push_back(l);
        }
        std::vector<size_t> output_dims_desired{
            static_cast<size_t>(N), static_cast<size_t>(C * P), static_cast<size_t>(L)};
        auto output_dims = outputDesc.GetLengths();
        if(output_dims != output_dims_desired)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Unfold: Invalid output dimension");
        }
        return true;
    }

    bool IsValidType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Unfold: The input tensor and output tensor has mismatch type.");
        }
        return true;
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    const uint64_t* kernel_size;
    const uint64_t kernel_size_size;
    const uint64_t* stride;
    const uint64_t stride_size;
    const uint64_t* padding;
    const uint64_t padding_size;
    const uint64_t* dilation;
    const uint64_t dilation_size;
};

struct UnfoldBwdProblemDescription : ProblemDescriptionBase
{
    UnfoldBwdProblemDescription(const TensorDescriptor& dinputDesc_,
                                const TensorDescriptor& doutputDesc_,
                                const uint64_t* kernel_size_,
                                const uint64_t kernel_size_size_,
                                const uint64_t* stride_,
                                const uint64_t stride_size_,
                                const uint64_t* padding_,
                                const uint64_t padding_size_,
                                const uint64_t* dilation_,
                                const uint64_t dilation_size_)
        : dinputDesc(dinputDesc_),
          doutputDesc(doutputDesc_),
          kernel_size(kernel_size_),
          kernel_size_size(kernel_size_size_),
          stride(stride_),
          stride_size(stride_size_),
          padding(padding_),
          padding_size(padding_size_),
          dilation(dilation_),
          dilation_size(dilation_size_)
    {
        IsValidSize();
        IsValidType();
    }

    bool IsValidSize() const
    {
        if(dinputDesc.GetNumDims() != 4)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Unfold: The input gradient tensor should be 4D.");
        }
        uint64_t spatial_dim_size = dinputDesc.GetNumDims() - 2;
        if(kernel_size_size != spatial_dim_size || stride_size != spatial_dim_size ||
           padding_size != spatial_dim_size || dilation_size != spatial_dim_size)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Unfold: Argument length should be 2D");
        }
        auto input_dims  = dinputDesc.GetLengths();
        const uint64_t N = static_cast<uint64_t>(input_dims[0]);
        const uint64_t C = static_cast<uint64_t>(input_dims[1]);
        uint64_t P = 1, L = 1;
        std::vector<uint64_t> ls;
        for(uint64_t i = 0; i < spatial_dim_size; ++i)
        {
            P *= kernel_size[i];
            uint64_t l = (static_cast<uint64_t>(input_dims[i + 2]) + 2 * padding[i] -
                          dilation[i] * (kernel_size[i] - 1) - 1) /
                             stride[i] +
                         1;
            L *= l;
            ls.push_back(l);
        }
        std::vector<size_t> output_dims_desired{
            static_cast<size_t>(N), static_cast<size_t>(C * P), static_cast<size_t>(L)};
        auto output_dims = doutputDesc.GetLengths();
        if(output_dims != output_dims_desired)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Unfold: Invalid output gradient dimension");
        }
        return true;
    }

    bool IsValidType() const
    {
        if(dinputDesc.GetType() != doutputDesc.GetType())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Unfold: The input gradient tensor and output gradient tensor has mismatch type.");
        }
        return true;
    }

    const TensorDescriptor& GetDinputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetDoutputDesc() const { return doutputDesc; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor dinputDesc;
    TensorDescriptor doutputDesc;
    const uint64_t* kernel_size;
    const uint64_t kernel_size_size;
    const uint64_t* stride;
    const uint64_t stride_size;
    const uint64_t* padding;
    const uint64_t padding_size;
    const uint64_t* dilation;
    const uint64_t dilation_size;
};

struct FoldFwdProblemDescription : ProblemDescriptionBase
{
    FoldFwdProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& outputDesc_,
                              const uint64_t* kernel_size_,
                              const uint64_t kernel_size_size_,
                              const uint64_t* stride_,
                              const uint64_t stride_size_,
                              const uint64_t* padding_,
                              const uint64_t padding_size_,
                              const uint64_t* dilation_,
                              const uint64_t dilation_size_)
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
        IsValidSize();
        IsValidType();
    }

    bool IsValidSize() const
    {
        if(outputDesc.GetNumDims() != 4)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Fold: The output tensor should be 4D.");
        }
        uint64_t spatial_dim_size = outputDesc.GetNumDims() - 2;
        if(kernel_size_size != spatial_dim_size || stride_size != spatial_dim_size ||
           padding_size != spatial_dim_size || dilation_size != spatial_dim_size)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Fold: Argument length should be 2D");
        }
        auto input_dims  = inputDesc.GetLengths();
        auto output_dims = outputDesc.GetLengths();
        const uint64_t N = static_cast<uint64_t>(output_dims[0]);
        const uint64_t C = static_cast<uint64_t>(output_dims[1]);
        uint64_t P = 1, L = 1;
        std::vector<uint64_t> ls;
        for(uint64_t i = 0; i < spatial_dim_size; ++i)
        {
            P *= kernel_size[i];
            uint64_t l = (static_cast<uint64_t>(output_dims[i + 2]) + 2 * padding[i] -
                          dilation[i] * (kernel_size[i] - 1) - 1) /
                             stride[i] +
                         1;
            L *= l;
            ls.push_back(l);
        }
        std::vector<size_t> input_dims_desired{
            static_cast<size_t>(N), static_cast<size_t>(C * P), static_cast<size_t>(L)};
        if(input_dims != input_dims_desired)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Fold: Invalid input dimension");
        }
        return true;
    }

    bool IsValidType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Fold: The input tensor and output tensor has mismatch type.");
        }
        return true;
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    const uint64_t* kernel_size;
    const uint64_t kernel_size_size;
    const uint64_t* stride;
    const uint64_t stride_size;
    const uint64_t* padding;
    const uint64_t padding_size;
    const uint64_t* dilation;
    const uint64_t dilation_size;
};

struct FoldBwdProblemDescription : ProblemDescriptionBase
{
    FoldBwdProblemDescription(const TensorDescriptor& dinputDesc_,
                              const TensorDescriptor& doutputDesc_,
                              const uint64_t* kernel_size_,
                              const uint64_t kernel_size_size_,
                              const uint64_t* stride_,
                              const uint64_t stride_size_,
                              const uint64_t* padding_,
                              const uint64_t padding_size_,
                              const uint64_t* dilation_,
                              const uint64_t dilation_size_)
        : dinputDesc(dinputDesc_),
          doutputDesc(doutputDesc_),
          kernel_size(kernel_size_),
          kernel_size_size(kernel_size_size_),
          stride(stride_),
          stride_size(stride_size_),
          padding(padding_),
          padding_size(padding_size_),
          dilation(dilation_),
          dilation_size(dilation_size_)
    {
        IsValidSize();
        IsValidType();
    }

    bool IsValidSize() const
    {
        if(doutputDesc.GetNumDims() != 4)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Fold: The output gradient tensor should be 4D.");
        }
        uint64_t spatial_dim_size = doutputDesc.GetNumDims() - 2;
        if(kernel_size_size != spatial_dim_size || stride_size != spatial_dim_size ||
           padding_size != spatial_dim_size || dilation_size != spatial_dim_size)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Fold: Argument length should be 2D");
        }
        auto input_dims  = dinputDesc.GetLengths();
        auto output_dims = doutputDesc.GetLengths();
        const uint64_t N = static_cast<uint64_t>(output_dims[0]);
        const uint64_t C = static_cast<uint64_t>(output_dims[1]);
        uint64_t P = 1, L = 1;
        std::vector<uint64_t> ls;
        for(uint64_t i = 0; i < spatial_dim_size; ++i)
        {
            P *= kernel_size[i];
            uint64_t l = (static_cast<uint64_t>(output_dims[i + 2]) + 2 * padding[i] -
                          dilation[i] * (kernel_size[i] - 1) - 1) /
                             stride[i] +
                         1;
            L *= l;
            ls.push_back(l);
        }
        std::vector<size_t> input_dims_desired{
            static_cast<size_t>(N), static_cast<size_t>(C * P), static_cast<size_t>(L)};
        if(input_dims != input_dims_desired)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Fold: Invalid input gradient dimension");
        }
        return true;
    }

    bool IsValidType() const
    {
        if(dinputDesc.GetType() != doutputDesc.GetType())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Fold: The input gradient tensor and output gradient tensor has mismatch type.");
        }
        return true;
    }

    const TensorDescriptor& GetDinputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetDoutputDesc() const { return doutputDesc; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor dinputDesc;
    TensorDescriptor doutputDesc;
    const uint64_t* kernel_size;
    const uint64_t kernel_size_size;
    const uint64_t* stride;
    const uint64_t stride_size;
    const uint64_t* padding;
    const uint64_t padding_size;
    const uint64_t* dilation;
    const uint64_t dilation_size;
};

} // namespace fold

} // namespace miopen
