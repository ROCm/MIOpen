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

#include <string>

namespace miopen {

struct NetworkConfig;

namespace embedding {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescription : ProblemDescriptionBase
{
    // forward
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& outputDesc_,
                       const TensorDescriptor& errorDesc_,
                       bool has_max_norm_,
                       float max_norm_,
                       float norm_type_)
        : direction(Direction::Forward),
          inputDesc(inputDesc_),
          weightDesc(weightDesc_),
          outputDesc(outputDesc_),
          errorDesc(errorDesc_),
          has_max_norm(has_max_norm_),
          max_norm(max_norm_),
          norm_type(norm_type_)
    {
        auto& input_dims  = inputDesc.GetLengths();
        auto& weight_dims = weightDesc.GetLengths();
        auto& output_dims = outputDesc.GetLengths();

        if((input_dims.size() + 1) != output_dims.size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Embedding: Input and output tensor dimention mismatch");
        }

        for(size_t i = 0; i < input_dims.size(); ++i)
        {
            if(input_dims[i] != output_dims[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Embedding: Input and output tensor size mismatch");
            }
        }
        if(weight_dims.size() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Embedding: Weight tensor should be 2D");
        }

        if(weight_dims.back() != output_dims.back())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Embedding: Weight and output tensor size mismatch");
        }

        if(has_max_norm && weightDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Embedding: The dtype of weight tensor should not be int64 when applying max_norm");
        }
    }

    // backward
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& outputDesc_,
                       const TensorDescriptor& errorDesc_,
                       int64_t padding_idx_,
                       bool scale_grad_by_freq_)
        : direction(Direction::Backward),
          inputDesc(inputDesc_),
          weightDesc(weightDesc_),
          outputDesc(outputDesc_),
          errorDesc(errorDesc_),
          padding_idx(padding_idx_),
          scale_grad_by_freq(scale_grad_by_freq_)
    {
        auto& input_dims  = inputDesc.GetLengths();
        auto& weight_dims = weightDesc.GetLengths();
        auto& output_dims = outputDesc.GetLengths();

        if((input_dims.size() + 1) != output_dims.size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Embedding: Input and output tensor dimention mismatch");
        }

        for(size_t i = 0; i < input_dims.size(); ++i)
        {
            if(input_dims[i] != output_dims[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Embedding: Input and output tensor size mismatch");
            }
        }
        if(weight_dims.size() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Embedding: Weight tensor should be 2D");
        }

        if(weight_dims.back() != output_dims.back())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Embedding: Weight and output tensor size mismatch");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    size_t GetEmbeddingDim() const { return outputDesc.GetLengths().back(); }
    size_t GetNumEmbeddings() const { return weightDesc.GetLengths().front(); }
    bool IsForward() const { return direction == Direction::Forward; }
    bool ScaleGradByFreq() const { return scale_grad_by_freq; }
    bool HasMaxNorm() const { return has_max_norm; }
    bool IsDeterministicMode() const { return deterministic_mode; };
    bool UseTraverseOpt() const { return (GetNumEmbeddings() <= 32); };
    /*
    bool IsAllContigous() const
    {
        if(inputDesc.IsContiguous() && weightDesc.IsContiguous() && outputDesc.IsContiguous())
            return true;
        return false;
    }
    */

    NetworkConfig MakeNetworkConfig() const override;

private:
    Direction direction;
    TensorDescriptor inputDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor errorDesc;

    int64_t padding_idx     = 0;
    bool scale_grad_by_freq = false;
    bool has_max_norm       = false;
    float max_norm          = 0.0;
    float norm_type         = 0.0;
    bool deterministic_mode = false;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace embedding

} // namespace miopen
