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
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace softmax {

int nextPow2(int v);

void getParams(const TensorDescriptor& in_yDesc,
               miopenSoftmaxMode_t in_mode,
               int& out_n,
               int& out_c,
               int& out_h,
               int& out_w,
               int& out_grid_size,
               int& out_spatial_dim,
               int& out_vector_size,
               int& out_num_batch,
               bool& out_usefp16,
               bool& out_usefp32,
               std::vector<size_t>& out_vld,
               std::vector<size_t>& out_vgd,
               size_t& out_workgroups,
               int& out_batch_size,
               int& out_u_batch_size);

struct ProblemDescription : ProblemDescriptionBase
{
    // softmax forward constructor
    ProblemDescription(const void* alpha_,
                       const void* beta_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       miopenSoftmaxAlgorithm_t algorithm_,
                       miopenSoftmaxMode_t mode_)
        : isForward(true),
          alpha(alpha_),
          beta(beta_),
          xdxDesc(xDesc_),
          yDesc(yDesc_),

          // initalize this reference to some value, but it will not be used for Forward
          dyDesc(yDesc_),

          algorithm(algorithm_),
          mode(mode_)
    {
        CheckCommonParams();

        if(xdxDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(xdxDesc.GetLengths() != yDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }
    }

    ProblemDescription(const void* alpha_,
                       const void* beta_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& dyDesc_,
                       const TensorDescriptor& dxDesc_,
                       miopenSoftmaxAlgorithm_t algorithm_,
                       miopenSoftmaxMode_t mode_)
        : isForward(false),
          alpha(alpha_),
          beta(beta_),
          xdxDesc(dxDesc_),
          yDesc(yDesc_),
          dyDesc(dyDesc_),
          algorithm(algorithm_),
          mode(mode_)
    {
        CheckCommonParams();

        if(yDesc != dyDesc)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        if(xdxDesc.GetType() != dyDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(xdxDesc.GetLengths() != dyDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }
    }

    bool IsForward() const { return isForward; }
    const miopenSoftmaxAlgorithm_t GetAlgorithm() const { return algorithm; }
    const miopenSoftmaxMode_t GetMode() const { return mode; }
    const void* GetAlpha() const { return alpha; }
    const void* GetBeta() const { return beta; }

    // for forward
    const TensorDescriptor& GetXDesc() const { return xdxDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    // for backward
    const TensorDescriptor& GetdYDesc() const { return dyDesc; }
    const TensorDescriptor& GetdXDesc() const { return xdxDesc; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    void CheckCommonParams()
    {
        if(alpha == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Alpha value is nullptr");
        }

        if(beta == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Beta value is nullptr");
        }
    }

private:
    const bool isForward;
    const void* alpha;
    const void* beta;

    // for forward xDesc is stored in xdxDesc, for backward dxDesc is stored in xdxDesc
    const TensorDescriptor& xdxDesc;
    const TensorDescriptor& yDesc;

    const TensorDescriptor& dyDesc;

    const miopenSoftmaxAlgorithm_t algorithm;
    const miopenSoftmaxMode_t mode;
};

} // namespace softmax
} // namespace miopen
