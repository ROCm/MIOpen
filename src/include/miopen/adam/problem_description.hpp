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

namespace adam {

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& paramInDesc_,
                       const TensorDescriptor& paramOutDesc_,
                       const TensorDescriptor& paramOutFloat16Desc_,
                       const TensorDescriptor& gradInDesc_,
                       const TensorDescriptor& expAvgInDesc_,
                       const TensorDescriptor& expAvgOutDesc_,
                       const TensorDescriptor& expAvgSqInDesc_,
                       const TensorDescriptor& expAvgSqOutDesc_,
                       const TensorDescriptor& maxExpAvgSqInDesc_,
                       const TensorDescriptor& maxExpAvgSqOutDesc_,
                       const TensorDescriptor& gradScaleDesc_,
                       const TensorDescriptor& foundInfDesc_,
                       const TensorDescriptor& stepInDesc_,
                       const TensorDescriptor& stepOutDesc_,
                       uint32_t step_,
                       double lr_,
                       double beta1_,
                       double beta2_,
                       double weight_decay_,
                       double eps_,
                       bool amsgrad_,
                       bool maximize_,
                       bool adamw_,
                       bool is_amp_)
        : paramInDesc(paramInDesc_),
          paramOutDesc(paramOutDesc_),
          gradInDesc(gradInDesc_),
          expAvgInDesc(expAvgInDesc_),
          expAvgOutDesc(expAvgOutDesc_),
          expAvgSqInDesc(expAvgSqInDesc_),
          expAvgSqOutDesc(expAvgSqOutDesc_),
          paramOutFloat16Desc(paramOutFloat16Desc_),
          maxExpAvgSqInDesc(maxExpAvgSqInDesc_),
          maxExpAvgSqOutDesc(maxExpAvgSqOutDesc_),
          gradScaleDesc(gradScaleDesc_),
          foundInfDesc(foundInfDesc_),
          stepInDesc(stepInDesc_),
          stepOutDesc(stepOutDesc_),
          step(step_),
          lr(lr_),
          beta1(beta1_),
          beta2(beta2_),
          weight_decay(weight_decay_),
          eps(eps_),
          amsgrad(amsgrad_),
          maximize(maximize_),
          adamw(adamw_),
          is_amp(is_amp_)
    {
        if(amsgrad &&
           (maxExpAvgSqInDesc.GetLengths().empty() || maxExpAvgSqOutDesc.GetLengths().empty()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Adam: In the amsgrad, the max_exp_avg_sq tensor is required.");
        }

        auto dtype = paramInDesc.GetType();

        if((dtype == miopenBFloat16) || (gradInDesc.GetType() == miopenBFloat16))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: bfloat16 type is not supported.");
        }

        if((paramOutDesc.GetType() != dtype) || (!is_amp && gradInDesc.GetType() != dtype) ||
           (expAvgInDesc.GetType() != dtype) || (expAvgOutDesc.GetType() != dtype) ||
           (expAvgSqInDesc.GetType() != dtype) || (expAvgSqOutDesc.GetType() != dtype) ||
           (!maxExpAvgSqInDesc.GetLengths().empty() && maxExpAvgSqInDesc.GetType() != dtype) ||
           (!maxExpAvgSqOutDesc.GetLengths().empty() && maxExpAvgSqOutDesc.GetType() != dtype))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: Tensor types do not match.");
        }

        if(is_amp && !paramOutFloat16Desc.GetLengths().empty() &&
           (paramOutFloat16Desc.GetType() != miopenHalf))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: Invalid type of param_out_float16.");
        }

        auto numel = paramInDesc.GetElementSize();
        if((paramOutDesc.GetElementSize() != numel) || (gradInDesc.GetElementSize() != numel) ||
           (expAvgInDesc.GetElementSize() != numel) || (expAvgOutDesc.GetElementSize() != numel) ||
           (expAvgSqInDesc.GetElementSize() != numel) ||
           (expAvgSqOutDesc.GetElementSize() != numel) ||
           (is_amp && !paramOutFloat16Desc.GetLengths().empty() &&
            paramOutFloat16Desc.GetElementSize() != numel) ||
           (!maxExpAvgSqInDesc.GetLengths().empty() &&
            maxExpAvgSqInDesc.GetElementSize() != numel) ||
           (!maxExpAvgSqOutDesc.GetLengths().empty() &&
            maxExpAvgSqOutDesc.GetElementSize() != numel))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: Tensor dimension lengths do not match.");
        }
    }

    const TensorDescriptor& GetParamDesc() const { return paramInDesc; }
    const TensorDescriptor& GetGradDesc() const { return gradInDesc; }
    bool ExistStepTensor() const { return !stepInDesc.GetLengths().empty(); }
    bool IsAmp() const { return is_amp; }
    bool IsAdamW() const { return adamw; }
    bool IsAllPacked() const
    {
        if(!(paramInDesc.IsPacked() && gradInDesc.IsPacked() && expAvgInDesc.IsPacked() &&
             expAvgSqInDesc.IsPacked()))
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor paramInDesc;
    TensorDescriptor paramOutDesc;
    TensorDescriptor gradInDesc;
    TensorDescriptor expAvgInDesc;
    TensorDescriptor expAvgOutDesc;
    TensorDescriptor expAvgSqInDesc;
    TensorDescriptor expAvgSqOutDesc;
    TensorDescriptor paramOutFloat16Desc;
    TensorDescriptor maxExpAvgSqInDesc;
    TensorDescriptor maxExpAvgSqOutDesc;
    TensorDescriptor gradScaleDesc;
    TensorDescriptor foundInfDesc;
    TensorDescriptor stepInDesc;
    TensorDescriptor stepOutDesc;

    uint32_t step       = 0;
    double lr           = 0.0;
    double beta1        = 0.0;
    double beta2        = 0.0;
    double weight_decay = 0.0;
    double eps          = 0.0;
    bool amsgrad        = false;
    bool maximize       = false;
    bool adamw          = false;
    bool is_amp         = false;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace adam

} // namespace miopen
