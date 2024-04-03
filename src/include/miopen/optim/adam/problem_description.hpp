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
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace adam {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& paramInDesc_,
                       const TensorDescriptor& paramOutDesc_,
                       const TensorDescriptor* paramOutFloat16Desc_,
                       const TensorDescriptor& gradInDesc_,
                       const TensorDescriptor& expAvgInDesc_,
                       const TensorDescriptor& expAvgOutDesc_,
                       const TensorDescriptor& expAvgSqInDesc_,
                       const TensorDescriptor& expAvgSqOutDesc_,
                       const TensorDescriptor* maxExpAvgSqInDesc_,
                       const TensorDescriptor* maxExpAvgSqOutDesc_,
                       const TensorDescriptor* gradScaleDesc_,
                       const TensorDescriptor* foundInfDesc_,
                       const TensorDescriptor* stepInDesc_,
                       const TensorDescriptor* stepOutDesc_,
                       int32_t step_,
                       double lr_,
                       double beta1_,
                       double beta2_,
                       double weight_decay_,
                       double eps_,
                       bool amsgrad_,
                       bool maximize_)
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
          maximize(maximize_)
    {
        if(gradScaleDesc != nullptr || foundInfDesc != nullptr)
            is_amp = true;

        if(is_amp && (stepInDesc == nullptr || stepOutDesc == nullptr))
            MIOPEN_THROW(miopenStatusBadParm,
                         "AmpAdam: In amp adam, State step tensor is required.");

        if(amsgrad && (maxExpAvgSqInDesc == nullptr || maxExpAvgSqOutDesc == nullptr))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Adam: In the amsgrad, the max_exp_avg_sq tensor is required.");
        }

        auto dtype = paramInDesc.GetType();
        if((paramOutDesc.GetType() != dtype) || (expAvgInDesc.GetType() != dtype) ||
           (expAvgOutDesc.GetType() != dtype) || (expAvgSqInDesc.GetType() != dtype) ||
           (expAvgSqOutDesc.GetType() != dtype) ||
           (maxExpAvgSqInDesc != nullptr && maxExpAvgSqInDesc->GetType() != dtype) ||
           (maxExpAvgSqOutDesc != nullptr && maxExpAvgSqOutDesc->GetType() != dtype))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: Tensor types do not match.");
        }

        if((dtype == miopenBFloat16) || (gradInDesc.GetType() == miopenBFloat16))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: bfloat16 type is not supported.");
        }

        if((paramOutFloat16Desc != nullptr) && (paramOutFloat16Desc->GetType() != miopenHalf))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: Invalid type of param_out_float16.");
        }

        auto numel = paramInDesc.GetElementSize();
        if((paramOutDesc.GetElementSize() != numel) || (expAvgInDesc.GetElementSize() != numel) ||
           (expAvgOutDesc.GetElementSize() != numel) ||
           (expAvgSqInDesc.GetElementSize() != numel) ||
           (expAvgSqOutDesc.GetElementSize() != numel) ||
           (paramOutFloat16Desc != nullptr && paramOutFloat16Desc->GetElementSize() != numel) ||
           (maxExpAvgSqInDesc != nullptr && maxExpAvgSqInDesc->GetElementSize() != numel) ||
           (maxExpAvgSqOutDesc != nullptr && maxExpAvgSqOutDesc->GetElementSize() != numel))
        {
            MIOPEN_THROW(miopenStatusBadParm, "Adam: Tensor dimension lengths do not match.");
        }
    }

    const TensorDescriptor& GetParamDesc() const { return paramInDesc; }
    const TensorDescriptor& GetGradDesc() const { return gradInDesc; }
    bool ExistStepOut() const { return (stepOutDesc != nullptr); }
    bool IsAmp() const { return is_amp; }
    bool IsAllPacked() const
    {
        if(!(paramInDesc.IsPacked() && gradInDesc.IsPacked() && expAvgInDesc.IsPacked() &&
             expAvgSqInDesc.IsPacked()))
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor paramInDesc{};
    TensorDescriptor paramOutDesc{};
    TensorDescriptor gradInDesc{};
    TensorDescriptor expAvgInDesc{};
    TensorDescriptor expAvgOutDesc{};
    TensorDescriptor expAvgSqInDesc{};
    TensorDescriptor expAvgSqOutDesc{};
    const TensorDescriptor* paramOutFloat16Desc = nullptr;
    const TensorDescriptor* maxExpAvgSqInDesc   = nullptr;
    const TensorDescriptor* maxExpAvgSqOutDesc  = nullptr;
    const TensorDescriptor* gradScaleDesc       = nullptr;
    const TensorDescriptor* foundInfDesc        = nullptr;
    const TensorDescriptor* stepInDesc          = nullptr;
    const TensorDescriptor* stepOutDesc         = nullptr;

    int32_t step        = 0;
    double lr           = 0.0;
    double beta1        = 0.0;
    double beta2        = 0.0;
    double weight_decay = 0.0;
    double eps          = 0.0;
    bool amsgrad        = false;
    bool maximize       = false;
    bool is_amp         = false;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace adam

} // namespace miopen
