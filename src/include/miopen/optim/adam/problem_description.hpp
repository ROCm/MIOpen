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
                       const TensorDescriptor& gradInDesc_,
                       const TensorDescriptor& expAvgInDesc_,
                       const TensorDescriptor& expAvgSqInDesc_,
                       const TensorDescriptor* maxExpAvgSqInDesc_,
                       const TensorDescriptor* gradScaleDesc_,
                       const TensorDescriptor* foundInfDesc_,
                       const TensorDescriptor* stepInDesc_,
                       int32_t step_,
                       double lr_,
                       double beta1_,
                       double beta2_,
                       double weight_decay_,
                       double eps_,
                       bool amsgrad_,
                       bool maximize_,
                       const TensorDescriptor* paramOutDesc_,
                       const TensorDescriptor* expAvgOutDesc_,
                       const TensorDescriptor* expAvgSqOutDesc_,
                       const TensorDescriptor* maxExpAvgSqOutDesc_,
                       const TensorDescriptor* stepOutDesc_)
        : paramInDesc(paramInDesc_),
          gradInDesc(gradInDesc_),
          expAvgInDesc(expAvgInDesc_),
          expAvgSqInDesc(expAvgSqInDesc_),
          maxExpAvgSqInDesc(maxExpAvgSqInDesc_),
          gradScaleDesc(gradScaleDesc_),
          foundInfDesc(foundInfDesc_),
          stepInDesc(stepInDesc_),
          step(step_),
          lr(lr_),
          beta1(beta1_),
          beta2(beta2_),
          weight_decay(weight_decay_),
          eps(eps_),
          amsgrad(amsgrad_),
          maximize(maximize_),
          paramOutDesc(paramOutDesc_),
          expAvgOutDesc(expAvgOutDesc_),
          expAvgSqOutDesc(expAvgSqOutDesc_),
          maxExpAvgSqOutDesc(maxExpAvgSqOutDesc_),
          stepOutDesc(stepOutDesc_)
    {
        if(gradScaleDesc != nullptr || foundInfDesc != nullptr)
            is_amp = true;
    }

    const TensorDescriptor& GetParamDesc() const { return paramInDesc; }
    const TensorDescriptor& GetGradDesc() const { return gradInDesc; }
    double GetLr() const { return lr; }
    double GetBeta1() const { return beta1; }
    double GetBeta2() const { return beta2; }
    double GetWeight_decay() const { return weight_decay; }
    double GetEps() const { return eps; }
    bool GetAmsgrad() const { return amsgrad; }
    bool GetMaximize() const { return maximize; }
    bool ExistStepOut() const { return (stepOutDesc != nullptr); }

    bool IsAmp() const { return is_amp; }
    bool IsValidType() const { return true; }

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
    TensorDescriptor gradInDesc{};
    TensorDescriptor expAvgInDesc{};
    TensorDescriptor expAvgSqInDesc{};
    const TensorDescriptor* maxExpAvgSqInDesc = nullptr;
    const TensorDescriptor* gradScaleDesc     = nullptr;
    const TensorDescriptor* foundInfDesc      = nullptr;
    const TensorDescriptor* stepInDesc        = nullptr;

    int32_t step        = 0;
    double lr           = 0.0;
    double beta1        = 0.0;
    double beta2        = 0.0;
    double weight_decay = 0.0;
    double eps          = 0.0;
    bool amsgrad        = false;
    bool maximize       = false;
    bool is_amp         = false;

    const TensorDescriptor* paramOutDesc       = nullptr;
    const TensorDescriptor* expAvgOutDesc      = nullptr;
    const TensorDescriptor* expAvgSqOutDesc    = nullptr;
    const TensorDescriptor* maxExpAvgSqOutDesc = nullptr;
    const TensorDescriptor* stepOutDesc        = nullptr;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace adam

} // namespace miopen
