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
    ProblemDescription(const TensorDescriptor& paramDesc_,
                       const TensorDescriptor& gradDesc_,
                       const TensorDescriptor& expAvgDesc_,
                       const TensorDescriptor& expAvgSqDesc_,
                       const TensorDescriptor& stepDesc_,
                       double lr_,
                       double beta1_,
                       double beta2_,
                       double weight_decay_,
                       double eps_,
                       bool amsgrad_,
                       bool maximize_,
                       bool amp_,
                       const TensorDescriptor* gradScaleDesc_,
                       const TensorDescriptor* foundInfDesc_)
        : paramDesc(paramDesc_),
          gradDesc(gradDesc_),
          expAvgDesc(expAvgDesc_),
          expAvgSqDesc(expAvgSqDesc_),
          stepDesc(stepDesc_),
          lr(lr_),
          beta1(beta1_),
          beta2(beta2_),
          weight_decay(weight_decay_),
          eps(eps_),
          amsgrad(amsgrad_),
          maximize(maximize_),
          is_amp(amp_)
    {
        if(is_amp)
        {
            gradScaleDesc = *gradScaleDesc_;
            foundInfDesc  = *foundInfDesc_;
        }
    }

    const TensorDescriptor& GetParamDesc() const { return paramDesc; }
    const TensorDescriptor& GetGradDesc() const { return gradDesc; }
    const TensorDescriptor& GetExpAvgDesc() const { return expAvgDesc; }
    const TensorDescriptor& GetExpAvgSqsDesc() const { return expAvgSqDesc; }
    const TensorDescriptor& GetStepDesc() const { return stepDesc; }
    const TensorDescriptor& GetGradScaleDesc() const { return gradScaleDesc; }
    const TensorDescriptor& GetFoundInfDesc() const { return foundInfDesc; }
    double GetLr() const { return lr; }
    double GetBeta1() const { return beta1; }
    double GetBeta2() const { return beta2; }
    double GetWeight_decay() const { return weight_decay; }
    double GetEps() const { return eps; }
    bool GetAmsgrad() const { return amsgrad; }
    bool GetMaximize() const { return maximize; }

    bool IsAmp() const { return is_amp; }
    bool IsValidType() const { return true; }

    bool IsAllPacked() const
    {
        if(!(paramDesc.IsPacked() && gradDesc.IsPacked() && expAvgDesc.IsPacked() &&
             expAvgSqDesc.IsPacked() && stepDesc.IsPacked() && gradScaleDesc.IsPacked() &&
             foundInfDesc.IsPacked()))
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor paramDesc{};
    TensorDescriptor gradDesc{};
    TensorDescriptor expAvgDesc{};
    TensorDescriptor expAvgSqDesc{};
    TensorDescriptor stepDesc{};
    TensorDescriptor gradScaleDesc{};
    TensorDescriptor foundInfDesc{};

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
