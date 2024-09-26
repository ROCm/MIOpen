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
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace sigmoidfocalloss {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

struct SigmoidFocalLossProblemDescription : ProblemDescriptionBase
{
    SigmoidFocalLossProblemDescription(const TensorDescriptor& inputDesc_,
                                       const TensorDescriptor& targetDesc_,
                                       const miopenLossReductionMode_t reduction_)
        : inputDesc(inputDesc_), targetDesc(targetDesc_), reduction(reduction_)
    {
        if(!checkSameLength(inputDesc, targetDesc))
            MIOPEN_THROW(miopenStatusBadParm,
                         "SigmoidFocalLoss: Input, target tensor sizes do not match.");
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }

public:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    miopenLossReductionMode_t reduction;
};

struct SigmoidFocalLossFwdProblemDescription : SigmoidFocalLossProblemDescription
{
    SigmoidFocalLossFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                          const TensorDescriptor& targetDesc_,
                                          const TensorDescriptor& outputDesc_,
                                          const miopenLossReductionMode_t reduction_)
        : SigmoidFocalLossProblemDescription(inputDesc_, targetDesc_, reduction_),
          outputDesc(outputDesc_)
    {
        miopenDataType_t dtype = inputDesc.GetType();
        if(dtype != targetDesc.GetType() || dtype != outputDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SigmoidFocalLoss: Input, target, output tensor type do not match.");
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

public:
    TensorDescriptor outputDesc;
};

struct SigmoidFocalLossBwdProblemDescription : SigmoidFocalLossProblemDescription
{
    SigmoidFocalLossBwdProblemDescription(const TensorDescriptor& inputDesc_,
                                          const TensorDescriptor& targetDesc_,
                                          const TensorDescriptor& doutputDesc_,
                                          const TensorDescriptor& dinputDesc_,
                                          const TensorDescriptor& dtargetDesc_,
                                          const miopenLossReductionMode_t reduction_)
        : SigmoidFocalLossProblemDescription(inputDesc_, targetDesc_, reduction_),
          doutputDesc(doutputDesc_),
          dinputDesc(dinputDesc_),
          dtargetDesc(dtargetDesc_)
    {
        miopenDataType_t dtype = inputDesc.GetType();
        if(dtype != targetDesc.GetType() || dtype != doutputDesc.GetType() ||
           dtype != dinputDesc.GetType() || dtype != dtargetDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SigmoidFocalLoss: Input, target, doutput, dinput, dtarget tensor type do "
                         "not match.");
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetDoutputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetDinputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetDtargetDesc() const { return dtargetDesc; }

public:
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;
    TensorDescriptor dtargetDesc;
};

} // namespace sigmoidfocalloss

} // namespace miopen
