/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace batchnorm {

enum class Direction
{
    ForwardTraining,
    ForwardInference,
    Backward,
};

struct ProblemDescription
{
    ProblemDescription(Direction direction_,
                       miopenBatchNormMode_t bn_mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& bnScaleBiasMeanVarDesc_,
                       double expAvgFactor_,
                       double epsilon_,
                       bool resultsave_,
                       bool resultrunning_)
        : direction(direction_),
          bn_mode(bn_mode_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          bnScaleBiasMeanVarDesc(bnScaleBiasMeanVarDesc_),
          expAvgFactor(expAvgFactor_),
          epsilon(epsilon_),
          resultsave(resultsave_),
          resultrunning(resultrunning_)
    {
    }

    Direction GetDirection() const { return direction; }
    miopenBatchNormMode_t GetMode() const { return bn_mode; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetBnScaleBiasMeanVarDesc() const { return bnScaleBiasMeanVarDesc; }
    bool GetResultSave() const { return resultsave; }
    bool GetResultRunning() const { return resultrunning; }

    NetworkConfig MakeNetworkConfig() const;

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    private:
    Direction direction;
    miopenBatchNormMode_t bn_mode;
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    TensorDescriptor bnScaleBiasMeanVarDesc;
    double expAvgFactor;
    double epsilon;
    bool resultsave;
    bool resultrunning;
};

} // namespace batchnorm

} // namespace miopen
