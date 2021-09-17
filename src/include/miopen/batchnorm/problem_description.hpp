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

#include <cassert>
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
    // Forward
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
          yOrDyDesc(yDesc_),
          scaleBiasDesc(bnScaleBiasMeanVarDesc_),
          expAvgFactor(expAvgFactor_),
          epsilon(epsilon_),
          resultsave(resultsave_),
          resultrunning(resultrunning_)
    {
    }

    // Backward
    ProblemDescription(miopenBatchNormMode_t bn_mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& dyDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& bnScaleBiasDiffDesc_,
                       double epsilon_,
                       bool useSaved_)
        : direction(Direction::Backward),
          bn_mode(bn_mode_),
          xDesc(xDesc_),
          yOrDyDesc(dyDesc_),
          dxDesc(dxDesc_),
          scaleBiasDesc(bnScaleBiasDiffDesc_),
          epsilon(epsilon_),
          useSaved(useSaved_)
    {
    }

    Direction GetDirection() const { return direction; }
    miopenBatchNormMode_t GetMode() const { return bn_mode; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }

    const TensorDescriptor& GetYDesc() const
    {
        assert(direction == Direction::ForwardTraining || direction == Direction::ForwardInference);
        return yOrDyDesc;
    }

    const TensorDescriptor& GetDYDesc() const
    {
        assert(direction == Direction::Backward);
        return yOrDyDesc;
    }

    const TensorDescriptor& GetDXDesc() const
    {
        assert(direction == Direction::Backward);
        return dxDesc;
    }

    const TensorDescriptor& GetBnScaleBiasMeanVarDesc() const
    {
        assert(direction == Direction::ForwardTraining || direction == Direction::ForwardInference);
        return scaleBiasDesc;
    }

    const TensorDescriptor& GetScaleBiasDiffDesc() const
    {
        assert(direction == Direction::Backward);
        return scaleBiasDesc;
    }

    bool GetResultSave() const
    {
        assert(direction == Direction::ForwardTraining);
        return resultsave;
    }

    bool GetResultRunning() const
    {
        assert(direction == Direction::ForwardTraining);
        return resultrunning;
    }

    bool UseSaved() const
    {
        assert(direction == Direction::Backward);
        return useSaved;
    }

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
    TensorDescriptor yOrDyDesc;
    TensorDescriptor dxDesc;
    TensorDescriptor scaleBiasDesc;
    double expAvgFactor = 0;
    double epsilon;
    bool resultsave    = false;
    bool resultrunning = false;
    bool useSaved      = false;

    NetworkConfig MakeForwardTrainingNetworkConfig() const;
    NetworkConfig MakeForwardInferenceNetworkConfig() const;
    NetworkConfig MakeBackwardNetworkConfig() const;
};

} // namespace batchnorm

} // namespace miopen
