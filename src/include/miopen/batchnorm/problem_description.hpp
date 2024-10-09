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

#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>
#include <miopen/mlo_internal.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;
struct ExecutionContext;

namespace batchnorm {

enum class Direction
{
    ForwardTraining,
    ForwardInference,
    Backward,
};

struct ProblemDescriptionTag
{
};

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase, ProblemDescriptionTag
{
    // Forward Training
    ProblemDescription(miopenBatchNormMode_t bn_mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& bnScaleBiasMeanVarDesc_,
                       double expAvgFactor_,
                       double epsilon_,
                       bool resultsave_,
                       bool resultrunning_)
        : direction(Direction::ForwardTraining),
          bn_mode(bn_mode_),
          xDesc(xDesc_),
          yOrDyDesc(yDesc_),
          scaleBiasDesc(bnScaleBiasMeanVarDesc_),
          expAvgFactor(expAvgFactor_),
          epsilon(epsilon_),
          resultsave(resultsave_),
          resultrunning(resultrunning_)
    {
        SetSpatialDims();
        in_layout  = ComputeInLayout();
        out_layout = ComputeOutLayout();
    }

    // Forward Inference
    ProblemDescription(miopenBatchNormMode_t bn_mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& bnScaleBiasMeanVarDesc_,
                       double epsilon_)
        : direction(Direction::ForwardInference),
          bn_mode(bn_mode_),
          xDesc(xDesc_),
          yOrDyDesc(yDesc_),
          scaleBiasDesc(bnScaleBiasMeanVarDesc_),
          epsilon(epsilon_)
    {
        SetSpatialDims();
        in_layout  = ComputeInLayout();
        out_layout = ComputeOutLayout();
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
        SetSpatialDims();
        in_layout  = ComputeInLayout();
        out_layout = ComputeOutLayout();
        din_layout = ComputeDinLayout();
    }

    void SetSpatialDims()
    {
        if(Is2D())
            spatial_dim = 2;
        else if(Is3D())
            spatial_dim = 3;
        else
            MIOPEN_THROW("Unknown spatial dim!");
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

    bool IsLayoutNHWC() const
    {
        if(direction == Direction::Backward)
        {
            return xDesc.GetLengths().size() == 4
                       ? ((in_layout == "NHWC") && (out_layout == "NHWC") && (din_layout == "NHWC"))
                       : ((in_layout == "NDHWC") && (out_layout == "NDHWC") &&
                          (din_layout == "NDHWC"));
        }

        return xDesc.GetLengths().size() == 4 ? ((in_layout == "NHWC") && (out_layout == "NHWC"))
                                              : ((in_layout == "NDHWC") && (out_layout == "NDHWC"));
    }

    bool Is2D() const { return xDesc.GetLengths().size() == 4; }
    bool Is3D() const { return xDesc.GetLengths().size() == 5; }

    bool IsFp64() const { return xDesc.GetType() == miopenDouble; }
    bool IsFp32() const { return xDesc.GetType() == miopenFloat; }
    bool IsFp16() const { return xDesc.GetType() == miopenHalf; }
    bool IsBfp16() const { return xDesc.GetType() == miopenBFloat16; }

    NetworkConfig MakeNetworkConfig() const override;

    // This declaration marks batchnorm as a primitive with tuning enabled.
    // Any tunable solver would be able pick it and fetch a db instance in ExecutePrimitive.
    // It has to be discoverable via ADL from problem description.
    friend auto GetDb(const ExecutionContext& ctx, const ProblemDescriptionTag&) -> PerformanceDb;

private:
    Direction direction;
    miopenBatchNormMode_t bn_mode;
    TensorDescriptor xDesc;     // input
    TensorDescriptor yOrDyDesc; // output
    TensorDescriptor dxDesc;
    TensorDescriptor scaleBiasDesc;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

    double expAvgFactor = 0;
    double epsilon;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

    bool resultsave         = false;
    bool resultrunning      = false;
    bool useSaved           = false;
    std::string in_layout   = "NCHW";
    std::string out_layout  = "NCHW";
    std::string din_layout  = "NCHW";
    std::size_t spatial_dim = 2;

    NetworkConfig MakeForwardTrainingNetworkConfig() const;
    NetworkConfig MakeForwardInferenceNetworkConfig() const;
    NetworkConfig MakeBackwardNetworkConfig() const;

    std::string ComputeLayout(const TensorDescriptor& td) const { return td.GetLayout_str(); }
    std::string ComputeInLayout() const { return ComputeLayout(xDesc); }
    std::string ComputeOutLayout() const { return ComputeLayout(yOrDyDesc); }
    std::string ComputeDinLayout() const { return ComputeLayout(dxDesc); }
};

} // namespace batchnorm

} // namespace miopen
