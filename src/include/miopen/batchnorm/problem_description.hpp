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

bool is_fp16_or_bfp16(miopenDataType_t type);
bool is_fp32_or_fp64(miopenDataType_t type);

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase,
                                                    ProblemDescriptionTag
#if MIOPEN_ENABLE_SQLITE
    ,
                                                    SQLiteSerializable<ProblemDescription>
#endif
{
    // Forward Training
    ProblemDescription(miopenBatchNormMode_t bn_mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& scaleDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& sMeanDesc_,
                       const TensorDescriptor& sVarianceDesc_,
                       double expAvgFactor_,
                       double epsilon_,
                       bool resultsave_,
                       bool resultrunning_)
        : direction(Direction::ForwardTraining),
          bn_mode(bn_mode_),
          xDesc(xDesc_),
          yOrDyDesc(yDesc_),
          scaleDesc(scaleDesc_),
          biasDesc(biasDesc_),
          sMeanDesc(sMeanDesc_),
          sVarianceDesc(sVarianceDesc_),
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
                       const TensorDescriptor& scaleDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& sMeanDesc_,
                       const TensorDescriptor& sVarianceDesc_,
                       double epsilon_)
        : direction(Direction::ForwardInference),
          bn_mode(bn_mode_),
          xDesc(xDesc_),
          yOrDyDesc(yDesc_),
          scaleDesc(scaleDesc_),
          biasDesc(biasDesc_),
          sMeanDesc(sMeanDesc_),
          sVarianceDesc(sVarianceDesc_),
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
                       const TensorDescriptor& scaleDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& sMeanDesc_,
                       const TensorDescriptor& sVarianceDesc_,
                       double epsilon_,
                       bool useSaved_)
        : direction(Direction::Backward),
          bn_mode(bn_mode_),
          xDesc(xDesc_),
          yOrDyDesc(dyDesc_),
          dxDesc(dxDesc_),
          scaleDesc(scaleDesc_),
          biasDesc(biasDesc_),
          sMeanDesc(sMeanDesc_),
          sVarianceDesc(sVarianceDesc_),
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

    const TensorDescriptor& GetBnScale() const { return scaleDesc; }

    const TensorDescriptor& GetBnBias() const { return biasDesc; }

    const TensorDescriptor& GetBnSMean() const { return sMeanDesc; }

    const TensorDescriptor& GetBnSVar() const { return sVarianceDesc; }

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

    bool IsLayoutNCHW() const
    {
        if(direction == Direction::Backward)
        {
            return xDesc.GetLengths().size() == 4
                       ? ((in_layout == "NCHW") && (out_layout == "NCHW") && (din_layout == "NCHW"))
                       : ((in_layout == "NCDHW") && (out_layout == "NCDHW") &&
                          (din_layout == "NCDHW"));
        }

        return xDesc.GetLengths().size() == 4 ? ((in_layout == "NCHW") && (out_layout == "NCHW"))
                                              : ((in_layout == "NCDHW") && (out_layout == "NCDHW"));
    }

    bool Is2D() const { return xDesc.GetLengths().size() == 4; }
    bool Is3D() const { return xDesc.GetLengths().size() == 5; }

    bool IsFp64() const { return xDesc.GetType() == miopenDouble; }
    bool IsFp32() const { return xDesc.GetType() == miopenFloat; }
    bool IsFp16() const { return xDesc.GetType() == miopenHalf; }
    bool IsBFp16() const { return xDesc.GetType() == miopenBFloat16; }
    bool IsMix() const { return (IsFp16() || IsBFp16()) && sMeanDesc.GetType() == miopenFloat; }
    bool IsScaleFp16() const { return scaleDesc.GetType() == miopenHalf; }
    bool IsScaleFp32() const { return scaleDesc.GetType() == miopenFloat; }

    void Serialize(std::ostream& stream) const { stream << MakeNetworkConfig().ToString(); }

    NetworkConfig MakeNetworkConfig() const override;

    template <class Self>
    static void Visit(Self&& self, std::function<void(int64_t, std::string)> f)
    {
        // The column names match the driver command line argument names
        f(self.spatial_dim, "spatial_dim");
        f(self.GetBatchSize(), "batchsize");
        f(self.GetChannel(), "in_channels");
        f(self.GetHeight(), "in_h");
        f(self.GetWidth(), "in_w");
        f(self.GetDepth(), "in_d");

        f(self.resultsave, "resultsave");
        f(self.resultrunning, "resultrunning");
        f(self.useSaved, "useSaved");
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(std::string, std::string)> f)
    {
        f(self.ComputeInLayout(), "layout");
        f(self.GetDirectionStr(), "direction");
        f(GetDataTypeName(self.xDesc.GetType()), "data_type");
        f(self.GetModeStr(), "mode");
    }

    template <class Self, class Visitor>
    static void VisitAll(Self&& self, const Visitor& f)
    {
        Visit(std::forward<Self>(self), [&](int64_t value, std::string name) { f(value, name); });
        Visit(std::forward<Self>(self),
              [&](std::string value, std::string name) { f(value, name); });
    }

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

    TensorDescriptor scaleDesc; // scale
    TensorDescriptor biasDesc;  // bias (shift)
    TensorDescriptor sMeanDesc;
    TensorDescriptor sVarianceDesc;

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

    size_t GetSpatialDims() const { return spatial_dim; }

    std::size_t GetBatchSize() const { return GetN5(GetSpatialDims(), xDesc.GetLengths()); }
    std::size_t GetChannel() const { return GetC5(GetSpatialDims(), xDesc.GetLengths()); }
    std::size_t GetHeight() const { return GetH5(GetSpatialDims(), xDesc.GetLengths()); }
    std::size_t GetWidth() const { return GetW5(GetSpatialDims(), xDesc.GetLengths()); }
    std::size_t GetDepth() const { return GetD5(GetSpatialDims(), xDesc.GetLengths()); }

    std::string GetDirectionStr() const
    {
        std::string s;

        switch(direction)
        {
        case Direction::ForwardInference: return "Inf"; ;
        case Direction::ForwardTraining: return "Trn";
        case Direction::Backward: return "Bwd";
        default: MIOPEN_THROW(miopenStatusInvalidValue, "Wrong Batchnorm Direction provided");
        }

        return s;
    }

    std::string GetModeStr() const
    {
        switch(bn_mode)
        {
        case miopenBNPerActivation: return "0";
        case miopenBNSpatial: return "1";
        default: MIOPEN_THROW(miopenStatusInvalidValue, "Wrong Batchnorm Mode provided");
        }
    }
};

bool IsOCLInferTypeValid(const ProblemDescription& bn_problem);
bool IsCKInferTypeValid(const ProblemDescription& bn_problem);

bool IsOCLFwdTrainTypeValid(const ProblemDescription& bn_problem);
bool IsCKFwdTrainTypeValid(const ProblemDescription& bn_problem);

bool IsOCLBwdTypeValid(const ProblemDescription& bn_problem);
bool IsCKBwdTypeValid(const ProblemDescription& bn_problem);

} // namespace batchnorm

} // namespace miopen
