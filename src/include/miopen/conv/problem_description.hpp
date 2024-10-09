/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <boost/any.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/names.hpp>
#include <miopen/scalar.hpp>

#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>

#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif

namespace miopen {

struct ExecutionContext;

MIOPEN_INTERNALS_EXPORT std::string
EncodeDataTypesForKey(miopenDataType_t in, miopenDataType_t weights, miopenDataType_t out);

template <class TElement>
constexpr auto GetDHW(unsigned spatial_dims, const std::vector<TElement>& data)
{
    if(spatial_dims == 2)
        return std::make_tuple(0, data[0], data[1]);
    return std::make_tuple(data[0], data[1], data[2]);
}

template <class TElement>
constexpr TElement GetD3(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<0>(GetDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetH3(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<1>(GetDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetW3(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<2>(GetDHW(spatial_dims, data));
}
template <class TElement>
constexpr auto GetCHWN(const std::vector<TElement>& data)
{
    return miopen::tien<4>(data, 1);
}

template <class TElement>
constexpr TElement GetNofCHWN(const std::vector<TElement>& data)
{
    return std::get<3>(GetCHWN(data));
}

template <class TElement>
constexpr TElement GetCofCHWN(const std::vector<TElement>& data)
{
    return std::get<0>(GetCHWN(data));
}

template <class TElement>
constexpr TElement GetHofCHWN(const std::vector<TElement>& data)
{
    return std::get<1>(GetCHWN(data));
}

template <class TElement>
constexpr TElement GetWofCHWN(const std::vector<TElement>& data)
{
    return std::get<2>(GetCHWN(data));
}

template <class TElement>
constexpr TElement GetN5(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<0>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetC5(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<1>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetD5(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<2>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetH5(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<3>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetW5(unsigned spatial_dims, const std::vector<TElement>& data)
{
    return std::get<4>(GetNCDHW(spatial_dims, data));
}

namespace conv {

MIOPEN_INTERNALS_EXPORT miopenAlphaBetaCase_t ClassifyAlphaBeta(const Scalar& alpha,
                                                                const Scalar& beta);

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase
#if MIOPEN_ENABLE_SQLITE
    ,
                                                    SQLiteSerializable<ProblemDescription>
#endif
{
    ProblemDescription() = default;

    /// \todo Get rid of the swapping of x and y.
    ProblemDescription(const TensorDescriptor& in_, // x for Forward, y for Backward*
                       const TensorDescriptor& weights_,
                       const TensorDescriptor& out_, // y for Forward, x for Backward*
                       const ConvolutionDescriptor& conv_,
                       Direction direction_,
                       int bias_            = 0,
                       const Scalar& alpha_ = Scalar(1.0),
                       const Scalar& beta_  = Scalar(0.0))
        : in(in_),
          weights(weights_),
          out(out_),
          conv(conv_),
          in_layout(ComputeInLayout()),
          weights_layout(ComputeWeightsLayout()),
          out_layout(ComputeOutLayout()),
          direction(direction_),
          bias(bias_),
          alpha(alpha_),
          beta(beta_),
          alpha_beta_case(ClassifyAlphaBeta(alpha, beta))
    {
        HeuristicUpdateLayouts();
    }

    // Conv descriptor getters
    unsigned GetSpatialDims() const { return conv.GetSpatialDimension(); }
    int GetPadD() const { return GetD3(GetSpatialDims(), conv.GetConvPads()); }
    int GetPadH() const { return GetH3(GetSpatialDims(), conv.GetConvPads()); }
    int GetPadW() const { return GetW3(GetSpatialDims(), conv.GetConvPads()); }
    int GetKernelStrideD() const { return GetD3(GetSpatialDims(), conv.GetConvStrides()); }
    int GetKernelStrideH() const { return GetH3(GetSpatialDims(), conv.GetConvStrides()); }
    int GetKernelStrideW() const { return GetW3(GetSpatialDims(), conv.GetConvStrides()); }
    int GetDilationD() const { return GetD3(GetSpatialDims(), conv.GetConvDilations()); }
    int GetDilationH() const { return GetH3(GetSpatialDims(), conv.GetConvDilations()); }
    int GetDilationW() const { return GetW3(GetSpatialDims(), conv.GetConvDilations()); }
    int GetGroupCount() const { return conv.GetGroupCount(); }
    int GetVectorLength() const { return in.GetVectorLength(); }

    // In getters
    miopenDataType_t GetInDataType() const { return in.GetType(); }
    std::optional<miopenDataType_t> GetInCastType() const { return in.GetCastType(); }
    std::size_t GetInBatchSize() const { return GetN5(GetSpatialDims(), in.GetLengths()); }
    std::size_t GetBatchSize() const { return GetInBatchSize(); } // alias of GetInBatchSize()
    std::size_t GetInChannels() const { return GetC5(GetSpatialDims(), in.GetLengths()); }
    std::size_t GetInDepth() const { return GetD5(GetSpatialDims(), in.GetLengths()); }
    std::size_t GetInHeight() const { return GetH5(GetSpatialDims(), in.GetLengths()); }
    std::size_t GetInWidth() const { return GetW5(GetSpatialDims(), in.GetLengths()); }
    std::size_t GetInBatchStride() const { return GetN5(GetSpatialDims(), in.GetStrides()); }
    std::size_t GetInChannelStride() const { return GetC5(GetSpatialDims(), in.GetStrides()); }
    std::size_t GetInStrideD() const { return GetD5(GetSpatialDims(), in.GetStrides()); }
    std::size_t GetInStrideH() const { return GetH5(GetSpatialDims(), in.GetStrides()); }
    std::size_t GetInStrideW() const { return GetW5(GetSpatialDims(), in.GetStrides()); }
    std::string GetInLayout() const { return in_layout; }
    std::size_t GetInElementSize() const { return GetTypeSize(GetInDataType()); }
    std::size_t GetInSize() const { return in.GetNumBytes(); }

    // Out getters
    miopenDataType_t GetOutDataType() const { return out.GetType(); }
    std::optional<miopenDataType_t> GetOutCastType() const { return out.GetCastType(); }
    std::size_t GetOutBatchSize() const { return GetN5(GetSpatialDims(), out.GetLengths()); }
    std::size_t GetOutChannels() const { return GetC5(GetSpatialDims(), out.GetLengths()); }
    std::size_t GetOutDepth() const { return GetD5(GetSpatialDims(), out.GetLengths()); }
    std::size_t GetOutHeight() const { return GetH5(GetSpatialDims(), out.GetLengths()); }
    std::size_t GetOutWidth() const { return GetW5(GetSpatialDims(), out.GetLengths()); }
    std::size_t GetOutBatchStride() const { return GetN5(GetSpatialDims(), out.GetStrides()); }
    std::size_t GetOutChannelStride() const { return GetC5(GetSpatialDims(), out.GetStrides()); }
    std::size_t GetOutStrideD() const { return GetD5(GetSpatialDims(), out.GetStrides()); }
    std::size_t GetOutStrideH() const { return GetH5(GetSpatialDims(), out.GetStrides()); }
    std::size_t GetOutStrideW() const { return GetW5(GetSpatialDims(), out.GetStrides()); }
    std::string GetOutLayout() const { return out_layout; }
    std::size_t GetOutElementSize() const { return GetTypeSize(GetOutDataType()); }
    std::size_t GetOutSize() const { return out.GetNumBytes(); }

    // Weights getters
    miopenDataType_t GetWeightsDataType() const { return weights.GetType(); }
    std::optional<miopenDataType_t> GetWeightsCastType() const { return weights.GetCastType(); }
    std::size_t GetWeightsDepth() const { return GetD5(GetSpatialDims(), weights.GetLengths()); }
    std::size_t GetWeightsHeight() const
    {
        if(weights_layout == "CHWNc")
            return GetHofCHWN(weights.GetLengths());
        else
            return GetH5(GetSpatialDims(), weights.GetLengths());
    }
    std::size_t GetWeightsWidth() const
    {
        if(weights_layout == "CHWNc")
            return GetWofCHWN(weights.GetLengths());
        else
            return GetW5(GetSpatialDims(), weights.GetLengths());
    }
    std::size_t GetWeightsStrideK() const { return GetN5(GetSpatialDims(), weights.GetStrides()); }
    std::size_t GetWeightsStrideC() const { return GetC5(GetSpatialDims(), weights.GetStrides()); }
    std::size_t GetWeightsStrideD() const { return GetD5(GetSpatialDims(), weights.GetStrides()); }
    std::size_t GetWeightsStrideH() const { return GetH5(GetSpatialDims(), weights.GetStrides()); }
    std::size_t GetWeightsStrideW() const { return GetW5(GetSpatialDims(), weights.GetStrides()); }
    std::string GetWeightsLayout() const { return weights_layout; }
    std::size_t GetWeightsElementSize() const { return GetTypeSize(GetWeightsDataType()); }
    std::size_t GetWeightsSize() const { return weights.GetNumBytes(); }

    const TensorDescriptor& GetIn() const { return in; }
    const TensorDescriptor& GetWeights() const { return weights; }
    const TensorDescriptor& GetOut() const { return out; }
    const ConvolutionDescriptor& GetConv() const { return conv; }

    Direction GetDirection() const { return direction; }
    bool IsDirectionForward() const { return direction == conv::Direction::Forward; }
    bool IsDirectionBackwardData() const { return direction == conv::Direction::BackwardData; }
    bool IsDirectionBackwardWrW() const { return direction == conv::Direction::BackwardWeights; }
    std::string GetDirectionStr() const;

    const Scalar& GetAlpha() const { return alpha; }
    const Scalar& GetBeta() const { return beta; }

    miopenAlphaBetaCase_t GetAlphaBetaCase() const { return alpha_beta_case; }

    std::string GetAlphaBetaCaseStr() const;

    int GetBias() const { return bias; }

    std::size_t GetBiasSize() const
    {
        return (GetBias() != 0) ? (GetOutChannels() * GetOutElementSize()) : 0;
    }

    int64_t GetBackwardPadW() const
    {
        return static_cast<int64_t>(GetWeightsWidth()) - GetPadW() - 1;
    }
    int64_t GetBackwardPadH() const
    {
        return static_cast<int64_t>(GetWeightsHeight()) - GetPadH() - 1;
    }

    bool IsAsymmetricPadH() const
    {
        return conv.paddingMode == miopenPaddingSame && (GetWeightsHeight() % 2) == 0;
    }
    bool IsAsymmetricPadW() const
    {
        return conv.paddingMode == miopenPaddingSame && (GetWeightsWidth() % 2) == 0;
    }

    bool Is2d() const { return GetSpatialDims() == 2; }
    bool Is3d() const { return GetSpatialDims() == 3; }

    bool IsFp32() const
    {
        return GetInDataType() == miopenFloat && GetWeightsDataType() == miopenFloat &&
               GetOutDataType() == miopenFloat;
    }
    bool IsFp16() const
    {
        return GetInDataType() == miopenHalf && GetWeightsDataType() == miopenHalf &&
               GetOutDataType() == miopenHalf;
    }
    bool IsBfp16() const
    {
        return GetInDataType() == miopenBFloat16 && GetWeightsDataType() == miopenBFloat16 &&
               GetOutDataType() == miopenBFloat16;
    }
    bool IsInt8() const
    {
        return GetInDataType() == miopenInt8 && GetWeightsDataType() == miopenInt8 &&
               (GetOutDataType() == miopenInt32 || GetOutDataType() == miopenFloat);
    }
    bool IsFp8() const
    {
        return GetInDataType() == miopenFloat8 || GetWeightsDataType() == miopenFloat8 ||
               GetOutDataType() == miopenFloat8;
    }
    bool IsBfp8() const
    {
        return GetInDataType() == miopenBFloat8 || GetWeightsDataType() == miopenBFloat8 ||
               GetOutDataType() == miopenBFloat8;
    }
    bool IsTensorsCasted() const
    {
        return GetInCastType() || GetWeightsCastType() || GetOutCastType();
    }

    // To be used in Solvers that do not implement ALT FP16 kernels.
    // Those Solvers must be non-applicable for gfx90a when this function returns true.
    bool IsGfx90aFp16altRequired() const
    {
        if(!IsFp16())
            return false;
        if(direction == conv::Direction::Forward)
            return conv.attribute.gfx90aFp16alt.GetFwd();
        if(direction == conv::Direction::BackwardData)
            return conv.attribute.gfx90aFp16alt.GetBwd();
        if(direction == conv::Direction::BackwardWeights)
            return conv.attribute.gfx90aFp16alt.GetWrW();
        MIOPEN_THROW("Direction must be known!");
    }

    bool IsLayoutDefault() const;
    bool IsLayoutNHWC() const;
    bool IsLayoutNCHWc() const;
    bool IsNCHWc_NCHWc() const;
    bool IsNCHWc_CHWNc() const;

    bool HasNonPackedTensors() const
    {
        return !(in.IsPacked() && weights.IsPacked() && out.IsPacked());
    }

    bool HasMixedDataTypes() const
    {
        return !(GetInDataType() == GetWeightsDataType() &&
                 GetWeightsDataType() == GetOutDataType());
    }

    bool AllTensorsDimsFitIntoInt() const
    {
        return in.AllDimsFitIntoInt() && weights.AllDimsFitIntoInt() && out.AllDimsFitIntoInt();
    }

    bool AllTensorsLengthsFitIntoInt() const
    {
        return in.AllLengthsFitIntoInt() && weights.AllLengthsFitIntoInt() &&
               out.AllLengthsFitIntoInt();
    }

    void HeuristicUpdateLayouts();

    void MakeNetworkConfig(std::string& conf_key) const;

    NetworkConfig MakeNetworkConfig() const override
    {
        std::string ret;
        MakeNetworkConfig(ret);
        return NetworkConfig{ret};
    }

    // Todo: remove after fixing fin
    [[deprecated]] NetworkConfig BuildConfKey() const { return MakeNetworkConfig(); }

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

#if MIOPEN_ENABLE_SQLITE
    static std::string table_name() { return "config"; }
#endif

    template <class Self>
    static void Visit(Self&& self, std::function<void(int64_t, std::string)> f)
    {
        // The column names match the driver command line argument names
        f(self.GetSpatialDims(), "spatial_dim");
        f(self.GetInChannels(), "in_channels");
        f(self.GetInHeight(), "in_h");
        f(self.GetInWidth(), "in_w");
        f(self.GetInDepth(), "in_d");
        f(self.GetWeightsHeight(), "fil_h");
        f(self.GetWeightsWidth(), "fil_w");
        f(self.GetWeightsDepth(), "fil_d");
        f(self.GetOutChannels(), "out_channels");
        f(self.GetBatchSize(), "batchsize");
        f(self.GetPadH(), "pad_h");
        f(self.GetPadW(), "pad_w");
        f(self.GetPadD(), "pad_d");
        f(self.GetKernelStrideH(), "conv_stride_h");
        f(self.GetKernelStrideW(), "conv_stride_w");
        f(self.GetKernelStrideD(), "conv_stride_d");
        f(self.GetDilationH(), "dilation_h");
        f(self.GetDilationW(), "dilation_w");
        f(self.GetDilationD(), "dilation_d");
        f(self.GetBias(), "bias");
        f(self.GetGroupCount(), "group_count");
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(std::string, std::string)> f)
    {
        f(self.GetInLayout(), "layout");
        std::string data_type = EncodeDataTypesForKey(
            self.GetInDataType(), self.GetWeightsDataType(), self.GetOutDataType());
        f(data_type, "data_type");
        f(self.GetDirectionStr(), "direction");
    }

    template <class Self, class Visitor>
    static void VisitAll(Self&& self, const Visitor& f)
    {
        Visit(std::forward<Self>(self), [&](int64_t value, std::string name) { f(value, name); });
        Visit(std::forward<Self>(self),
              [&](std::string value, std::string name) { f(value, name); });
    }

    void SetupFloats(ExecutionContext& ctx) const;

private:
    std::string ComputeLayout(const TensorDescriptor& td) const;
    std::string ComputeInLayout() const;
    std::string ComputeOutLayout() const;
    std::string ComputeWeightsLayout() const;

    TensorDescriptor in;
    TensorDescriptor weights;
    TensorDescriptor out;
    ConvolutionDescriptor conv;
    std::string in_layout;
    std::string weights_layout;
    std::string out_layout;
    Direction direction                   = Direction::Forward;
    int bias                              = 0;
    Scalar alpha                          = Scalar(1.0);
    Scalar beta                           = Scalar(0.0);
    miopenAlphaBetaCase_t alpha_beta_case = DEFAULT;
};

} // namespace conv
} // namespace miopen
