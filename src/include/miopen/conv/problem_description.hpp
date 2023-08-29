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

#include <miopen/conv_algo_name.hpp>
#include <miopen/convolution.hpp>
#include <miopen/names.hpp>
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif
#include <miopen/tensor.hpp>
#include <miopen/problem_description_base.hpp>

#include <boost/any.hpp>

namespace miopen {

struct ExecutionContext;

std::string
EncodeDataTypesForKey(miopenDataType_t in, miopenDataType_t weights, miopenDataType_t out);

inline std::string GetDataTypeName(miopenDataType_t data_type)
{
    switch(data_type)
    {
    case miopenFloat: return "FP32";
    case miopenHalf: return "FP16";
    case miopenInt8: return "INT8";
    case miopenInt8x4: return "INT8x4";
    case miopenInt32: return "INT32";
    case miopenBFloat16: return "BF16";
    case miopenDouble: return "FP64";
    case miopenFloat8: return "FP8";
    case miopenBFloat8: return "BFP8";
    }

    return "Unknown(" + std::to_string(data_type) + ")";
}

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

struct ProblemDescription : ProblemDescriptionBase
#if MIOPEN_ENABLE_SQLITE
    ,
                            SQLiteSerializable<ProblemDescription>
#endif
{
    ProblemDescription() = default;

    ProblemDescription(const TensorDescriptor& in_,
                       const TensorDescriptor& weights_,
                       const TensorDescriptor& out_,
                       const ConvolutionDescriptor& conv_,
                       Direction direction_,
                       int bias_ = 0)
        : in(in_),
          weights(weights_),
          out(out_),
          conv(conv_),
          in_layout(ComputeInLayout()),
          weights_layout(ComputeWeightsLayout()),
          out_layout(ComputeOutLayout()),
          direction(direction_),
          bias(bias_)
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
    unsigned GetInBatchSize_() const { return GetN5(GetSpatialDims(), in.GetLengths()); }
    unsigned GetBatchSize_() const { return GetInBatchSize_(); } // alias of GetInBatchSize_()
    unsigned GetInChannels_() const { return GetC5(GetSpatialDims(), in.GetLengths()); }
    unsigned GetInDepth_() const { return GetD5(GetSpatialDims(), in.GetLengths()); }
    unsigned GetInHeight_() const { return GetH5(GetSpatialDims(), in.GetLengths()); }
    unsigned GetInWidth_() const { return GetW5(GetSpatialDims(), in.GetLengths()); }
    unsigned GetInBatchStride_() const { return GetN5(GetSpatialDims(), in.GetStrides()); }
    unsigned GetInChannelStride_() const { return GetC5(GetSpatialDims(), in.GetStrides()); }
    unsigned GetInStrideD_() const { return GetD5(GetSpatialDims(), in.GetStrides()); }
    unsigned GetInStrideH_() const { return GetH5(GetSpatialDims(), in.GetStrides()); }
    unsigned GetInStrideW_() const { return GetW5(GetSpatialDims(), in.GetStrides()); }
    std::string GetInLayout() const { return in_layout; }
    std::string ComputeInLayout() const
    {
        if(GetSpatialDims() == 2)
        {
            return in.GetLayout(in.GetLayout_str());
        }
        else
        {
            return in.GetLayout("NCDHW");
        }
    }
    std::size_t GetInElementSize() const { return GetTypeSize(GetInDataType()); }

    std::size_t GetInSize() const
    {
        return static_cast<size_t>(GetInBatchSize_()) * GetInChannels_() * GetInDepth_() *
               GetInHeight_() * GetInWidth_() * GetInElementSize();
    }

    // Out getters
    miopenDataType_t GetOutDataType() const { return out.GetType(); }
    std::optional<miopenDataType_t> GetOutCastType() const { return out.GetCastType(); }
    unsigned GetOutBatchSize_() const { return GetN5(GetSpatialDims(), out.GetLengths()); }
    unsigned GetOutChannels_() const { return GetC5(GetSpatialDims(), out.GetLengths()); }
    unsigned GetOutDepth_() const { return GetD5(GetSpatialDims(), out.GetLengths()); }
    unsigned GetOutHeight_() const { return GetH5(GetSpatialDims(), out.GetLengths()); }
    unsigned GetOutWidth_() const { return GetW5(GetSpatialDims(), out.GetLengths()); }
    unsigned GetOutBatchStride_() const { return GetN5(GetSpatialDims(), out.GetStrides()); }
    unsigned GetOutChannelStride_() const { return GetC5(GetSpatialDims(), out.GetStrides()); }
    unsigned GetOutStrideD_() const { return GetD5(GetSpatialDims(), out.GetStrides()); }
    unsigned GetOutStrideH_() const { return GetH5(GetSpatialDims(), out.GetStrides()); }
    unsigned GetOutStrideW_() const { return GetW5(GetSpatialDims(), out.GetStrides()); }
    std::string GetOutLayout() const { return out_layout; }
    std::string ComputeOutLayout() const
    {
        if(GetSpatialDims() == 2)
        {
            return out.GetLayout(out.GetLayout_str());
        }
        else
        {
            return out.GetLayout("NCDHW");
        }
    }
    std::size_t GetOutElementSize() const { return GetTypeSize(GetOutDataType()); }

    std::size_t GetOutSize() const
    {
        return static_cast<size_t>(GetOutBatchSize_()) * GetOutChannels_() * GetOutDepth_() *
               GetOutHeight_() * GetOutWidth_() * GetOutElementSize();
    }

    // Weights getters
    miopenDataType_t GetWeightsDataType() const { return weights.GetType(); }
    std::optional<miopenDataType_t> GetWeightsCastType() const { return weights.GetCastType(); }
    unsigned GetWeightsDepth_() const { return GetD5(GetSpatialDims(), weights.GetLengths()); }
    unsigned GetWeightsHeight_() const
    {
        if(weights.GetLayout_str() == "CHWNc")
            return GetHofCHWN(weights.GetLengths());
        else
            return GetH5(GetSpatialDims(), weights.GetLengths());
    }
    unsigned GetWeightsWidth_() const
    {
        if(weights.GetLayout_str() == "CHWNc")
            return GetWofCHWN(weights.GetLengths());
        else
            return GetW5(GetSpatialDims(), weights.GetLengths());
    }
    // unsigned GetWeightsStrideD() const { return GetD5(GetSpatialDims(), weights.GetStrides()); }
    // unsigned GetWeightsStrideH() const { return GetH5(GetSpatialDims(), weights.GetStrides()); }
    // unsigned GetWeightsStrideW() const { return GetW5(GetSpatialDims(), weights.GetStrides()); }
    std::string GetWeightsLayout() const { return weights_layout; }
    std::string ComputeWeightsLayout() const
    {
        if(GetSpatialDims() == 2)
        {
            return weights.GetLayout(weights.GetLayout_str());
        }
        else
        {
            return weights.GetLayout("NCDHW");
        }
    }
    std::size_t GetWeightsElementSize() const { return GetTypeSize(GetWeightsDataType()); }

    std::size_t GetWeightsSize() const
    {
        return static_cast<size_t>(GetInChannels_()) * GetOutChannels_() * GetWeightsDepth_() *
               GetWeightsHeight_() * GetWeightsWidth_() * GetWeightsElementSize();
    }

    const TensorDescriptor& GetIn() const { return in; }
    const TensorDescriptor& GetWeights() const { return weights; }
    const TensorDescriptor& GetOut() const { return out; }
    const ConvolutionDescriptor& GetConv() const { return conv; }

    Direction GetDirection() const { return direction; }
    std::string GetDirectionStr() const;

    int GetBias() const { return bias; }

    std::size_t GetBiasSize() const
    {
        return (GetBias() != 0) ? (GetOutChannels_() * GetOutElementSize()) : 0;
    }

    int GetBackwardPadW() const { return static_cast<int>(GetWeightsWidth_()) - GetPadW() - 1; }
    int GetBackwardPadH() const { return static_cast<int>(GetWeightsHeight_()) - GetPadH() - 1; }

    bool IsAsymmetricPadH() const
    {
        return conv.paddingMode == miopenPaddingSame && (GetWeightsHeight_() % 2) == 0;
    }
    bool IsAsymmetricPadW() const
    {
        return conv.paddingMode == miopenPaddingSame && (GetWeightsWidth_() % 2) == 0;
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

    void HeuristicUpdateLayouts();

    void BuildConfKey(std::string& conf_key) const;

    NetworkConfig BuildConfKey() const
    {
        std::string ret;
        BuildConfKey(ret);
        return NetworkConfig{ret};
    }

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

#if MIOPEN_ENABLE_SQLITE
    static std::string table_name() { return "config"; }

    template <class Self>
    static void Visit(Self&& self, std::function<void(int, std::string)> f)
    {
        // The column names match the driver command line argument names
        f(self.GetSpatialDims(), "spatial_dim");
        f(self.GetInChannels_(), "in_channels");
        f(self.GetInHeight_(), "in_h");
        f(self.GetInWidth_(), "in_w");
        f(self.GetInDepth_(), "in_d");
        f(self.GetWeightsHeight_(), "fil_h");
        f(self.GetWeightsWidth_(), "fil_w");
        f(self.GetWeightsDepth_(), "fil_d");
        f(self.GetOutChannels_(), "out_channels");
        f(self.GetBatchSize_(), "batchsize");
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
#endif

    void SetupFloats(ExecutionContext& ctx) const;

private:
    TensorDescriptor in;
    TensorDescriptor weights;
    TensorDescriptor out;
    ConvolutionDescriptor conv;
    std::string in_layout;
    std::string weights_layout;
    std::string out_layout;
    Direction direction = Direction::Forward;
    int bias            = 0;
};

} // namespace conv
} // namespace miopen
