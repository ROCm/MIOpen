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
#include <miopen/sqlite_db.hpp>
#include <miopen/tensor.hpp>

#include <boost/any.hpp>

namespace miopen {

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
    }

    return "Unknown(" + std::to_string(data_type) + ")";
}

template <class TElement>
constexpr auto GetDHW(int spatial_dims, const std::vector<TElement>& data)
{
    if(spatial_dims == 2)
        return std::make_tuple(0, data[0], data[1]);
    return std::make_tuple(data[0], data[1], data[2]);
}

template <class TElement>
constexpr TElement GetD3(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<0>(GetDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetH3(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<1>(GetDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetW3(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<2>(GetDHW(spatial_dims, data));
}

template <class TElement>
constexpr auto GetNCDHW(int spatial_dims, const std::vector<TElement>& data)
{
    if(spatial_dims == 3)
        return miopen::tien<5>(data, 1);
    else
        return std::make_tuple(data[0], data[1], static_cast<TElement>(1), data[2], data[3]);
}

template <class TElement>
constexpr TElement GetN5(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<0>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetC5(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<1>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetD5(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<2>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetH5(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<3>(GetNCDHW(spatial_dims, data));
}

template <class TElement>
constexpr TElement GetW5(int spatial_dims, const std::vector<TElement>& data)
{
    return std::get<4>(GetNCDHW(spatial_dims, data));
}

namespace conv {

struct ProblemDescription
#if MIOPEN_ENABLE_SQLITE
    : SQLiteSerializable<ProblemDescription>
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
    std::size_t GetSpatialDims() const { return conv.GetSpatialDimension(); }
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

    // In getters
    miopenDataType_t GetInDataType() const { return in.GetType(); }
    std::size_t GetInBatchSize() const { return GetN5(GetSpatialDims(), in.GetLengths()); }
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
    std::string ComputeInLayout() const
    {
        if(GetSpatialDims() == 2)
        {
            return in.GetLayout("NCHW");
        }
        else
        {
            return in.GetLayout("NCDHW");
        }
    }
    std::size_t GetInElementSize() const { return GetTypeSize(GetInDataType()); }

    std::size_t GetInSize() const
    {
        // clang-format off
        return GetInBatchSize() * GetInChannels() * GetInDepth() * GetInHeight() *
            GetInWidth() * GetInElementSize();
        // clang-format on
    }

    // Out getters
    miopenDataType_t GetOutDataType() const { return out.GetType(); }
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
    std::string ComputeOutLayout() const
    {
        if(GetSpatialDims() == 2)
        {
            return out.GetLayout("NCHW");
        }
        else
        {
            return out.GetLayout("NCDHW");
        }
    }
    std::size_t GetOutElementSize() const { return GetTypeSize(GetOutDataType()); }

    std::size_t GetOutSize() const
    {
        // clang-format off
        return GetOutBatchSize() * GetOutChannels() * GetOutDepth() * GetOutHeight() *
               GetOutWidth() * GetOutElementSize();
        // clang-format on
    }

    // Weights getters
    miopenDataType_t GetWeightsDataType() const { return weights.GetType(); }
    std::size_t GetWeightsDepth() const { return GetD5(GetSpatialDims(), weights.GetLengths()); }
    std::size_t GetWeightsHeight() const { return GetH5(GetSpatialDims(), weights.GetLengths()); }
    std::size_t GetWeightsWidth() const { return GetW5(GetSpatialDims(), weights.GetLengths()); }
    // std::size_t GetWeightsStrideD() const { return GetD5(GetSpatialDims(), weights.GetStrides());
    // }
    // std::size_t GetWeightsStrideH() const { return GetH5(GetSpatialDims(), weights.GetStrides());
    // }
    // std::size_t GetWeightsStrideW() const { return GetW5(GetSpatialDims(), weights.GetStrides());
    // }
    std::string GetWeightsLayout() const { return weights_layout; }
    std::string ComputeWeightsLayout() const
    {
        if(GetSpatialDims() == 2)
        {
            return weights.GetLayout("NCHW");
        }
        else
        {
            return weights.GetLayout("NCDHW");
        }
    }
    std::size_t GetWeightsElementSize() const { return GetTypeSize(GetWeightsDataType()); }

    std::size_t GetWeightsSize() const
    {
        // clang-format off
        return GetInChannels() * GetOutChannels() * GetWeightsDepth() * GetWeightsHeight() *
               GetWeightsWidth() * GetWeightsElementSize();
        // clang-format on
    }

    const TensorDescriptor& GetIn() const { return in; }
    const TensorDescriptor& GetWeights() const { return weights; }
    const TensorDescriptor& GetOut() const { return out; }
    const ConvolutionDescriptor& GetConv() const { return conv; }
    Direction GetDirection() const { return direction; }
    int GetBias() const { return bias; }

    std::size_t GetBaiasSize() const
    {
        return (GetBias() != 0) ? (GetOutChannels() * GetOutElementSize()) : 0;
    }

    std::size_t GetBackwardPadW() const { return GetWeightsWidth() - GetPadW() - 1; }
    std::size_t GetBackwardPadH() const { return GetWeightsHeight() - GetPadW() - 1; }

    bool IsAsymmetricPadH() const
    {
        return conv.paddingMode == miopenPaddingSame && (GetWeightsHeight() % 2) == 0;
    }
    bool IsAsymmetricPadW() const
    {
        return conv.paddingMode == miopenPaddingSame && (GetWeightsWidth() % 2) == 0;
    }

    bool Is2d() const { return GetSpatialDims() == 2; }

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

    bool IsLayoutDefault() const;

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

    static std::string table_name() { return "config"; }
    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        // Todo: shouldn't sqlitedb serialization support 3d convs?
        f(std::to_string(self.GetInChannels()), "in_channels");
        f(std::to_string(self.GetInHeight()), "in_h");
        f(std::to_string(self.GetInWidth()), "in_w");
        f(std::to_string(self.GetWeightsHeight()), "filter_h");
        f(std::to_string(self.GetWeightsWidth()), "filter_w");
        f(std::to_string(self.GetOutChannels()), "out_channels");
        f(std::to_string(self.GetInBatchSize()), "batchsize");
        f(std::to_string(self.GetPadH()), "pad_h");
        f(std::to_string(self.GetPadW()), "pad_w");
        f(std::to_string(self.GetKernelStrideH()), "conv_stride_1");
        f(std::to_string(self.GetKernelStrideW()), "conv_stride_0");
        f(std::to_string(self.GetDilationH()), "dilation_h");
        f(std::to_string(self.GetDilationW()), "dilation_w");
        f(std::to_string(self.GetBias()), "bias");
        f("'" + self.GetInLayout() + "'", "layout");
        std::string data_type = EncodeDataTypesForKey(
            self.GetInDataType(), self.GetWeightsDataType(), self.GetOutDataType());
        f("'" + data_type + "'", "data_type");

        switch(self.GetDirection())
        {
        case Direction::Forward: f("'F'", "direction"); break;
        case Direction::BackwardData: f("'B'", "direction"); break;
        case Direction::BackwardWeights: f("'W'", "direction"); break;
        }

        f(std::to_string(self.GetGroupCount()), "group_count");
    }

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
