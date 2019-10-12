/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef GUARD_PROBLEM_DESCRIPTION_HPP_
#define GUARD_PROBLEM_DESCRIPTION_HPP_

#include <miopen/tensor.hpp>
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif
#include <cassert>
#include <string>
#include <unordered_map>
#include <boost/optional/optional.hpp>
namespace miopen {

// Tensor Helper APIs
std::string
EncodeDataTypesForKey(miopenDataType_t in, miopenDataType_t weights, miopenDataType_t out);

template <class TTo, class TFunc>
size_t
SetDescFromMLDesc(int spatial_dims, TTo& to, const TensorDescriptor& tensor, const TFunc method)
{
    int n, c, d = 1, h, w;
    int ns, cs, hs, ws;

    if(spatial_dims == 3)
        std::tie(n, c, d, h, w) = miopen::tien<5>(tensor.GetLengths(), 1);
    else
        std::tie(n, c, h, w) = miopen::tien<4>(tensor.GetLengths(), 1);

    std::tie(ns, cs, hs, ws) = miopen::tien<4>(tensor.GetStrides(), 0);

    (to.*method)("NCHW", tensor.GetType(), n, c, d, h, w, ns, cs, hs, ws);

    return tensor.GetElementSpace();
}

struct ConvolutionDescriptor;

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
    }

    return "Unknown(" + std::to_string(data_type) + ")";
}

struct ProblemDescription
#if MIOPEN_ENABLE_SQLITE
    : SQLiteSerializable<ProblemDescription>
#endif
{
    int spatial_dims      = 2;
    int n_inputs          = 0;
    int in_height         = 0;
    int in_width          = 0;
    int in_depth          = 0;
    int kernel_size_h     = 0;
    int kernel_size_w     = 0;
    int kernel_size_d     = 0;
    int n_outputs         = 0;
    int out_height        = 0;
    int out_width         = 0;
    int out_depth         = 0;
    int batch_sz          = 0;
    int pad_h             = 0;
    int pad_w             = 0;
    int pad_d             = 0;
    int kernel_stride_h   = 0;
    int kernel_stride_w   = 0;
    int kernel_stride_d   = 0;
    int kernel_dilation_h = 0;
    int kernel_dilation_w = 0;
    int kernel_dilation_d = 0;
    int bias              = 0;
    std::string in_layout;
    std::string weights_layout;
    std::string out_layout;
    miopenDataType_t in_data_type      = static_cast<miopenDataType_t>(-1);
    miopenDataType_t weights_data_type = static_cast<miopenDataType_t>(-1);
    miopenDataType_t out_data_type     = static_cast<miopenDataType_t>(-1);
    size_t bot_sz                      = 0;
    size_t top_sz                      = 0;
    size_t weights_sz                  = 0;
    size_t bias_sz                     = 0;
    int deconvolution                  = 0;
    int in_stride                      = 0;
    int out_stride                     = 0;
    int in_channel_stride              = 0;
    int in_batch_stride                = 0;
    int out_channel_stride             = 0;
    int out_batch_stride               = 0;
    int group_counts                   = 0;

    static std::string table_name() { return "config"; }
    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        if(!self.direction.IsKnown())
            MIOPEN_THROW("!direction.IsKnown()");
        f(std::to_string(self.n_inputs), "in_channels");
        f(std::to_string(self.in_height), "in_h");
        f(std::to_string(self.in_width), "in_w");
        f(std::to_string(self.kernel_size_h), "filter_h");
        f(std::to_string(self.kernel_size_w), "filter_w");
        f(std::to_string(self.n_outputs), "out_channels");
        f(std::to_string(self.batch_sz), "batchsize");
        f(std::to_string(self.pad_h), "pad_h");
        f(std::to_string(self.pad_w), "pad_w");
        f(std::to_string(self.kernel_stride_h), "conv_stride_1");
        f(std::to_string(self.kernel_stride_w), "conv_stride_0");
        f(std::to_string(self.kernel_dilation_h), "dilation_h");
        f(std::to_string(self.kernel_dilation_w), "dilation_w");
        f(std::to_string(self.bias), "bias");
        f("'" + self.in_layout + "'", "layout");
        std::string data_type =
            EncodeDataTypesForKey(self.in_data_type, self.weights_data_type, self.out_data_type);
        f("'" + data_type + "'", "data_type");
        std::string dir =
            self.direction.IsForward() ? "F" : self.direction.IsBackwardData() ? "B" : "W";
        f("'" + dir + "'", "direction");
        f(std::to_string(self.group_counts), "group_count");
    }

    struct Direction
    {
        enum class Value
        {
            Unknown,
            Forward,
            Backward,
            BackwardWrW,
        };

        private:
        Value v = Value::Unknown;

        public:
        bool IsKnown() const { return v != Value::Unknown; }
        bool IsForward() const { return v == Value::Forward; }
        bool IsBackwardData() const { return v == Value::Backward; } // Syntax glue.
        bool IsBackwardWrW() const { return v == Value::BackwardWrW; }
        void Set(int forward)
        {
            assert(0 <= forward && forward <= 1);
            v = forward != 0 ? Value::Forward : Value::Backward;
        }
        template <typename T>
        void Set(T) = delete;
        void SetBackwardWrW() { v = Value::BackwardWrW; }
    } direction;
    int GetBackwardPadW() const { return kernel_size_w - pad_w - 1; }
    int GetBackwardPadH() const { return kernel_size_h - pad_h - 1; }

    bool Is2d() const { return spatial_dims == 2; }

    bool IsFp32() const
    {
        return in_data_type == miopenFloat && weights_data_type == miopenFloat &&
               out_data_type == miopenFloat;
    }
    bool IsFp16() const
    {
        return in_data_type == miopenHalf && weights_data_type == miopenHalf &&
               out_data_type == miopenHalf;
    }
    bool IsBfp16() const
    {
        return in_data_type == miopenBFloat16 && weights_data_type == miopenBFloat16 &&
               out_data_type == miopenBFloat16;
    }

    ProblemDescription() = default;

    ProblemDescription(const TensorDescriptor& in,
                       const TensorDescriptor& weights,
                       const TensorDescriptor& out,
                       const ConvolutionDescriptor& conv,
                       int dir,
                       int bias_ = 0);

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    /*
     * set top tensor
     */
    void setTopDescr(const std::string& layout,
                     miopenDataType_t data_type,
                     int batch,
                     int channels,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        batch_sz     = batch;
        int data_len = GetTypeSize(data_type);
        size_t size  = (layout == "NCHW")
                          ? batch * channels * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        out_width          = width;
        out_height         = height;
        out_depth          = depth;
        n_outputs          = channels;
        out_batch_stride   = batch_stride;
        out_channel_stride = channel_stride;
        out_stride         = stride;
        top_sz             = size;
        out_layout         = layout;
        out_data_type      = data_type;
        bias_sz            = (bias != 0) ? (n_outputs * data_len) : 0;
    }

    /*
     *  set bot tensor
     */

    void setBotDescr(const std::string& layout,
                     miopenDataType_t data_type,
                     int batch,
                     int channels,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        batch_sz     = batch;
        int data_len = GetTypeSize(data_type);
        size_t size  = (layout == "NCHW")
                          ? batch * channels * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        in_width          = width;
        in_height         = height;
        in_depth          = depth;
        n_inputs          = channels;
        in_batch_stride   = batch_stride;
        in_channel_stride = channel_stride;
        in_stride         = stride;
        bot_sz            = size;
        in_layout         = layout;
        in_data_type      = data_type;
        //			_tens_layout = layout;
        //			_tens_data_format = data_type;
    }
    /*
     * set top df tensor
     */
    void setTopDfDescr(const std::string& /*layout*/,
                       miopenDataType_t /*data_type*/,
                       int batch,
                       int channels,
                       int /*depth*/,
                       int /*height*/,
                       int /*width*/,
                       int /*batch_stride*/,
                       int /*channel_stride*/,
                       int /*stride*/,
                       int /*w_stride*/)
    {
        batch_sz  = batch;
        n_outputs = channels;
    }

    /*
     *  set bot df tensor
     */

    void setBotDfDescr(const std::string& /*layout*/,
                       miopenDataType_t /*data_type*/,
                       int batch,
                       int channels,
                       int /*depth*/,
                       int /*height*/,
                       int /*width*/,
                       int /*batch_stride*/,
                       int /*channel_stride*/,
                       int /*stride*/,
                       int /*w_stride*/)
    {
        batch_sz = batch;
        n_inputs = channels;
    }

    int mloBuildConf_Key(std::string& conf_key) const;

    private:
    /*
     * set convolutional parameters
     */
    void setConvDescr(const ConvolutionDescriptor& conv);

    /*
     * set weights tensor
     */
    void setWeightsDescr(const std::string& layout,
                         miopenDataType_t data_type,
                         int batch,
                         int channels,
                         int depth,
                         int height,
                         int width,
                         int batch_stride,
                         int channel_stride,
                         int stride,
                         int w_stride)
    {
        kernel_size_w     = width;
        kernel_size_h     = height;
        kernel_size_d     = depth;
        weights_data_type = data_type;
        int data_len      = GetTypeSize(data_type);
        size_t size       = (layout == "NCHW")
                          ? batch * channels * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        weights_sz = size;
    }

    /*
     * set output tensor
     */
    void setOutputDescr(const std::string& layout,
                        miopenDataType_t data_type,
                        int batch,
                        int channels,
                        int depth,
                        int height,
                        int width,
                        int batch_stride,
                        int channel_stride,
                        int stride,
                        int w_stride)
    {
        batch_sz     = batch;
        int data_len = GetTypeSize(data_type);
        size_t size  = (layout == "NCHW")
                          ? batch * channels * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        if(direction.IsForward())
        {

            out_width          = width;
            out_height         = height;
            out_depth          = depth;
            n_outputs          = channels;
            out_batch_stride   = batch_stride;
            out_channel_stride = channel_stride;
            out_stride         = stride;
            top_sz             = size;
            out_layout         = layout;
            out_data_type      = data_type;
        }
        else
        {
            in_width          = width;
            in_height         = height;
            in_depth          = depth;
            n_inputs          = channels;
            in_batch_stride   = batch_stride;
            in_channel_stride = channel_stride;
            in_stride         = stride;
            bot_sz            = size;
            in_layout         = layout;
            in_data_type      = data_type;
            //			_tens_layout = layout;
            //			_tens_data_format = data_type;
        }
    }

    /*
     *  set input tensor
     */

    void setInputDescr(const std::string& layout,
                       miopenDataType_t data_type,
                       int batch,
                       int channels,
                       int depth,
                       int height,
                       int width,
                       int batch_stride,
                       int channel_stride,
                       int stride,
                       int w_stride)
    {
        batch_sz     = batch;
        int data_len = GetTypeSize(data_type);
        size_t size  = (layout == "NCHW")
                          ? batch * channels * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        if(direction.IsForward())
        {

            in_width          = width;
            in_height         = height;
            in_depth          = depth;
            n_inputs          = channels;
            in_batch_stride   = batch_stride;
            in_channel_stride = channel_stride;
            in_stride         = stride;
            bot_sz            = size;
            in_layout         = layout;
            in_data_type      = data_type;
            //			_tens_layout = layout;
            //			_tens_data_format = data_type;
        }
        else
        {
            out_width          = width;
            out_height         = height;
            out_depth          = depth;
            n_outputs          = channels;
            out_batch_stride   = batch_stride;
            out_channel_stride = channel_stride;
            out_stride         = stride;
            top_sz             = size;
            out_layout         = layout;
            out_data_type      = data_type;
        }

        bias_sz = (bias) != 0 ? n_outputs * data_len : 0;
    }
};
} // namespace miopen

#endif
