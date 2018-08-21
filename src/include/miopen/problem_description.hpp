/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <miopen/errors.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

// Tensor Helper APIs

template <class TTo, class TFunc>
size_t SetDescFromMLDesc(TTo& to, const TensorDescriptor& tensor, const TFunc method)
{
    int n, c, h, w;
    int ns, cs, hs, ws;

    std::tie(n, c, h, w)     = miopen::tien<4>(tensor.GetLengths(), 1);
    std::tie(ns, cs, hs, ws) = miopen::tien<4>(tensor.GetStrides(), 0);
    const auto data_type = tensor.GetType() == miopenFloat ? "FP32" : "FP16";

    (to.*method)("NCHW", data_type, n, c, h, w, ns, cs, hs, ws);

    return tensor.GetElementSpace();
}

struct ConvolutionDescriptor;

struct ProblemDescription
{
    int vec_size         = 1;
    int n_inputs         = 0;
    int in_height        = 0;
    int in_width         = 0;
    int kernel_size1     = 0;
    int kernel_size0     = 0;
    int n_outputs        = 0;
    int out_height       = 0;
    int out_width        = 0;
    int batch_sz         = 0;
    int pad0             = 0;
    int pad1             = 0;
    int kernel_stride0   = 0;
    int kernel_stride1   = 0;
    int kernel_dilation0 = 0;
    int kernel_dilation1 = 0;
    int bias             = 0;
    std::string in_layout;
    std::string in_data_type;
    std::string weights_layout;
    std::string out_data_type;
    std::string out_layout;
    int float_size         = 32;
    size_t bot_sz          = 0;
    size_t top_sz          = 0;
    size_t weights_sz      = 0;
    size_t bias_sz         = 0;
    int deconvolution      = 0;
    int in_stride          = 0;
    int out_stride         = 0;
    int in_channel_stride  = 0;
    int in_batch_stride    = 0;
    int out_channel_stride = 0;
    int out_batch_stride   = 0;
    int group_counts       = 0;
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
    int GetBackwardPad0() const { return kernel_size0 - pad0 - 1; }
    int GetBackwardPad1() const { return kernel_size1 - pad1 - 1; }

    ProblemDescription() = default;

    ProblemDescription(const TensorDescriptor& in,
                       const TensorDescriptor& weights,
                       const TensorDescriptor& out,
                       const ConvolutionDescriptor& conv,
                       int dir,
                       int bias_ = 0);

    void Serialize(std::ostream& stream) const
    {
        if(!direction.IsKnown())
            MIOPEN_THROW("!direction.IsKnown()");
        const auto sep = '-';
        // clang-format off
        // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NCHW-FP32-F
        stream
            << n_inputs << sep << in_height << sep << in_width
            << sep << kernel_size1 << 'x' << kernel_size0
            << sep << n_outputs << sep << out_height << sep << out_width
            << sep << batch_sz
            << sep << pad1 << 'x' << pad0
            << sep << kernel_stride1 << 'x' << kernel_stride0
            << sep << kernel_dilation1 << 'x' << kernel_dilation1
            << sep << bias
            << sep << in_layout
            << sep << in_data_type
            << sep << (direction.IsForward() ? "F"
                     : direction.IsBackwardData() ? "B" : "W"); // clang-format on
    }

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    /*
     * set top tensor
     */
    void setTopDescr(const std::string& layout,
                     const std::string& data_type,
                     int batch,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        batch_sz     = batch;
        int data_len = (data_type == "FP16") ? 2 : (data_type == "FP32") ? 4 : 8;
        float_size   = (data_type == "FP32" ? 32 : 16);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        out_width          = width;
        out_height         = height;
        n_outputs          = depth;
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
                     const std::string& data_type,
                     int batch,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        batch_sz     = batch;
        int data_len = (data_type == "FP16") ? 2 : (data_type == "FP32") ? 4 : 8;
        float_size   = (data_type == "FP32" ? 32 : 16);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        in_width          = width;
        in_height         = height;
        n_inputs          = depth;
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
                       const std::string& data_type,
                       int batch,
                       int depth,
                       int /*height*/,
                       int /*width*/,
                       int /*batch_stride*/,
                       int /*channel_stride*/,
                       int /*stride*/,
                       int /*w_stride*/)
    {
        batch_sz   = batch;
        float_size = (data_type == "FP32" ? 32 : 16);
        n_outputs  = depth;
    }

    /*
     *  set bot df tensor
     */

    void setBotDfDescr(const std::string& /*layout*/,
                       const std::string& data_type,
                       int batch,
                       int depth,
                       int /*height*/,
                       int /*width*/,
                       int /*batch_stride*/,
                       int /*channel_stride*/,
                       int /*stride*/,
                       int /*w_stride*/)
    {
        batch_sz   = batch;
        float_size = (data_type == "FP32" ? 32 : 16);
        n_inputs   = depth;
    }

    int mloBuildConf_Key(std::string& conf_key) const;

    private:
    /*
     * set convolutional parameters
     */
    void setConvDescr(
        int u_padding, int v_padding, int u_stride, int v_stride, int h_dilation, int w_dilation)
    {
        pad1             = u_padding;
        pad0             = v_padding;
        kernel_stride0   = u_stride;
        kernel_stride1   = v_stride;
        kernel_dilation0 = h_dilation;
        kernel_dilation1 = w_dilation;
    }

    /*
     * set weights tensor
     */
    void setWeightsDescr(const std::string& layout,
                         const std::string& data_type,
                         int batch,
                         int depth,
                         int height,
                         int width,
                         int batch_stride,
                         int channel_stride,
                         int stride,
                         int w_stride)
    {
        kernel_size0 = width;
        kernel_size1 = height;
        int data_len = (data_type == "FP16") ? 2 : (data_type == "FP32") ? 4 : 8;
        float_size   = (data_type == "FP32" ? 32 : 16);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        weights_sz = size;
        vec_size   = 32 / float_size;
    }

    /*
     * set output tensor
     */
    void setOutputDescr(const std::string& layout,
                        const std::string& data_type,
                        int batch,
                        int depth,
                        int height,
                        int width,
                        int batch_stride,
                        int channel_stride,
                        int stride,
                        int w_stride)
    {
        batch_sz     = batch;
        int data_len = (data_type == "FP16") ? 2 : (data_type == "FP32") ? 4 : 8;
        float_size   = (data_type == "FP32" ? 32 : 16);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        if(direction.IsForward())
        {

            out_width          = width;
            out_height         = height;
            n_outputs          = depth;
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
            n_inputs          = depth;
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
                       const std::string& data_type,
                       int batch,
                       int depth,
                       int height,
                       int width,
                       int batch_stride,
                       int channel_stride,
                       int stride,
                       int w_stride)
    {
        batch_sz     = batch;
        int data_len = (data_type == "FP16") ? 2 : (data_type == "FP32") ? 4 : 8;
        float_size   = (data_type == "FP32" ? 32 : 16);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        if(direction.IsForward())
        {

            in_width          = width;
            in_height         = height;
            n_inputs          = depth;
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
            n_outputs          = depth;
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
