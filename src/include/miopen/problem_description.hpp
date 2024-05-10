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

#include <miopen/conv/problem_description.hpp>
#include <miopen/tensor.hpp>

#include <cstdint>
#include <string>

namespace miopen {

// Tensor Helper APIs
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

// For mlo_construct_base
// TODO remove this
struct ProblemDescriptionCompatTemporary
{
    int spatial_dims = 2;
    int n_inputs     = 0;
    int in_height    = 0;
    int in_width     = 0;
    int in_depth     = 0;
    // TODO add check to solver that vectorLength = 1
    // int vectorLength      = 1;
    int n_outputs  = 0;
    int out_height = 0;
    int out_width  = 0;
    int out_depth  = 0;
    int batch_sz   = 0;
    int bias       = 0;
    std::string in_layout;
    std::string out_layout;
    miopenDataType_t in_data_type  = miopenFloat;
    miopenDataType_t out_data_type = miopenFloat;
    size_t bot_sz                  = 0;
    size_t top_sz                  = 0;
    size_t bias_sz                 = 0;
    int in_stride                  = 0; // GetInStrideH()
    int out_stride                 = 0; // GetOutStrideH()
    int in_channel_stride          = 0;
    int in_batch_stride            = 0;
    int out_channel_stride         = 0;
    int out_batch_stride           = 0;

    int GetSpatialDims() const { return spatial_dims; }
    int GetInChannels() const { return n_inputs; }
    int GetInHeight() const { return in_height; }
    int GetInWidth() const { return in_width; }
    // int GetInDepth() const { return in_depth; }
    // int GetVectorLength() const { return vectorLength; }
    int GetOutChannels() const { return n_outputs; }
    int GetOutHeight() const { return out_height; }
    int GetOutWidth() const { return out_width; }
    // int GetOutDepth() const { return out_depth; }
    int GetBatchSize() const { return batch_sz; }
    int GetBias() const { return bias; }
    // std::string GetInLayout() const { return in_layout; }
    // std::string GetOutLayout() const { return out_layout; }
    miopenDataType_t GetInDataType() const { return in_data_type; }
    miopenDataType_t GetOutDataType() const { return out_data_type; }
    // size_t GetInSize() const { return bot_sz; }
    // size_t GetOutSize() const { return top_sz; }
    // size_t GetBiasSize() const { return bias_sz; }
    int GetInStride() const { return in_stride; }
    int GetOutStride() const { return out_stride; }
    int GetInChannelStride() const { return in_channel_stride; }
    int GetInBatchStride() const { return in_batch_stride; }
    int GetOutChannelStride() const { return out_channel_stride; }
    int GetOutBatchStride() const { return out_batch_stride; }

    ProblemDescriptionCompatTemporary(conv::Direction dir) : direction(dir) {}

    bool IsDirectionForward() const { return direction == conv::Direction::Forward; }

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
        batch_sz           = batch;
        const int data_len = GetTypeSize(data_type);
        const size_t size =
            (layout == "NCHW")
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
        batch_sz           = batch;
        const int data_len = GetTypeSize(data_type);
        const size_t size =
            (layout == "NCHW")
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

private:
    conv::Direction direction;
};

struct UnifiedDescriptionConv2d
{
    size_t K;
    size_t S;
    size_t C;
    size_t N;
    size_t R;
    int64_t pad_w; // Negative padding is possible for Bwd.
    int64_t pad_h;
    size_t U;
    size_t V;
    size_t out_w;
    size_t out_h;
    size_t input_stride_w;
    size_t input_stride_h;
    size_t filter_stride_w;
    size_t filter_stride_h;

    UnifiedDescriptionConv2d() = delete;

    // KT      XLS             DRIVER                                  PROBLEM DESCRIPTION
    // -----------------------------------------------------------------------------------
    // fdil := filter_stride   -l/j filter dilation                    kernel_dilation
    // strd := U/V             -u/v convolution stride (output stride) kernel_stride
    // idil := input dilation  (n/a except transposed convolutions)    ?

    UnifiedDescriptionConv2d(const conv::ProblemDescription& problem)
    {
        if(!problem.Is2d())
            MIOPEN_THROW(miopenStatusInternalError, "UnifiedDescriptionConv2d supports only 2D");

        const auto n_inputs_per_group  = problem.GetInChannels() / problem.GetGroupCount();
        const auto n_outputs_per_group = problem.GetOutChannels() / problem.GetGroupCount();
        if(!problem.IsDirectionBackwardWrW())
        {
            R     = problem.GetWeightsHeight();
            S     = problem.GetWeightsWidth();
            U     = problem.IsDirectionForward() ? problem.GetKernelStrideH() : 1;
            V     = problem.IsDirectionForward() ? problem.GetKernelStrideW() : 1;
            C     = n_inputs_per_group;     // Bwd: C and K is reversed in ProblemDescription.
            K     = n_outputs_per_group;    // Ditto.
            out_h = problem.GetOutHeight(); // Bwd: height/width is reversed in ProblemDescription.
            out_w = problem.GetOutWidth();  // Ditto.
            N     = problem.GetBatchSize();
            pad_h = problem.IsDirectionForward() ? problem.GetPadH() : problem.GetBackwardPadH();
            pad_w = problem.IsDirectionForward() ? problem.GetPadW() : problem.GetBackwardPadW();
            input_stride_h  = problem.IsDirectionForward() ? 1 : problem.GetKernelStrideH();
            input_stride_w  = problem.IsDirectionForward() ? 1 : problem.GetKernelStrideW();
            filter_stride_h = problem.GetDilationH();
            filter_stride_w = problem.GetDilationW();
        }
        else
        { // WrW
            R               = problem.GetInHeight();
            S               = problem.GetInWidth();
            U               = problem.GetDilationH();
            V               = problem.GetDilationW();
            C               = problem.GetBatchSize();
            K               = n_inputs_per_group;
            out_h           = problem.GetWeightsHeight();
            out_w           = problem.GetWeightsWidth();
            N               = n_outputs_per_group;
            pad_h           = problem.GetPadH();
            pad_w           = problem.GetPadW();
            input_stride_h  = 1;
            input_stride_w  = 1;
            filter_stride_h = problem.GetKernelStrideH();
            filter_stride_w = problem.GetKernelStrideW();
        }
    }
};

} // namespace miopen

#endif
