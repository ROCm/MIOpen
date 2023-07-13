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
#include <miopen/names.hpp>
#include <miopen/tensor.hpp>
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif

#include <cassert>
#include <cstdint>
#include <string>

#define FIN_OLD_PROBLEM_DESCRIPTION_COMPAT 1

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

struct ConvolutionDescriptor;

// Todo: change all uses in convolution to conv::ProblemDescription and remove this
struct ProblemDescription : conv::ProblemDescription
{
    int GetSpatialDims2() const { return GetSpatialDims1(); }
    int GetInChannels2() const { return GetInChannels1(); }
    int GetInHeight2() const { return GetInHeight1(); }
    int GetInWidth2() const { return GetInWidth1(); }
    int GetInDepth2() const { return GetInDepth1(); }
    int GetWeightsHeight2() const { return GetWeightsHeight1(); }
    int GetWeightsWidth2() const { return GetWeightsWidth1(); }
    int GetWeightsDepth2() const { return GetWeightsDepth1(); }
    int GetOutChannels2() const { return GetOutChannels1(); }
    int GetOutHeight2() const { return GetOutHeight1(); }
    int GetOutWidth2() const { return GetOutWidth1(); }
    int GetOutDepth2() const { return GetOutDepth1(); }
    int GetBatchSize2() const { return GetBatchSize1(); }
    int GetInStride2() const { return GetInStrideH1(); }
    int GetOutStride2() const { return GetOutStrideH1(); }
    int GetInChannelStride2() const { return GetInChannelStride1(); }
    int GetInBatchStride2() const { return GetInBatchStride1(); }
    int GetOutChannelStride2() const { return GetOutChannelStride1(); }
    int GetOutBatchStride2() const { return GetOutBatchStride1(); }

    struct Direction
    {
    public:
        bool IsForward() const { return v == conv::Direction::Forward; }
        bool IsBackwardData() const { return v == conv::Direction::BackwardData; }
        bool IsBackwardWrW() const { return v == conv::Direction::BackwardWeights; }

        std::string GetStr() const { return IsForward() ? "F" : IsBackwardData() ? "B" : "W"; }

        Direction() = default;
        Direction(conv::Direction value) : v(value) {}

    private:
        conv::Direction v = conv::Direction::Forward;
    } direction;

    ProblemDescription() = default;

    ProblemDescription(conv::ProblemDescription desc);

#if 0
    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }
#endif

#if FIN_OLD_PROBLEM_DESCRIPTION_COMPAT
    struct
    {
        void SetupFloats(ExecutionContext& ctx) const { p->SetupFloats(ctx); }
    private:
        const conv::ProblemDescription* p = nullptr;
        friend struct ProblemDescription;
    } conv_problem;
#endif
};

// For mlo_construct_base, SQLitePerfDb and test_sqlite_perfdb
// TODO remove this
struct ProblemDescriptionCompatTemporary
#if MIOPEN_ENABLE_SQLITE
    : SQLiteSerializable<ProblemDescriptionCompatTemporary>
#endif
{
    int spatial_dims = 2;
    int n_inputs     = 0;
    int in_height    = 0;
    int in_width     = 0;
    int in_depth     = 0;
    // TODO add check to solver that vectorLength = 1
    // int vectorLength      = 1;
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
    // std::string weights_layout;
    std::string out_layout;
    miopenDataType_t in_data_type      = miopenFloat;
    miopenDataType_t weights_data_type = miopenFloat;
    miopenDataType_t out_data_type     = miopenFloat;
    size_t bot_sz                      = 0;
    size_t top_sz                      = 0;
    size_t weights_sz                  = 0;
    size_t bias_sz                     = 0;
    int in_stride                      = 0; // GetInStrideH()
    int out_stride                     = 0; // GetOutStrideH()
    int in_channel_stride              = 0;
    int in_batch_stride                = 0;
    int out_channel_stride             = 0;
    int out_batch_stride               = 0;
    int group_counts                   = 0;

    int GetSpatialDims() const { return spatial_dims; }
    int GetSpatialDims1() const { return GetSpatialDims(); }
    int GetInChannels() const { return n_inputs; }
    int GetInChannels1() const { return GetInChannels(); }
    int GetInHeight() const { return in_height; }
    int GetInHeight1() const { return GetInHeight(); }
    int GetInWidth() const { return in_width; }
    int GetInWidth1() const { return GetInWidth(); }
    int GetInDepth1() const { return in_depth; }
    // int GetVectorLength() const { return vectorLength; }
    int GetWeightsHeight1() const { return kernel_size_h; }
    int GetWeightsWidth1() const { return kernel_size_w; }
    int GetWeightsDepth1() const { return kernel_size_d; }
    int GetOutChannels() const { return n_outputs; }
    int GetOutChannels1() const { return GetOutChannels(); }
    int GetOutHeight() const { return out_height; }
    int GetOutWidth() const { return out_width; }
    // int GetOutDepth() const { return out_depth; }
    int GetBatchSize() const { return batch_sz; }
    int GetBatchSize1() const { return GetBatchSize(); }
    int GetPadH() const { return pad_h; }
    int GetPadW() const { return pad_w; }
    int GetPadD() const { return pad_d; }
    int GetKernelStrideH() const { return kernel_stride_h; }
    int GetKernelStrideW() const { return kernel_stride_w; }
    int GetKernelStrideD() const { return kernel_stride_d; }
    int GetDilationH() const { return kernel_dilation_h; }
    int GetDilationW() const { return kernel_dilation_w; }
    int GetDilationD() const { return kernel_dilation_d; }
    int GetBias() const { return bias; }
    std::string GetInLayout() const { return in_layout; }
    // std::string GetWeightsLayout() const { return weights_layout; }
    // std::string GetOutLayout() const { return out_layout; }
    miopenDataType_t GetInDataType() const { return in_data_type; }
    miopenDataType_t GetWeightsDataType() const { return weights_data_type; }
    miopenDataType_t GetOutDataType() const { return out_data_type; }
    // size_t GetInSize() const { return bot_sz; }
    // size_t GetOutSize() const { return top_sz; }
    // size_t GetWeightsSize() const { return weights_sz; }
    // size_t GetBiasSize() const { return bias_sz; }
    int GetInStride() const { return in_stride; }
    int GetOutStride() const { return out_stride; }
    int GetInChannelStride() const { return in_channel_stride; }
    int GetInBatchStride() const { return in_batch_stride; }
    int GetOutChannelStride() const { return out_channel_stride; }
    int GetOutBatchStride() const { return out_batch_stride; }
    int GetGroupCount() const { return group_counts; }

#if MIOPEN_ENABLE_SQLITE
    static std::string table_name() { return ProblemDescription::table_name(); }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        ProblemDescription::Visit(self, f);
    }
#endif

    ProblemDescriptionCompatTemporary() = default;
    ProblemDescriptionCompatTemporary(miopen::conv::Direction dir) : direction(dir) {}

    ProblemDescription::Direction direction;

    std::string GetDirectionStr() const { return direction.GetStr(); }

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

    UnifiedDescriptionConv2d(const ProblemDescription& problem)
    {
        if(!problem.Is2d())
            MIOPEN_THROW(miopenStatusInternalError, "UnifiedDescriptionConv2d supports only 2D");

        const auto n_inputs_per_group  = problem.GetInChannels2() / problem.GetGroupCount();
        const auto n_outputs_per_group = problem.GetOutChannels2() / problem.GetGroupCount();
        if(!problem.direction.IsBackwardWrW())
        {
            R     = problem.GetWeightsHeight2();
            S     = problem.GetWeightsWidth2();
            U     = problem.direction.IsForward() ? problem.GetKernelStrideH() : 1;
            V     = problem.direction.IsForward() ? problem.GetKernelStrideW() : 1;
            C     = n_inputs_per_group;      // Bwd: C and K is reversed in ProblemDescription.
            K     = n_outputs_per_group;     // Ditto.
            out_h = problem.GetOutHeight2(); // Bwd: height/width is reversed in ProblemDescription.
            out_w = problem.GetOutWidth2();  // Ditto.
            N     = problem.GetBatchSize2();
            pad_h = problem.direction.IsForward() ? problem.GetPadH() : problem.GetBackwardPadH();
            pad_w = problem.direction.IsForward() ? problem.GetPadW() : problem.GetBackwardPadW();
            input_stride_h  = problem.direction.IsForward() ? 1 : problem.GetKernelStrideH();
            input_stride_w  = problem.direction.IsForward() ? 1 : problem.GetKernelStrideW();
            filter_stride_h = problem.GetDilationH();
            filter_stride_w = problem.GetDilationW();
        }
        else
        { // WrW
            R               = problem.GetInHeight2();
            S               = problem.GetInWidth2();
            U               = problem.GetDilationH();
            V               = problem.GetDilationW();
            C               = problem.GetBatchSize2();
            K               = n_inputs_per_group;
            out_h           = problem.GetWeightsHeight2();
            out_w           = problem.GetWeightsWidth2();
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
