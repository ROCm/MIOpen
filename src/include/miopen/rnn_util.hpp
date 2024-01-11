/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_RNN_UTIL_HPP_
#define GUARD_MIOPEN_RNN_UTIL_HPP_

#include <miopen/rnn.hpp>
#include <miopen/miopen.h>
#include <miopen/common.hpp>
#include <miopen/handle.hpp>
#include <miopen/seq_tensor.hpp>

namespace miopen {

enum class RnnDirection
{
    Forward  = 0,
    Backward = 1
};

struct RnnBatches
{

    int at(int time, RnnDirection direction) const { return batches.at(cur_time(time, direction)); }

    int next(int time, RnnDirection direction) const
    {
        return batches.at(next_time(time, direction));
    }

    int prev(int time, RnnDirection direction) const
    {
        return batches.at(prev_time(time, direction));
    }

    void push_back(int batch) { batches.push_back(batch); }

    RnnBatches(std::vector<int>& input) : batches(input){};
    RnnBatches(){};

    int back() const { return batches.back(); }

private:
    int cur_time(int time, RnnDirection direction) const
    {
        return direction == RnnDirection::Forward ? time : batches.size() - time - 1;
    }

    int next_time(int time, RnnDirection direction) const
    {
        return direction == RnnDirection::Forward ? cur_time(time, direction) + 1
                                                  : cur_time(time, direction) - 1;
    }

    int prev_time(int time, RnnDirection direction) const
    {
        return direction == RnnDirection::Forward ? cur_time(time, direction) - 1
                                                  : cur_time(time, direction) + 1;
    }

    std::vector<int> batches;
};

#if MIOPEN_BACKEND_HIP
inline void RNNProfilingBegin(const miopen::Handle& handle,
                              miopen::HipEventPtr& start,
                              miopen::HipEventPtr& stop)
{
    start = miopen::make_hip_event();
    stop  = miopen::make_hip_event();
    hipEventRecord(start.get(), handle.GetStream());
}

inline float
RNNProfilingEnd(const miopen::Handle& handle, miopen::HipEventPtr& start, miopen::HipEventPtr& stop)
{
    hipEventRecord(stop.get(), handle.GetStream());
    hipEventSynchronize(stop.get());
    float mS = 0;
    hipEventElapsedTime(&mS, start.get(), stop.get());
    return mS;
}

inline miopen::HipEventPtr make_hip_fast_event()
{
    hipEvent_t result = nullptr;
    hipEventCreateWithFlags(&result, hipEventDisableTiming);
    return miopen::HipEventPtr{result};
}
#endif //#if MIOPEN_BACKEND_HIP

void LSTMForwardHiddenStateUpdate(const Handle& handle,
                                  miopenDataType_t rnn_data_type,
                                  bool is_inference,
                                  bool is_seq_begin,
                                  int direction,
                                  int max_batch,
                                  int cur_batch,
                                  int use_batch,
                                  int hy_h,
                                  int hy_stride,
                                  int wei_len,
                                  int wei_stride,
                                  ConstData_t cx,
                                  std::size_t cx_offset,
                                  Data_t reserve_space,
                                  std::size_t i_offset,
                                  std::size_t f_offset,
                                  std::size_t o_offset,
                                  std::size_t c_offset,
                                  std::size_t cell_offset,
                                  std::size_t cell_offset_pre,
                                  std::size_t activ_cell_offset,
                                  std::size_t hidden_offset);

void LSTMBackwardHiddenStateUpdate(const Handle& handle,
                                   miopenDataType_t rnn_data_type,
                                   bool is_seq_begin,
                                   bool is_seq_end,
                                   int direction,
                                   int max_batch,
                                   int cur_batch,
                                   int use_batch,
                                   int use_batch2,
                                   int hy_h,
                                   int hy_stride,
                                   int wei_len,
                                   int wei_stride,
                                   ConstData_t cx,
                                   std::size_t cx_offset,
                                   Data_t reserve_space,
                                   std::size_t i_offset,
                                   std::size_t f_offset,
                                   std::size_t o_offset,
                                   std::size_t c_offset,
                                   std::size_t activ_cell_offset,
                                   std::size_t cell_offset_pre,
                                   ConstData_t dcy,
                                   std::size_t dcy_offset,
                                   Data_t work_space,
                                   std::size_t di_offset,
                                   std::size_t df_offset,
                                   std::size_t do_offset,
                                   std::size_t dc_offset,
                                   std::size_t dcell_offset,
                                   std::size_t dcell_offset_pre,
                                   std::size_t dhidden_offset,
                                   std::size_t f_offset_pre);

struct ReluWeightOffsets
{
public:
    ReluWeightOffsets(int input_vector_sz,
                      int hidden_vec_sz,
                      int layers_cnt,
                      int bias_mode,
                      int bi,
                      int nHiddenTensorsPerLayer)
        : weight_stride(hidden_vec_sz * bi * nHiddenTensorsPerLayer),
          in_vec_sz(input_vector_sz),
          h_vec_sz(hidden_vec_sz),
          num_layers(layers_cnt),
          bi_scale(bi),
          bias_count(bias_mode)
    {
    }

    int input_weight_offset(int layer) const
    {
        return layer == 0 ? 0
                          : first_layer_offset() +
                                (h_vec_sz + h_vec_sz * bi_scale) * weight_stride * (layer - 1);
    }

    int hidden_weight_offset(int layer, RnnDirection reverse) const
    {
        return layer == 0 ? input_weight_offset(layer) + in_vec_sz * weight_stride +
                                static_cast<int>(reverse) * h_vec_sz * h_vec_sz
                          : input_weight_offset(layer) + bi_scale * h_vec_sz * weight_stride +
                                static_cast<int>(reverse) * h_vec_sz * h_vec_sz;
    }

    size_t bias_stride() const { return static_cast<size_t>(h_vec_sz) * bi_scale; }

    int bias_off() const
    {
        return first_layer_offset() +
               (h_vec_sz * bi_scale + h_vec_sz) * (num_layers - 1) * weight_stride;
    }

    int bias_off(int layer_id) const { return bias_off() + bias_count * layer_id * weight_stride; }
    int weight_stride;

private:
    const int in_vec_sz, h_vec_sz;

public:
    const int num_layers;
    const int bi_scale   = 1;
    const int bias_count = 0;

    int first_layer_offset() const { return (in_vec_sz + h_vec_sz) * weight_stride; }
};

struct ReluReserveBufferOffsets
{
    struct RBuffHelper
    {
        int element, save_point, batch;
        size_t layer, table;
    };

private:
    auto Reserve_Buffer_strides(int save_point_sz, int batches_per_l, int layers_cnt) const
    {
        const auto element_st    = 1;
        const auto save_point_st = element_st * save_point_sz;
        const auto batch_st      = save_point_st;
        const auto layer_st      = static_cast<size_t>(batch_st) * batches_per_l;
        const auto table_st      = layers_cnt * layer_st;

        return RBuffHelper{element_st, save_point_st, batch_st, layer_st, table_st};
    }

public:
    ReluReserveBufferOffsets(
        int hidden_vec_size, int layers_cnt, int batches_per_l, int bi_scale, int workspace_scale)
        : hidden_size(hidden_vec_size),
          batches_per_layer(batches_per_l),
          save_point_size(hidden_vec_size * bi_scale * workspace_scale),
          layers(layers_cnt),
          strides(Reserve_Buffer_strides(save_point_size, batches_per_l, layers_cnt))
    {
    }

    size_t layer_offset(int layer_id) const
    {
        return static_cast<size_t>(layer_id) * strides.layer;
    }

    size_t layer_stride() const { return strides.layer; }

    int gemm_write_size() const { return strides.save_point; }

    size_t gemm_write_stride() const { return strides.batch; }

    size_t gemm_write_offset(int layer_id, int batch_id, RnnDirection reverse) const
    {
        return layer_offset(layer_id) + static_cast<size_t>(gemm_write_stride()) * batch_id +
               static_cast<size_t>(reverse) * hidden_size;
    }

    size_t hidden_offset(int layer_id, int batch_id, RnnDirection reverse) const
    {
        return strides.table + gemm_write_offset(layer_id, batch_id, reverse);
    }

private:
    const int hidden_size;

public:
    const int batches_per_layer;
    const int save_point_size;
    const int layers;
    const RBuffHelper strides;
};

struct RNNTensorPaddingConverter
{
    static void ConvertTensorData(const Handle& handle,
                                  const TensorDescriptor& padded_tensor_desc,
                                  std::vector<int>& bsize_per_time,
                                  ConstData_t src,
                                  Data_t dst,
                                  bool is_src_padded);

    static std::tuple<size_t, size_t>
    GetTempPackedBuffersSpace(RNNDescriptor rnn_desc, size_t total_batch, size_t in_vec)
    {
        auto type_size       = GetTypeSize(rnn_desc.dataType);
        size_t in_buff_size  = type_size * total_batch * in_vec;
        size_t out_buff_size = type_size * total_batch * rnn_desc.hsize *
                               (rnn_desc.dirMode == miopenRNNbidirection ? 2 : 1);
        return {in_buff_size, out_buff_size};
    }

    static std::tuple<size_t, size_t>
    GetTempPackedBuffersSpace(RNNDescriptor rnn_desc,
                              c_array_view<const miopenTensorDescriptor_t> desc_array)
    {
        size_t total_batch = std::accumulate(
            desc_array.data,
            desc_array.data + desc_array.size(),
            0,
            [](size_t x, miopenTensorDescriptor_t y) { return x + deref(y).GetLengths()[0]; });

        return GetTempPackedBuffersSpace(rnn_desc, total_batch, desc_array[0].GetLengths()[1]);
    }
};

struct RNNTensorBaseLayoutConverter
{
    static void ConvertInputTensorGPUData(const Handle& handle,
                                          const SeqTensorDescriptor& src_tensor_desc,
                                          ConstData_t src,
                                          const SeqTensorDescriptor& dst_tensor_desc,
                                          Data_t dst,
                                          Data_t workspace,
                                          bool reverse);

    static void ReverseConvertInputTensorGPUData(const Handle& handle,
                                                 const SeqTensorDescriptor& src_tensor_desc,
                                                 ConstData_t src,
                                                 const SeqTensorDescriptor& dst_tensor_desc,
                                                 Data_t dst,
                                                 Data_t workspace)
    {
        ConvertInputTensorGPUData(
            handle, src_tensor_desc, src, dst_tensor_desc, dst, workspace, true);
    }

    static void ReorderHiddenTensorGPUData(const Handle& handle,
                                           const TensorDescriptor& tensor_desc,
                                           int reordering_dim,
                                           std::vector<size_t> sample_order,
                                           ConstData_t src,
                                           Data_t dst);

    static void ReorderInputTensorGPUData(const Handle& handle,
                                          const SeqTensorDescriptor& padded_tensor_desc,
                                          const std::vector<size_t>& sample_order,
                                          const SeqTensorDescriptor& dst_padded_tensor_desc,
                                          ConstData_t src,
                                          Data_t dst);

    static std::vector<size_t> GetSamplesDescendingOrder(const SeqTensorDescriptor& desc,
                                                         bool reverse = false);

    static std::tuple<size_t, size_t>
    GetTempPackedBuffersSpace(RNNDescriptor rnn_desc, size_t total_batch, size_t in_vec)
    {
        auto type_size       = GetTypeSize(rnn_desc.dataType);
        size_t in_buff_size  = type_size * total_batch * in_vec;
        size_t out_buff_size = type_size * total_batch * rnn_desc.hsize *
                               (rnn_desc.dirMode == miopenRNNbidirection ? 2 : 1);
        return {in_buff_size, out_buff_size};
    }

    static std::tuple<size_t, size_t>
    GetTempPackedBuffersSpace(RNNDescriptor rnn_desc,
                              c_array_view<const miopenTensorDescriptor_t> desc_array)
    {
        size_t total_batch = std::accumulate(
            desc_array.data,
            desc_array.data + desc_array.size(),
            0,
            [](size_t x, miopenTensorDescriptor_t y) { return x + deref(y).GetLengths()[0]; });

        return GetTempPackedBuffersSpace(rnn_desc, total_batch, desc_array[0].GetLengths()[1]);
    }

    static size_t GetWorkspaceSize(const SeqTensorDescriptor& src_tensor_desc,
                                   const SeqTensorDescriptor& dst_tensor_desc,
                                   bool reverse)
    {
        const size_t WorkspaceSize = std::max(src_tensor_desc.GetTensorMaxByteSpace(),
                                              dst_tensor_desc.GetTensorMaxByteSpace());

        auto src_layout = RNNDescriptor::getBaseLayoutFromDataTensor(src_tensor_desc);
        auto dst_layout = RNNDescriptor::getBaseLayoutFromDataTensor(dst_tensor_desc);

        if(src_layout == miopenRNNDataBatchMajorPadded)
        {
            if(dst_layout == miopenRNNDataSeqMajorPadded)
                return 0;
            if(dst_layout == miopenRNNDataSeqMajorNotPadded)
                return WorkspaceSize * 2;
        }
        else if(src_layout == miopenRNNDataSeqMajorPadded)
        {
            if(dst_layout == miopenRNNDataBatchMajorPadded)
                return 0;
            if(dst_layout == miopenRNNDataSeqMajorNotPadded)
                return WorkspaceSize;
        }
        else if(src_layout == miopenRNNDataSeqMajorNotPadded)
        {
            if(dst_layout == miopenRNNDataBatchMajorPadded)
                return WorkspaceSize * (reverse ? 2 : 1);
            if(dst_layout == miopenRNNDataSeqMajorPadded)
                return WorkspaceSize * (reverse ? 1 : 0);
        }
        MIOPEN_THROW(miopenStatusInternalError, "Unsupported layout.");
    }

    static std::vector<size_t> GetSortedLens(const SeqTensorDescriptor& desc)
    {
        std::vector<size_t> reordered_vec = desc.GetSequenceLengthsVector();
        sort(reordered_vec.begin(), reordered_vec.end(), std::greater<>());
        return reordered_vec;
    }

private:
    static void ChangeTensorGPUDataPadding(const Handle& handle,
                                           const SeqTensorDescriptor& tensor_desc,
                                           ConstData_t src,
                                           Data_t dst);
    static void ChangePaddedTensorGPUDataLayout(const Handle& handle,
                                                const SeqTensorDescriptor& src_padded_desc,
                                                ConstData_t src,
                                                const SeqTensorDescriptor& dst_padded_desc,
                                                Data_t dst);
};

void FillSeqTensorByPaddingMarker(const Handle& handle,
                                  const SeqTensorDescriptor& desc,
                                  Data_t data);

} // namespace miopen

#endif // GUARD_MIOPEN_RNN_UTIL_HPP_
