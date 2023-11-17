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

namespace miopen {

enum rnn_direction
{
    Forward  = 0,
    Backward = 1
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

struct RNNWeightOffsets
{

public:
    int input_offset(int layer) const;
    int hidden_offset(int layer) const;
    int bias_off();
    int bias_off(int layer) const;

private:
    int first_layer_offset() const;
};

struct GruWeightOffsets : public RNNWeightOffsets
{
    GruWeightOffsets(int input_vector_sz, int hidden_vec_sz, int layers_cnt, int bias_cnt)
        : weight_stride(matrixes::Count * hidden_vec_sz),
          in_vec_sz(input_vector_sz),
          h_vec_sz(hidden_vec_sz),
          num_layers(layers_cnt),
          bias_count(bias_cnt)
    {
    }

    int input_offset(int layer)
    {
        return layer == 0 ? 0 : first_layer_offset() + h_vec_sz * 2 * weight_stride * (layer - 1);
    }

    int hidden_offset(int layer)
    {
        return layer == 0 ? input_offset(layer) + in_vec_sz * weight_stride
                          : input_offset(layer) + h_vec_sz * weight_stride;
    }

    size_t bias_stride() { return matrixes::Count * h_vec_sz; }
    int bias_off()
    {
        return (in_vec_sz + h_vec_sz + bias_count * h_vec_sz * (num_layers - 1)) * weight_stride;
    }
    int bias_off(int layer_id) { return bias_off() + layer_id * bias_count * weight_stride; }
    int weight_stride;

private:
    const int in_vec_sz, h_vec_sz;
    const int num_layers;
    [[maybe_unused]] const int bi_scale = 0;
    const int bias_count                = 0;
    enum matrixes
    {
        Z     = 0,
        R     = 1,
        C     = 2,
        Count = 3
    };
    int first_layer_offset() { return (in_vec_sz + h_vec_sz) * weight_stride; }
};

struct ReluWeightOffsets : public RNNWeightOffsets
{
public:
    ReluWeightOffsets(int input_vector_sz,
                      int hidden_vec_sz,
                      int layers_cnt,
                      int bias_mode,
                      int bi,
                      int wei_stride)
        : weight_stride(wei_stride),
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

    int hidden_weight_offset(int layer, int reverse = 0) const
    {
        return layer == 0 ? input_weight_offset(layer) + in_vec_sz * weight_stride +
                                reverse * h_vec_sz * h_vec_sz
                          : input_weight_offset(layer) + bi_scale * h_vec_sz * weight_stride +
                                reverse * h_vec_sz * h_vec_sz;
    }

    size_t bias_stride() { return h_vec_sz; }

    int bias_off()
    {
        return first_layer_offset() +
               (h_vec_sz * bi_scale + h_vec_sz) * (num_layers - 1) * weight_stride;
    }

    int bias_off(int layer_id) { return bias_off() + bias_count * layer_id * weight_stride; }
    int weight_stride;

private:
    const int in_vec_sz, h_vec_sz;

public:
    const int num_layers;
    const int bi_scale   = 1;
    const int bias_count = 0;

    int first_layer_offset() const { return (in_vec_sz + h_vec_sz) * weight_stride; }
};

struct LSTMWeightsBufferHelper : public RNNWeightOffsets
{
public:
    const int first_layer_offset() const { return (in_vec_sz + h_vec_sz) * weight_stride; }

public:
    LSTMWeightsBufferHelper(
        int input_vector_sz, int hidden_vec_sz, int layers_cnt, int bias_mode, int bi)
        : weight_stride(hidden_vec_sz * gates_cnt),
          in_vec_sz(input_vector_sz),
          h_vec_sz(hidden_vec_sz),
          num_layers(layers_cnt),
          bi_scale(bi),
          bias_cnt(bias_mode)
    {
    }

    int input_weight_offset(int layer) const
    {
        return layer == 0 ? 0
                          : first_layer_offset() +
                                (h_vec_sz + h_vec_sz * bi_scale) * weight_stride * (layer - 1);
    }

    int hidden_weight_offset(int layer, int reverse = 0) const
    {
        return layer == 0 ? input_weight_offset(layer) + in_vec_sz * weight_stride +
                                reverse * h_vec_sz * h_vec_sz
                          : input_weight_offset(layer) + bi_scale * h_vec_sz * weight_stride +
                                reverse * h_vec_sz * h_vec_sz;
    }

    size_t bias_stride() { return bias_vector_mul_gate(); }

    int bias_off()
    {
        return first_layer_offset() +
               (h_vec_sz * bi_scale + h_vec_sz) * (num_layers - 1) * weight_stride;
    }

    int bias_off(int layer_id) { return bias_off() + layer_id * bias_cnt * weight_stride; }

    size_t bias_vector_mul_gate() const { return h_vec_sz * gates_cnt; }

    const int weight_stride;

private:
    static const int gates_cnt = 4;
    const int in_vec_sz;
    const int h_vec_sz;
    const int num_layers;
    const int bi_scale = 1;
    const int bias_cnt = 0;
};

struct RNNOffsets
{
    size_t layer_offset(int layer_id) const;

    size_t layer_stride() const;

    int gemm_write_size() const;

    size_t gemm_write_stride() const;

    size_t gemm_write_offset(int layer_id, int batch_id = 0, int reverse = 0) const;

    size_t hidden_offset(int layer_id, int batch_id = 0, int reverse = 0) const;
};

struct GRUOffsets : public RNNOffsets
{
public:
    GRUOffsets(int h_vec_size, int layers_cnt, int total_batch_size)
        : hidden_size(h_vec_size), batches_per_layer(total_batch_size), num_layers(layers_cnt)
    {
    }

    size_t layer_offset(int layer_id) const { return layer_id * layer_stride(); }

    size_t layer_stride() const { return gemm_write_stride() * batches_per_layer; }

    int gemm_write_size() const { return hidden_size; }

    size_t gemm_write_stride() const { return save_point::Count * gemm_write_size(); }

    size_t gemm_write_offset(int layer_id, int batch_num) const
    {
        return layer_offset(layer_id) + batch_num * gemm_write_stride();
    }

    size_t hidden_offset() const { return save_point::Ht * gemm_write_size(); }

private:
    const int hidden_size;

public:
    const int batches_per_layer;

    int r_offset() const { return save_point::R * gemm_write_size(); }

    int z_offset() const { return save_point::Z * gemm_write_size(); }

    int c_offset() const { return save_point::С * gemm_write_size(); }

    int activated_offset() const { return layer_stride() * num_layers; }

    size_t network_stride() { return layer_stride() * num_layers; }

private:
    int num_layers;

    enum save_point
    {
        Z     = 0,
        R     = 1,
        С     = 2,
        Ht    = 3,
        Count = 4
    };
};

struct ReluReserveBufferOffsets : public RNNOffsets
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
    ReluReserveBufferOffsets(int hidden_vec_size, int layers_cnt, int batches_per_l, int bi_scale)
        : hidden_size(hidden_vec_size),
          batches_per_layer(batches_per_l),
          save_point_size(hidden_vec_size * bi_scale),
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

    size_t gemm_write_offset(int layer_id, int batch_id, int reverse) const
    {
        return layer_offset(layer_id) + static_cast<size_t>(gemm_write_stride()) * batch_id +
               reverse * hidden_size;
    }

    size_t hidden_offset(int layer_id, int batch_id, int reverse) const
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

struct LSTMReserveBufferHelper : public RNNOffsets
{
    struct RBuffHelper
    {
        int element, save_point, batch;
        size_t layer, table;
    };

private:
    static const int gates_cnt = 4;
    auto Reserve_Buffer_strides(int save_point_sz,
                                int batches_per_l,
                                int save_points,
                                int layers_cnt,
                                int bidirect_mode = 0) const
    {
        const auto element_st = bidirect_mode ? 2 : 1;

        const auto save_point_st = element_st * save_point_sz;
        const auto batch_st      = save_point_st * save_points;
        const auto layer_st      = static_cast<size_t>(batch_st) * batches_per_l;
        const auto table_st      = layer_st * layers_cnt;

        if(bidirect_mode == 0)
            return RBuffHelper{element_st, save_point_st, batch_st, layer_st, table_st};

        MIOPEN_THROW("execution failure: bidirect is not supported by this solver");
    }

public:
    enum save_point
    {
        F     = 1,
        I     = 0,
        G     = 2,
        O     = 3,
        St    = 4,
        Ht    = 5,
        Count = 6
    };

    LSTMReserveBufferHelper(int hidden_vec_size,
                            int layers_cnt,
                            int batches_per_l,
                            int in_vec_sz,
                            int bidirect_mode = 0)
        : hidden_size(hidden_vec_size),
          batches_per_layer(batches_per_l),
          in_vec_size(in_vec_sz),
          save_point_size(bidirect_mode ? hidden_vec_size * 2 : hidden_vec_size),
          layers(layers_cnt),
          strides(Reserve_Buffer_strides(
              save_point_size, batches_per_layer, save_point::Count, layers_cnt, 0))
    {
    }

    size_t layer_offset(int layer) const { return static_cast<size_t>(layer) * strides.layer; }
    size_t layer_stride() const { return strides.layer; }

    int gemm_write_size() const { return save_point_size * gates_cnt; }
    size_t gemm_write_stride() const { return strides.batch; }

    size_t gemm_write_offset(int layer, int batch) const
    {
        return layer_offset(layer) + static_cast<size_t>(gemm_write_stride()) * batch;
    }

    size_t hidden_offset(int layer, int batch) const
    {
        return gemm_write_offset(layer, batch) + save_point::Ht * save_point_size;
    }

    const int hidden_size;
    const int batches_per_layer;
    const int in_vec_size;

    auto f_offset(int layer, int batch_num) const
    {
        return gemm_write_offset(layer, batch_num) + save_point::F * save_point_size;
    }

    auto i_offset(int layer, int batch_num) const
    {
        return gemm_write_offset(layer, batch_num) + save_point::I * save_point_size;
    }

    auto g_offset(int layer, int batch_num) const
    {
        return gemm_write_offset(layer, batch_num) + save_point::G * save_point_size;
    }

    auto o_offset(int layer, int batch_num) const
    {
        return gemm_write_offset(layer, batch_num) + save_point::O * save_point_size;
    }

    const int save_point_size; // for bidirect TODO
    const int layers;
    const RBuffHelper strides;

    auto st_offset(int layer, int batch_num)
    {
        return gemm_write_offset(layer, batch_num) + save_point::St * save_point_size;
    }

    size_t extra_save_point_offset(int layer, int batch_num) const
    {
        return strides.table // all data offset
               + static_cast<size_t>(batches_per_layer) * layer * hidden_size +
               static_cast<size_t>(batch_num * hidden_size);
    }
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
    GetTempPackedBuffersSpace(RNNDescriptor rnn_desc,
                              c_array_view<const miopenTensorDescriptor_t> desc_array)
    {
        size_t total_batch = std::accumulate(
            desc_array.data,
            desc_array.data + desc_array.size(),
            0,
            [](size_t x, miopenTensorDescriptor_t y) { return x + deref(y).GetLengths()[0]; });

        auto type_size       = GetTypeSize(desc_array[0].GetType());
        size_t in_buff_size  = type_size * total_batch * desc_array[0].GetLengths()[1];
        size_t out_buff_size = type_size * total_batch * rnn_desc.hsize *
                               (rnn_desc.dirMode == miopenRNNbidirection ? 2 : 1);
        return {in_buff_size, out_buff_size};
    }
};

} // namespace miopen

#endif // GUARD_MIOPEN_RNN_UTIL_HPP_
