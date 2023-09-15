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

struct GRUOffsets
{
public:
    GRUOffsets(int num_layers, int hidden_size, int total_batch_size)
        : num_layers(num_layers), hidden_size(hidden_size), batches_per_layer(total_batch_size)
    {
    }

    int r_offset() const { return save_point::R * hidden_size; }

    int z_offset() const { return save_point::Z * hidden_size; }

    int c_offset() const { return save_point::С * hidden_size; }

    int hidden_offset() const { return save_point::Ht * hidden_size; }

    size_t batch_offset(int layer_id, int batch_num) const
    {
        return layer_offset(layer_id) + batch_num * gemm_write_stride();
    }

    int activated_offset() const { return layer_stride() * num_layers; }

    int gemm_write_size() const { return hidden_size; }

    int gemm_write_stride() const { return save_point::Count * hidden_size; }

    int layer_offset(int layer) const { return layer * layer_stride(); }

    int batches_per_layer;

    size_t layer_stride() const { return gemm_write_stride() * batches_per_layer; }

    size_t network_stride() { return layer_stride() * num_layers; }

private:
    int num_layers;
    int hidden_size;

    enum save_point
    {
        Z     = 0,
        R     = 1,
        С     = 2,
        Ht    = 3,
        Count = 4
    };
};

struct GruWeightOffsets
{
    GruWeightOffsets(int input_vector_sz, int hidden_vec_sz, int layers_cnt, int bias_mode)
        : weight_stride(matrixes::Count * hidden_vec_sz),
          in_vec_sz(input_vector_sz),
          h_vec_sz(hidden_vec_sz),
          num_layers(layers_cnt)
    {
    }

    int weight_stride;
    const int in_vec_sz, h_vec_sz;
    const int num_layers;

    enum matrixes
    {
        Z     = 0,
        R     = 1,
        C     = 2,
        Count = 3
    };

public:
    int input_offset(int layer)
    {
        return layer == 0 ? 0
                          : (in_vec_sz + h_vec_sz) * weight_stride +
                                2 * h_vec_sz * weight_stride * (layer - 1);
    }

    int hidden_offset(int layer)
    {
        return layer == 0 ? input_offset(layer) + in_vec_sz * weight_stride
                          : input_offset(layer) + h_vec_sz * weight_stride;
    }

    int bias_stride() { return matrixes::Count * h_vec_sz; }
    int bias_off()
    {
        return (in_vec_sz + h_vec_sz + 2 * h_vec_sz * (num_layers - 1)) * weight_stride;
    }
    int bias_off(int layer_id) { return bias_off() + layer_id * weight_stride; }
};

struct ReluWeightOffsets
{
private:
    auto hidden_xinput_size(int hidden_sz, int bidirect_mode) const
    {
        if(bidirect_mode == 0)
            return hidden_sz;
        MIOPEN_THROW("execution failure: bidirect is not supported by this solver");
    }

    auto matrix_lin_layer_size(int input_vector_sz, int hidden_vec_sz) const
    {
        return (input_vector_sz + hidden_vec_sz) * hidden_vec_sz;
    }

    size_t bias_start_offset(int input_vector_sz,
                             int hidden_vec_sz,
                             int layers_cnt,
                             int bidirect_mode) const
    {
        return matrix_lin_layer_size(input_vector_sz, hidden_vec_sz) +
               static_cast<size_t>(hidden_vec_sz + hidden_xinput_size(hidden_vec_sz, 0)) *
                   hidden_vec_sz * static_cast<size_t>(layers_cnt - 1);
    }

public:
    ReluWeightOffsets(int input_vector_sz,
                      int hidden_vec_sz,
                      int layers_cnt,
                      int bias_mode,
                      int bidirectional,
                      int wei_stride = 0)
        : in_vec_sz(input_vector_sz),
          h_vec_sz(hidden_vec_sz),
          x_in_vec_sz(hidden_xinput_size(hidden_vec_sz, 0)),
          bias_cnt(bias_mode),
          matrix_normal_start_off(matrix_lin_layer_size(input_vector_sz, hidden_vec_sz)),
          bias_start_off(
              bias_start_offset(input_vector_sz, hidden_vec_sz, layers_cnt, bidirectional)),
          bidirectional(bidirectional),
          wei_stride(wei_stride)
    {
    }

private:
    const int in_vec_sz;
    const int bidirectional;
    const int x_in_vec_sz; // for bidirect TODO

    const int bias_cnt; // 0 - no bias; 1 - one bias; 2 - separate bias for x_vec and for hidden_vec

    const size_t matrix_normal_start_off;
    const size_t bias_start_off;

    auto get_input_matrix_size(int layer_id) const
    {
        return (layer_id > 0 ? x_in_vec_sz : in_vec_sz) * h_vec_sz;
    }

    auto get_hidden_matrix_size() const { return h_vec_sz * h_vec_sz; }

    auto get_matrix_layer_size(int layer_id) const
    {
        return get_input_matrix_size(layer_id) + get_hidden_matrix_size();
    }

    int bias_vector_size() const { return h_vec_sz; }

    size_t bias_relative_off(int layer_id, int bias_id) const
    {
        return static_cast<size_t>(layer_id * bias_cnt + bias_id) * h_vec_sz;
    }

public:
    const int h_vec_sz;
    const int wei_stride;

    size_t input_weight_offset(int layer_id) const
    {
        return hidden_weight_offset(layer_id, 0) + h_vec_sz * wei_stride;
    }

    size_t hidden_weight_offset(int layer_id, int reverse) const
    {
        return in_vec_sz * wei_stride +
               layer_id * (bidirectional * h_vec_sz + h_vec_sz) * wei_stride +
               reverse * h_vec_sz * h_vec_sz;
    }

    size_t input_offset(int layer_id) const
    {
        if(layer_id > 0)
            return matrix_normal_start_off +
                   static_cast<size_t>(layer_id - 1) * get_matrix_layer_size(layer_id);
        else
            return 0;
    };

    size_t hidden_offset(int layer_id) const
    {
        if(layer_id > 0)
            return input_offset(layer_id) + static_cast<size_t>(h_vec_sz * x_in_vec_sz);
        else
            return input_offset(layer_id) + static_cast<size_t>(h_vec_sz * in_vec_sz);
    };

    int bias_stride() const { return bias_vector_size(); }

    size_t bias_off(int layer_id, int bias_id) const
    {
        return bias_start_off + bias_relative_off(layer_id, bias_id);
    }
};

struct ReluReserveBufferOffsets
{
    struct RBuffHelper
    {
        int element, save_point, batch;
        size_t layer, table;
    };

private:
    auto Reserve_Buffer_strides(int save_point_sz,
                                int batches_per_layer,
                                int layers,
                                int bidirect_mode = 0) const
    {
        const auto element_st    = bidirect_mode ? 2 : 1;
        const auto save_point_st = element_st * save_point_sz;
        const auto batch_st      = save_point_st;
        const auto layer_st      = static_cast<size_t>(batch_st) * batches_per_layer;
        const auto table_st      = layers * layer_st;

        return RBuffHelper{element_st, save_point_st, batch_st, layer_st, table_st};
    }

public:
    ReluReserveBufferOffsets(int hidden_vec_sz,
                             int save_point_sz,
                             int layers_cnt,
                             int batches_per_layer,
                             int max_batch,
                             bool bidirect_mode = 0)
        : h_vec_size(hidden_vec_sz),
          save_point_size(save_point_sz),
          layers(layers_cnt),
          batches_per_layer(batches_per_layer),
          max_batch(max_batch),
          strides(
              Reserve_Buffer_strides(save_point_sz, batches_per_layer, layers_cnt, bidirect_mode))
    {
    }

    const int h_vec_size;
    const int save_point_size;

    const int layers;
    const int batches_per_layer;
    const RBuffHelper strides;
    const int max_batch;

    size_t layer_offset(int layer_id) const
    {
        return static_cast<size_t>(layer_id) * strides.layer;
    }

    auto layer_stride() const { return strides.layer; }

    auto gemm_write_size() const { return strides.save_point; }

    auto gemm_write_stride() const { return strides.batch; }

    size_t gemm_write_relative_offset(int batch_id) const
    {
        return static_cast<size_t>(gemm_write_stride()) * batch_id;
    }

    size_t gemm_write_offset(int layer_id, int batch_id, int reverse = 0) const
    {
        return layer_offset(layer_id) + static_cast<size_t>(gemm_write_stride()) * batch_id +
               reverse * h_vec_size;
    }

    size_t ht_offset(int layer_id, int batch_id, int reverse = 0) const
    {
        return strides.table + layer_offset(layer_id) + gemm_write_relative_offset(batch_id) +
               reverse * h_vec_size;
    }

    size_t ht_offset(int layer_id) const { return strides.table + layer_offset(layer_id); }
};

struct LSTMReserveBufferHelper
{
    struct RBuffHelper
    {
        int element, save_point, batch;
        size_t layer;
    };

private:
    auto Reserve_Buffer_strides(int save_point_sz,
                                int batches_per_layer,
                                int save_points,
                                int bidirect_mode = 0) const
    {
        const auto element_st    = 1;
        const auto save_point_st = element_st * save_point_sz;
        const auto batch_st      = save_point_st * save_points;
        const auto layer_st      = static_cast<size_t>(batch_st) * batches_per_layer;
        if(bidirect_mode == 0)
            return RBuffHelper{element_st, save_point_st, batch_st, layer_st};
        MIOPEN_THROW("execution failure: bidirect is not supported by this solver");
    }

public:
    enum save_point
    {
        F  = 1,
        I  = 0,
        G  = 2,
        O  = 3,
        St = 4,
        Ht = 5
    };

    LSTMReserveBufferHelper(int hidden_vec_sz,
                            int save_point_sz,
                            int layers_cnt,
                            int batches_per_layer,
                            int save_points,
                            int gates_cnt)
        : h_vec(hidden_vec_sz),
          save_point_size(save_point_sz),
          layers(layers_cnt),
          batches(batches_per_layer),
          save_points_cnt(save_points),
          gates(gates_cnt),
          strides(Reserve_Buffer_strides(save_point_sz, batches, save_points, 0))
    {
    }

    const int h_vec;
    const int save_point_size; // for bidirect TODO

    const int layers;
    const int batches;
    const int save_points_cnt;
    const int gates;
    const RBuffHelper strides;

    size_t layer_offset(int layer) const { return static_cast<size_t>(layer) * strides.layer; }
    auto layer_stride() const { return strides.layer; }

    auto gemm_write_size() const { return h_vec * gates; }
    auto gemm_write_stride() const { return strides.batch; } // save_point_size * save_points_cnt

    size_t gemm_write_relative_offset(int batch_id) const
    {
        return static_cast<size_t>(gemm_write_stride()) * batch_id;
    }

    size_t gemm_write_offset(int layer, int batch_id, int reverse = 0) const
    {
        return layer_offset(layer) + static_cast<size_t>(gemm_write_stride()) * batch_id;
    }

    auto ht_relative_offset() const { return save_point::Ht * save_point_size; }

    auto ct_relative_offset() const { return save_point::St * save_point_size; }

    auto get_gate_relative_offset(int gate_id) const { return gate_id * save_point_size; }

    size_t ht_offset(int layer_id, int batch_id) const
    {
        return layer_offset(layer_id) + gemm_write_relative_offset(batch_id) + ht_relative_offset();
    }

    size_t extra_save_point_offset(int layer_id, int batch_id) const
    {
        return (static_cast<size_t>(batches) * layers * gemm_write_stride()) // all data offset
               + (static_cast<size_t>(batches) * layer_id) * h_vec +
               static_cast<size_t>(batch_id * h_vec);
    }
};

struct LSTMWeightsBufferHelper
{
private:
    auto hidden_xinput_size(int hidden_sz, int bidirect_mode) const
    {
        if(bidirect_mode == 0)
            return hidden_sz;
        MIOPEN_THROW("execution failure: bidirect is not supported by this solver");
    }

    auto matrix_lin_layer_size(int input_vector_sz, int hidden_vec_sz, int gates) const
    {
        return (input_vector_sz + hidden_vec_sz) * hidden_vec_sz * gates;
    }
    size_t bias_start_offset(
        int input_vector_sz, int hidden_vec_sz, int layers_cnt, int gates, int bidirect_mode) const
    {
        if(bidirect_mode == 0)
            return matrix_lin_layer_size(input_vector_sz, hidden_vec_sz, gates) +
                   static_cast<size_t>(hidden_vec_sz + hidden_xinput_size(hidden_vec_sz, 0)) *
                       hidden_vec_sz * static_cast<size_t>(layers_cnt - 1) * gates;

        MIOPEN_THROW("execution failure: bidirect is not supported by this solver");
    }

public:
    LSTMWeightsBufferHelper(
        int input_vector_sz, int hidden_vec_sz, int layers_cnt, int bias_mode, int gates)
        : in_vec(input_vector_sz),
          h_vec(hidden_vec_sz),
          x_in_vec(hidden_xinput_size(hidden_vec_sz, 0)),
          layers(layers_cnt),
          gates_cnt(gates),
          bias_cnt(bias_mode),
          matrix_normal_start_off(matrix_lin_layer_size(input_vector_sz, hidden_vec_sz, gates)),
          bias_start_off(bias_start_offset(input_vector_sz, hidden_vec_sz, layers_cnt, gates, 0))
    {
    }

    const int in_vec, h_vec;
    const int x_in_vec; // for bidirect TODO

    const int layers;
    const int gates_cnt;
    const int bias_cnt; // 0 - no bisa; 1 - one bias; 2 - separate bias for x_vec and for hidden_vec
private:
    const size_t matrix_normal_start_off;
    const size_t bias_start_off;

public:
    auto get_matrix_x_size(int layer_id) const
    {
        return (layer_id > 0 ? x_in_vec : in_vec) * h_vec;
    }
    auto get_matrix_h_size() const { return h_vec * h_vec; }
    auto get_matrix_layer_size(int layer_id) const
    {
        return get_matrix_x_size(layer_id) * gates_cnt + get_matrix_h_size() * gates_cnt;
    }

    size_t get_matrix_x_off(int layer_id) const
    {
        if(layer_id > 0)
            return matrix_normal_start_off +
                   static_cast<size_t>(layer_id - 1) * get_matrix_layer_size(layer_id);
        else
            return 0;
    };

    size_t get_matrix_h_off(int layer_id) const
    {
        if(layer_id > 0)
            return get_matrix_x_off(layer_id) + static_cast<size_t>(h_vec * x_in_vec * gates_cnt);
        else
            return get_matrix_x_off(layer_id) + static_cast<size_t>(h_vec * in_vec) * gates_cnt;
    };

    int bias_vector_size() const { return h_vec; }
    int bias_vector_mul_gate() const { return bias_vector_size() * gates_cnt; }
    int bias_stride() const { return bias_vector_mul_gate(); }

    size_t bias_relative_off(int layer_id, int bias_id) const
    {
        return static_cast<size_t>(layer_id * bias_cnt + bias_id) * gates_cnt * h_vec;
    }

    size_t get_bias_off(int layer_id, int bias_id) const
    {
        return bias_start_off + bias_relative_off(layer_id, bias_id);
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
