/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/rnn.hpp>
#include <miopen/rnn_util.hpp>

#include <miopen/activ.hpp>
#include <miopen/env.hpp>

#include <vector>
#include <array>
#include <numeric>
#include <algorithm>

#include <cassert>
#include <miopen/rnn/base_ops.hpp>

namespace miopen {

// https://github.com/ROCm/MIOpen/issues/3048
namespace WA_RHEL {

template <class InputIt, class OutputIt, class T, class BinaryOp>
OutputIt exclusive_scan_wa(InputIt first, InputIt last, OutputIt d_first, T init, BinaryOp op)
{
    if(first == last)
        return d_first;

    auto acc = init;

    for(; first != last; ++first)
    {
        *d_first++ = acc;
        acc        = op(std::move(acc), *first); // std::move since C++11
    }

    return d_first;
}

} // namespace WA_RHEL

namespace rnn_base {

enum class SequenceDirection
{
    Forward = 0,
    Reverse = 1
};

enum class LstmGateAndState
{
    I  = 0,
    F  = 1,
    O  = 2,
    G  = 3,
    St = 4,
    Ht = 5
};

/**
 * \section Extensions for layouts.
 */

// GRU
// TODO

// LSTM
template <typename Derived>
class GeneralLstmWsExt
{
public:
    size_t getGasOffset(const size_t layer_id,
                        const size_t vector_id,
                        const SequenceDirection direction,
                        LstmGateAndState gas) const
    {
        return static_cast<const Derived*>(this)->getGasOffsetImpl(
            layer_id, vector_id, direction, gas);
    }

    const std::array<size_t, 4>& getGateAndStateStride(LstmGateAndState gas) const
    {
        return static_cast<const Derived*>(this)->getGateAndStateStrideImpl(gas);
    }
};

template <typename Derived>
class LstmWsGateBlockExt
{
public:
    size_t getGateBlockOffset(const size_t layer_id,
                              const size_t vector_id,
                              const SequenceDirection direction) const
    {
        return static_cast<const Derived*>(this)->getGasOffset(
            layer_id, vector_id, direction, LstmGateAndState::I);
    }

    const std::array<size_t, 4>& getGateBlockStride() const
    {
        return static_cast<const Derived*>(this)->getGateAndStateStride(LstmGateAndState::I);
    }

    const std::array<size_t, 4>& getGateBlockSize() const
    {
        return static_cast<const Derived*>(this)->getGateBlockSizeImpl();
    }
};

template <typename Derived>
class LstmActiveCellExt
{
public:
    size_t getActiveCellOffset(const size_t layer_id,
                               const size_t vector_id,
                               const SequenceDirection direction) const
    {
        return static_cast<const Derived*>(this)->getActiveCellOffsetImpl(
            layer_id, vector_id, direction);
    }

    const std::array<size_t, 4>& getActiveCellStride() const
    {
        return static_cast<const Derived*>(this)->getActiveCellStrideImpl();
    }
};

// pure RNN
template <typename Derived>
class BaseRnnWsBufferPacked
{
public:
    BaseRnnWsBufferPacked() = default;
    size_t getHiddenStateOffset(const size_t layer_id,
                                const size_t vector_id,
                                const SequenceDirection direction) const
    {
        return static_cast<const Derived*>(this)->getHiddenStateOffsetImpl(
            layer_id, vector_id, direction);
    }

    // layer, minor dim(seq or sample), directions, element
    const std::array<size_t, 4>& getHiddenStateStride() const
    {
        return static_cast<const Derived*>(this)->getHiddenStateStrideImpl();
    }

    size_t getBufferSize() const { return static_cast<const Derived*>(this)->getBufferSizeImpl(); }
};

/**
 * \section Standard layouts for temporary buffers
 */
//////

class GeneralRNNTempBuffer : public BaseRnnWsBufferPacked<GeneralRNNTempBuffer>
{
protected:
    GeneralRNNTempBuffer(const std::array<size_t, 4>& hstate_strides,
                         const std::array<size_t, 4>& hstate_sizes,
                         size_t total_element_cnt)
        : hStateStrides{hstate_strides},
          hStateSizes{hstate_sizes},
          totalElementCnt{total_element_cnt}
    {
    }

public:
    static GeneralRNNTempBuffer
    build(size_t layers_cnt, size_t vectors_per_layer, size_t directions, size_t hidden_vec_sz)
    {
        const std::array<size_t, 4> h_state_sizes{
            layers_cnt, vectors_per_layer, directions, hidden_vec_sz};
        std::array<size_t, 4> h_state_strides = {0, 0, 0, 1};
        std::partial_sum(h_state_sizes.crbegin(),
                         std::prev(h_state_sizes.crend()),
                         std::next(h_state_strides.rbegin()),
                         std::multiplies<size_t>{});

        auto total_element_cnt = h_state_strides[0] * h_state_sizes[0];
        return GeneralRNNTempBuffer{h_state_strides, h_state_sizes, total_element_cnt};
    }

    size_t getHiddenStateOffsetImpl(const size_t layer_id,
                                    const size_t vector_id,
                                    const SequenceDirection direction) const
    {
        const std::array<size_t, 3> pos{layer_id, vector_id, static_cast<size_t>(direction)};

        return std::inner_product(
            pos.cbegin(), pos.cend(), hStateStrides.cbegin(), static_cast<size_t>(0));
    }

    size_t getBufferSizeImpl() const { return totalElementCnt; }

    const std::array<size_t, 4>& getHiddenStateStrideImpl() const { return hStateStrides; }

    // layer, comb dim(seq, sample), direction, vector element
    const std::array<size_t, 4> hStateStrides;
    // layers, comb dims(seq, sample), directions, elements
    const std::array<size_t, 4> hStateSizes;
    const size_t totalElementCnt;
};

/*
 *struct Workspace_LSTM_BWD{ //packed
 *  struct layer{
 *    struct all_sequences_all_samples{
 *        struct mat_mul_block{
 *            struct gate_vector{
 *                float[hidden_size]; HidUpdate-> write
 *            } I, F, O, G;
 *        } gemm_block[directions];
 *
 *        struct dcx_vec{
 *            float[hidden_size];  HidUpdate-> write
 *        } dy[directions];
 *
 *        struct dy_vec{
 *            float[hidden_size]; HidUpdate <-read
 *        } dy[directions];
 *
 *    } combi_dim[seq_len * batch_size=totalBatch]; [seq][batch]
 *  } layers[nLayer];
 *}
 */

class GeneralLstmTempBuffer : public GeneralRNNTempBuffer,
                              public GeneralLstmWsExt<GeneralLstmTempBuffer>,
                              public LstmWsGateBlockExt<GeneralLstmTempBuffer>
{
protected:
    GeneralLstmTempBuffer(const std::array<size_t, 4>& h_state_strides,
                          const std::array<size_t, 4>& h_state_sizes,
                          const std::array<size_t, 4>& lstm_gate_sizes,
                          const std::array<size_t, 4>& lstm_gate_strides,
                          const std::array<size_t, 4>& lstm_gates_block_sizes,
                          size_t total_element_cnt)
        : GeneralRNNTempBuffer{h_state_strides, h_state_sizes, total_element_cnt},
          gateSizes{lstm_gate_sizes},
          gateStride{lstm_gate_strides},
          gateBlockSizes{lstm_gates_block_sizes}
    {
    }

public:
    static GeneralLstmTempBuffer
    build(size_t layers_cnt, size_t comp_dim_per_layer, size_t directions, size_t hidden_vec_sz)
    {

        constexpr size_t MS_len  = 2;
        constexpr size_t LS_size = 2;
        // Most significant sizes
        //{layer, comp_dim_per_layer}
        std::array<size_t, MS_len> block_MS_size = {layers_cnt, comp_dim_per_layer};

        //{layer, comp_dim_per_layer, directions,vec}
        const auto h_state_sizes =
            Concat(block_MS_size, std::array<size_t, 2>{directions, hidden_vec_sz});

        const auto lstm_gate_sizes =
            Concat(block_MS_size, std::array<size_t, 2>{directions, hidden_vec_sz});

        const auto lstm_gates_block_sizes =
            Concat(block_MS_size, std::array<size_t, 2>{directions, 4 * hidden_vec_sz});

        // Less significant stride
        // LSStride{directions,vec}
        std::array<size_t, LS_size> h_state_LS_strides{};
        std::array<size_t, LS_size> lstm_gate_LS_strides{};

        WA_RHEL::exclusive_scan_wa(h_state_sizes.crbegin(),
                                   std::next(h_state_sizes.crbegin(), h_state_LS_strides.size()),
                                   h_state_LS_strides.rbegin(),
                                   1LL,
                                   std::multiplies<size_t>{});

        WA_RHEL::exclusive_scan_wa(
            lstm_gates_block_sizes.crbegin(),
            std::next(lstm_gates_block_sizes.crbegin(), lstm_gate_LS_strides.size()),
            lstm_gate_LS_strides.rbegin(),
            1LL,
            std::multiplies<size_t>{});

        const auto gas_block_stride = [&lstm_gates_block_sizes,
                                       &h_state_sizes,
                                       &lstm_gate_LS_strides,
                                       &h_state_LS_strides](const auto last_dim) {
            auto gate_size_in_block    = lstm_gates_block_sizes[last_dim] * lstm_gate_LS_strides[0];
            auto h_state_size_in_block = h_state_sizes[last_dim] * h_state_LS_strides[0];
            auto c_state_size_in_block = h_state_size_in_block;
            return gate_size_in_block + c_state_size_in_block + h_state_size_in_block;
        }(MS_len);

        // Most significant stride
        // MSStride{layer, comp_dim_per_layer}
        std::array<size_t, MS_len> block_MS_strides{};

        WA_RHEL::exclusive_scan_wa(block_MS_size.crbegin(),
                                   std::next(block_MS_size.crbegin(), block_MS_strides.size()),
                                   block_MS_strides.rbegin(),
                                   gas_block_stride,
                                   std::multiplies<size_t>{});

        const std::array<size_t, 4> h_state_strides = Concat(block_MS_strides, h_state_LS_strides);
        const std::array<size_t, 4> lstm_gate_strides =
            Concat(block_MS_strides, lstm_gate_LS_strides);

        auto total_element_cnt = h_state_strides[0] * h_state_sizes[0];

        return {h_state_strides,
                h_state_sizes,
                lstm_gate_sizes,
                lstm_gate_strides,
                lstm_gates_block_sizes,
                total_element_cnt};
    }

    size_t getHiddenStateOffset(const size_t layer_id,
                                const size_t vector_id,
                                const SequenceDirection direction) const
    {
        return getGasOffset(layer_id, vector_id, direction, LstmGateAndState::Ht);
    }

    size_t getGasOffsetImpl(const size_t layer_id,
                            const size_t vector_id,
                            const SequenceDirection direction,
                            LstmGateAndState gas) const
    {
        auto start_ident = getGasIndent(gas);

        if(gas == LstmGateAndState::Ht || gas == LstmGateAndState::St)
            return start_ident +
                   GeneralRNNTempBuffer::getHiddenStateOffset(layer_id, vector_id, direction);

        const std::array<size_t, 3> pos{layer_id, vector_id, static_cast<size_t>(direction)};

        return start_ident +
               std::inner_product(
                   pos.cbegin(), pos.cend(), hStateStrides.cbegin(), static_cast<size_t>(0));
    }

    // layer, minor dim(seq or sample), directions, element
    const std::array<size_t, 4>& getGateAndStateStrideImpl(LstmGateAndState gas) const
    {
        if(gas == LstmGateAndState::Ht || gas == LstmGateAndState::St)
            return getHiddenStateStride();
        return gateStride;
    }

    // layer, minor dim(seq or sample), directions, element
    const std::array<size_t, 4>& getGateBlockSizeImpl() const { return gateBlockSizes; }

    const std::array<size_t, 4> gateSizes;
    const std::array<size_t, 4> gateStride;
    const std::array<size_t, 4> gateBlockSizes;

private:
    size_t getGasIndent(LstmGateAndState gas) const
    {
        switch(gas)
        {
        case LstmGateAndState::I:
        case LstmGateAndState::G:
        case LstmGateAndState::F:
        case LstmGateAndState::O: return static_cast<size_t>(gas) * gateStride[3] * gateSizes[3];
        case LstmGateAndState::St: return gateStride[2] * gateSizes[2]; // direction DIM
        case LstmGateAndState::Ht:
            return (gateStride[2] + getHiddenStateStride()[2]) * gateSizes[2];
        }
        return 0;
    }
};

/*
 *struct ReserveSpace_LSTM{ //packed
 *  struct layer{
 *    struct all_sequences_all_samples{
 *
 *        struct mat_mul_block{
 *            struct gate_vector{
 *                float[hidden_size]; HidUpdate-> write
 *            } I, F, O, G;
 *        } gemm_block[directions];
 *
 *        struct dcx_vec{
 *            float[hidden_size];  HidUpdate-> write
 *        } dy[directions];
 *
 *        struct dy_vec{
 *            float[hidden_size]; HidUpdate <-read
 *        } dy[directions];
 *
 *    } combi_dim[seq_len * batch_size=totalBatch]; [seq][batch]
 *  } layers[nLayer];
 *
 *  struct extra_activation_vec{
 *      float[hidden_size];
 *  } active_cell[nLayer*bidirect*totalBatch];
 *}
 */

class GeneralLstmRedBuffer : public GeneralLstmTempBuffer,
                             public LstmActiveCellExt<GeneralLstmRedBuffer>
{
protected:
    GeneralLstmRedBuffer(const GeneralLstmTempBuffer& base,
                         const std::array<size_t, 4>& active_cells_sizes,
                         const std::array<size_t, 4>& active_cells_strides,
                         size_t active_cells_ident,
                         size_t active_cell_elements)
        : GeneralLstmTempBuffer{base},
          activeCellSize{active_cells_sizes},
          activeCellStride{active_cells_strides},
          activeCellsIdent{active_cells_ident},
          activeCellElements{active_cell_elements}
    {
    }

public:
    static GeneralLstmRedBuffer
    build(size_t layers_cnt, size_t comp_dim_per_layer, size_t directions, size_t hidden_vec_sz)
    {

        auto base =
            GeneralLstmTempBuffer::build(layers_cnt, comp_dim_per_layer, directions, hidden_vec_sz);

        auto active_cells_ident = base.gateStride[0] * base.gateSizes[0];

        std::array<size_t, 4> active_cells_sizes{
            layers_cnt, comp_dim_per_layer, directions, hidden_vec_sz};

        std::array<size_t, 4> active_cells_strides = {};

        WA_RHEL::exclusive_scan_wa(active_cells_sizes.crbegin(),
                                   active_cells_sizes.crend(),
                                   active_cells_strides.rbegin(),
                                   1LL,
                                   std::multiplies<size_t>{});

        auto active_cell_elements = active_cells_strides[0] * active_cells_sizes[0];

        return {base,
                active_cells_sizes,
                active_cells_strides,
                active_cells_ident,
                active_cell_elements};
    }

    size_t getBufferSizeImpl() const { return activeCellsIdent + activeCellElements; }

    size_t getActiveCellOffsetImpl(const size_t layer_id,
                                   const size_t vector_id,
                                   const SequenceDirection direction) const
    {
        const std::array<size_t, 3> pos{layer_id, vector_id, static_cast<size_t>(direction)};

        return activeCellsIdent +
               std::inner_product(
                   pos.cbegin(), pos.cend(), activeCellStride.cbegin(), static_cast<size_t>(0));
    }

    const std::array<size_t, 4>& getActiveCellStrideImpl() const { return activeCellStride; }

    const std::array<size_t, 4> activeCellSize;
    const std::array<size_t, 4> activeCellStride;
    const size_t activeCellsIdent;
    const size_t activeCellElements;
};

/**
 * \section Сlasses for easier navigation.
 */

class BatchController
{
public:
    template <class VectorT>
    static BatchController Create(VectorT&& batchs_at_time)
    {
        std::vector<size_t> batch{std::forward<VectorT>(batchs_at_time)};
        std::vector<size_t> b_prefix_sums(batch.size());
        b_prefix_sums[0] = 0;

        std::partial_sum(batch.cbegin(), std::prev(batch.cend()), std::next(b_prefix_sums.begin()));

        return BatchController{std::move(batch), std::move(b_prefix_sums)};
    }

    static BatchController Create(const SeqTensorDescriptor& desc)
    {
        return Create(desc.GetBatchesPerSequence());
    }

    size_t getTotalBatchSum() const
    {
        return *batchAtTime.crbegin() + *batchPrefSumAtTime.crbegin();
    }

    template <typename TimeIndexT>
    size_t getBatchSize(TimeIndexT time_id) const
    {
        return batchAtTime[time_id];
    }

    template <typename TimeIndexT>
    size_t getBatchSum(TimeIndexT time_id) const
    {
        return batchPrefSumAtTime[time_id];
    }

private:
    template <class T, std::enable_if_t<std::is_same<T, std::vector<size_t>>::value, bool> = true>
    explicit BatchController(T&& batch_at_time, T&& batch_prefix_sums)
        : batchAtTime(batch_at_time), batchPrefSumAtTime{std::forward<T>(batch_prefix_sums)}
    {
    }

    const std::vector<size_t> batchAtTime;
    const std::vector<size_t> batchPrefSumAtTime;
};

class SequenceIterator
{
public:
    SequenceIterator(size_t logical_value, SequenceDirection dir, size_t seq_size, bool is_fwd_pass)
        : value(logical_value),
          startVal(is_fwd_pass ? 0 : seq_size - 1),
          endVal(is_fwd_pass ? seq_size - 1 : 0),
          maxVal(seq_size - 1),
          isReverseSeq(dir == SequenceDirection::Reverse),
          isFwdPass(is_fwd_pass)
    {
        assert(logical_value < seq_size);
    }

    size_t getLogVal() const { return value; }
    size_t getPhisVal() const { return !isReverseSeq ? value : maxVal - value; }

    SequenceIterator getNext() const
    {
        assert(!isLast());
        return PartClone(value + (isFwdPass ? 1 : -1));
    }
    SequenceIterator getPrev() const
    {
        assert(!isFirst());
        return PartClone(value + (isFwdPass ? -1 : 1));
    }

    bool isFirst() const { return value == startVal; }

    bool isLast() const { return value == endVal; }

private:
    SequenceIterator PartClone(const size_t newValue) const
    {
        return {newValue, startVal, endVal, maxVal, isReverseSeq, isFwdPass};
    }

    SequenceIterator(size_t value_,
                     size_t start_val,
                     size_t end_val,
                     size_t max_val,
                     bool is_reverse_seq,
                     bool is_fwd_pass)
        : value(value_),
          startVal(start_val),
          endVal(end_val),
          maxVal(max_val),
          isReverseSeq(is_reverse_seq),
          isFwdPass(is_fwd_pass)
    {
    }

    const size_t value;
    const size_t startVal;
    const size_t endVal;
    const size_t maxVal;
    const bool isReverseSeq;
    const bool isFwdPass;
};

/**
 * \section User defined buffers.
 */

/*
 *struct Weights{
 *  struct filter{
 *
 *    struct gateInputMatrix{
 *      float[hidden_size][input_size]
 *    } gateFilters[gate_cnt];
 *
 *    struct gateHiddenMatrix{
 *      float[hidden_size][hidden_size]
 *    } gateFiltersHiddden[gate_cnt];
 *  } filters[n_layers][bidirect];
 *
 *  struct bias{
 *      struct {
 *          float[hidden_size]
 *      }gateBias[gate_cnt]
 *  }biases[n_layers][bias_cnt][bidirect]
 *}
 */
struct WeightsBufferDescriptor
{
private:
    static size_t hidden_xinput_size(size_t hidden_sz, bool bidirect_mode)
    {
        if(!bidirect_mode)
            return hidden_sz;
        MIOPEN_THROW("execution failure: bidirect is not supported by this solver");
    }

    static auto filter_size(size_t input_vector_sz, size_t hidden_vec_sz, size_t gates)
    {
        return (input_vector_sz + hidden_vec_sz) * hidden_vec_sz * gates;
    }

    static size_t bias_start_offset(size_t phis_layer_filter_sz,
                                    size_t hidden_layer_filter_sz,
                                    size_t layers_cnt,
                                    bool bidirect_mode)
    {
        if(!bidirect_mode)
        {
            return phis_layer_filter_sz + hidden_layer_filter_sz * (layers_cnt - 1);
        }

        MIOPEN_THROW("execution failure: bidirect is not supported by this solver");
    }

    WeightsBufferDescriptor(size_t input_vector_sz,
                            size_t hidden_vec_sz,
                            size_t hidden_xinput_sz,
                            size_t layers_cnt,
                            size_t gates,
                            size_t bias_mode,

                            size_t lin_filter_sz,
                            size_t hidden_filter_sz,
                            const std::array<size_t, 4>& bias_strides,
                            const std::array<size_t, 4>& bias_lens,
                            size_t bias_off)
        : inVec(input_vector_sz),
          hVec(hidden_vec_sz),
          xInVec(hidden_xinput_sz),
          layers(layers_cnt),
          gatesCnt(gates),
          biasCnt(bias_mode),
          biasStrides{bias_strides},
          biasLens{bias_lens},
          biasStartOff(bias_off),
          linLayerFilterSize(lin_filter_sz),
          hiddenLayerFilterSize(hidden_filter_sz)
    {
    }

public:
    static WeightsBufferDescriptor create(size_t input_vector_sz,
                                          size_t hidden_vec_sz,
                                          size_t layers_cnt,
                                          size_t bias_mode,
                                          size_t input_mode,
                                          size_t gates,
                                          bool bidirect_mode)
    {
        const size_t directions_num = bidirect_mode ? 2 : 1;
        const size_t x_in_vec       = hidden_xinput_size(hidden_vec_sz, bidirect_mode);

        size_t input_vector_filter_sz = input_mode == 0 ? input_vector_sz : 0;

        const size_t linLayerFilterSize = filter_size(input_vector_filter_sz, hidden_vec_sz, gates);
        const size_t hiddenLayerFilterSize = filter_size(x_in_vec, hidden_vec_sz, gates);

        const size_t bias_start_off =
            bias_start_offset(linLayerFilterSize, hiddenLayerFilterSize, layers_cnt, bidirect_mode);

        //[layer][dir][param_id][vector]
        const std::array<size_t, 4> bias_lens{layers_cnt, directions_num, gates, hidden_vec_sz};
        const std::array<size_t, 4> bias_strides = [](const auto& lens, size_t bias_cnt) {
            std::array<size_t, 4> strides = {0, 0, 0, 1};
            std::partial_sum(lens.crbegin(),
                             std::prev(lens.crend()),
                             std::next(strides.rbegin()),
                             std::multiplies<size_t>{});
            strides[0] *= bias_cnt;
            return strides;
        }(bias_lens, bias_mode * 2);

        return {input_vector_filter_sz,
                hidden_vec_sz,
                x_in_vec,
                layers_cnt,
                gates,
                bias_mode,
                linLayerFilterSize,
                hiddenLayerFilterSize,
                bias_strides,
                bias_lens,
                bias_start_off};
    }

    const size_t inVec, hVec;
    const size_t xInVec; // for bidirect TODO

    const size_t layers;
    const size_t gatesCnt;
    const size_t
        biasCnt; // 0 - no bisa; 1 - one bias; 2 - separate bias for x_vec and for hidden_vec
private:
    //[layer][dir][param_id][vector]
    const std::array<size_t, 4> biasStrides;
    const std::array<size_t, 4> biasLens;

    const size_t biasStartOff;

    const size_t linLayerFilterSize;
    const size_t hiddenLayerFilterSize;

public:
    size_t getParamRelativeOff([[maybe_unused]] size_t layer_id, int /*dir_id*/, int param_id) const
    {
        assert(layer_id > 0);
        return param_id * hVec * xInVec;
    }

    size_t getPhisParamRelativeOff(int /*dir_id*/, int param_id) const
    {
        return hVec * ((static_cast<size_t>(param_id) >= gatesCnt)
                           ? inVec * gatesCnt + xInVec * (param_id - gatesCnt)
                           : inVec * param_id);
    }

    size_t getMatrixOff(size_t layer_id, int dir_id, int param_id) const
    {
        size_t offset = 0;
        if(layer_id > 0)
        {
            offset += linLayerFilterSize;
        }
        else
        {
            return getPhisParamRelativeOff(dir_id, param_id);
        }
        if(layer_id > 1)
        {
            offset += hiddenLayerFilterSize * (layer_id - 1);
        }
        return offset + getParamRelativeOff(layer_id, dir_id, param_id);
    }

    size_t getMatrixXinOff(size_t layer_id, int dir_id) const
    {
        return getMatrixOff(layer_id, dir_id, 0);
    }

    size_t getMatrixHidOff(size_t layer_id, int dir_id) const
    {
        return getMatrixOff(layer_id, dir_id, gatesCnt);
    }

    size_t getBiasOff(size_t layer_id, int dir_id, int param_id) const
    {
        return biasStartOff + biasStrides[0] * layer_id + biasStrides[1] * dir_id +
               biasStrides[2] * param_id;
    }

    size_t getBiasXinOff(size_t layer_id, int dir_id, int param_id) const
    {
        assert(param_id < gatesCnt);
        return getBiasOff(layer_id, dir_id, param_id);
    }

    size_t getBiasHidOff(size_t layer_id, int dir_id, int param_id) const
    {
        assert(param_id < gatesCnt);
        return getBiasOff(layer_id, dir_id, param_id + gatesCnt);
    }

    //[layers][dirs][params][vec_size]
    const std::array<size_t, 4>& getBiasSize() const { return biasLens; }

    //[layers][dirs][params][vec_size]
    const std::array<size_t, 4>& getBiasStride() const { return biasStrides; }
};

class HiddenBuffersDescriptor
{
public:
    explicit HiddenBuffersDescriptor(const TensorDescriptor& hx_desc)
        : lens(hx_desc.GetLengths().begin(), hx_desc.GetLengths().begin() + 3),
          strides(hx_desc.GetStrides().begin(), hx_desc.GetStrides().begin() + 3)
    {
    }

    inline size_t getOffset(const size_t virtual_layer_id) const
    {
        return getOffset(virtual_layer_id, 0);
    }

    inline size_t getOffset(const size_t virtual_layer_id, const size_t batch_id) const
    {
        return strides[0] * virtual_layer_id + strides[1] * batch_id;
    }

    inline const std::vector<size_t>& getStrides() const { return strides; }

    inline const std::vector<size_t>& GetLengths() const { return lens; }

    inline size_t getVirtualLayersCnt() const { return lens[0]; }
    inline size_t getMiniBatchSize() const { return lens[1]; }
    inline size_t getHiddenSize() const { return lens[2]; }

    static std::vector<size_t> MakeStrides(const std::vector<size_t>& lengths)
    {
        return {lengths[1] * lengths[2], lengths[2], 1};
    }

private:
    // local caching

    const std::vector<size_t> lens;
    const std::vector<size_t> strides;
};

class IOBufferDescriptor
{
    IOBufferDescriptor() = default;
    IOBufferDescriptor(std::vector<size_t>&& buffer_lens,
                       std::vector<size_t>&& buffer_strides,
                       std::vector<size_t>&& packed_lens,
                       std::vector<size_t>&& packed_strides,
                       std::vector<size_t>&& seq_lens_per_sample)
        : lens{std::move(buffer_lens)},
          strides{std::move(buffer_strides)},
          packedLens{std::move(packed_lens)},
          packedStrides{std::move(packed_strides)},
          seqLensPerSample{std::move(seq_lens_per_sample)}
    {
    }

public:
    static IOBufferDescriptor build(const SeqTensorDescriptor& xyDesc)
    {
        //{batch, seq_cnt, vector}
        auto lens    = xyDesc.GetLengths();
        auto strides = xyDesc.GetPaddedStrides();

        //{ combine(batch, seq_cnt), vector}
        std::vector<size_t> packed_lens{xyDesc.GetTotalSequenceLen(), lens[2]};
        std::vector<size_t> packed_strides(2);

        WA_RHEL::exclusive_scan_wa(packed_lens.crbegin(),
                                   std::next(packed_lens.crbegin(), packed_strides.size()),
                                   packed_strides.rbegin(),
                                   1LL,
                                   std::multiplies<size_t>{});

        std::vector<size_t> seq_lens_per_sample = xyDesc.GetSequenceLengthsVector();
        return {(std::move(lens)),
                (std::move(strides)),
                (std::move(packed_lens)),
                (std::move(packed_strides)),
                (std::move(seq_lens_per_sample))};
    }

    inline size_t getPackedOffset(const size_t batch_id) const
    {
        return packedStrides[0] * batch_id;
    }

    inline std::vector<size_t> getFullSeqMajorStrides() const { return packedStrides; }
    inline std::vector<size_t> getFullSeqMajorSize() const { return packedLens; }

    inline size_t getMiniBatchSize() const { return lens[0]; }
    inline size_t getMaxSeqSize() const { return lens[1]; }
    inline size_t getHiddenSize() const { return lens[2]; }

    inline size_t getSeqSize(size_t sample_id) const { return seqLensPerSample[sample_id]; }
    inline size_t getTotalSeqCnt() const { return packedLens[0]; }

    static std::vector<size_t> MakeStrides(const std::vector<size_t>& lengths)
    {
        return {lengths[1] * lengths[2], lengths[2], 1};
    }

    // private:
    //  local caching

    const std::vector<size_t> lens;
    const std::vector<size_t> strides;

    const std::vector<size_t> packedLens;
    const std::vector<size_t> packedStrides;

    const std::vector<size_t> seqLensPerSample;
};

} // namespace rnn_base
} // namespace miopen
