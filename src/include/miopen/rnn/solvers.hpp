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
#include "miopen/rnn/tmp_buffer_utils.hpp"

namespace miopen {

namespace rnn_base {

class RNNBackwardDataModularAlgo
{
public:
    static RNNBackwardDataModularAlgo create(const RNNDescriptor& rnnDesc,
                                             const SeqTensorDescriptor& xDesc,
                                             const SeqTensorDescriptor& yDesc,
                                             const TensorDescriptor& hDesc)
    {
        auto [max_layers_hid, max_batch_hid, hidden_vec_sz] = miopen::tien<3>(hDesc.GetLengths());
        auto [max_batch_in, max_seq, input_vec_sz]          = miopen::tien<3>(xDesc.GetLengths());

        assert(max_batch_in <= max_batch_hid);

        auto layers_cnt         = static_cast<int>(rnnDesc.nLayers);
        const bool is_seq_bidir = rnnDesc.dirMode == miopenRNNbidirection;

        assert(static_cast<size_t>(layers_cnt) * (is_seq_bidir ? 2 : 1) <= max_layers_hid);

        auto gates_cnt = static_cast<int>(rnnDesc.nHiddenTensorsPerLayer);

        // class update req
        assert(!is_seq_bidir);
        const size_t seq_directions = is_seq_bidir ? 2 : 1;

        // TODO all size_t
        GeneralLstmRedBuffer rb_layout = GeneralLstmRedBuffer::build(
            layers_cnt, xDesc.GetTotalSequenceLen(), seq_directions, hidden_vec_sz);

        GeneralLstmTempBuffer workspace_info = GeneralLstmTempBuffer::build(
            layers_cnt, xDesc.GetTotalSequenceLen(), seq_directions, hidden_vec_sz);

        WeightsBufferDescriptor weights_layout =
            WeightsBufferDescriptor::create(static_cast<int>(input_vec_sz),
                                            static_cast<int>(hidden_vec_sz),
                                            layers_cnt,
                                            rnnDesc.biasMode,
                                            rnnDesc.inputMode,
                                            gates_cnt,
                                            is_seq_bidir);

        BatchController batch_controller = BatchController::Create(xDesc);

        HiddenBuffersDescriptor hidden_hxcx_info{hDesc};

        IOBufferDescriptor x_info{IOBufferDescriptor::build(xDesc)};
        IOBufferDescriptor y_info{IOBufferDescriptor::build(yDesc)};

        return {std::move(rb_layout),
                workspace_info,
                weights_layout,
                hidden_hxcx_info,
                x_info,
                y_info,
                rnnDesc,
                batch_controller};
    }

    void PrepareWriteBuffers(const Handle& handle, Data_t dhx, Data_t dcx, Data_t workSpace) const;

    void PropDhy(const Handle& handle,
                 ConstData_t dhy,
                 Data_t workSpace,
                 unsigned int layer,
                 const SequenceIterator& currentSeq,
                 SequenceDirection direction) const;

    void PropHiddenDht(const Handle& handle,
                       ConstData_t w,
                       Data_t workSpace,
                       int layer,
                       const SequenceIterator& currentSeq,
                       SequenceDirection direction) const;

    void UpdateHStatePerTimeSeq(const Handle& handle,
                                ConstData_t dcy,
                                ConstData_t cx,
                                Data_t,
                                Data_t workSpace,
                                Data_t reserveSpace,
                                int layer,
                                const SequenceIterator& seq,
                                SequenceDirection direction) const;

    void PropDhxDcx(const Handle& handle,
                    ConstData_t w,
                    Data_t dhx,
                    Data_t dcx,
                    Data_t workSpace,
                    Data_t reserveSpace,
                    size_t layer,
                    const SequenceIterator& currentSeq,
                    SequenceDirection direction) const;

    void PropDy(const Handle& handle, ConstData_t dy, Data_t workSpace) const;

    void PropHiddenDy(const Handle& handle,
                      ConstData_t w,
                      Data_t workSpace,
                      Data_t reserveSpace,
                      size_t layer,
                      SequenceDirection direction) const;

    void PropHiddenDy(const Handle& handle,
                      ConstData_t w,
                      Data_t workSpace,
                      Data_t reserveSpace,
                      size_t layer,
                      SequenceDirection direction,
                      const SequenceIterator& firstSeq,
                      const SequenceIterator& lastSeq) const;

    void PropHiddenDy(const Handle& handle,
                      ConstData_t w,
                      Data_t workSpace,
                      Data_t reserveSpace,
                      size_t layer,
                      SequenceDirection direction,
                      size_t gemm_batch_size,
                      size_t gemm_batch_offset) const;

    void PropDx(const Handle& handle,
                ConstData_t w,
                ConstData_t workSpace,
                Data_t dx,
                SequenceDirection direction,
                const SequenceIterator& firstSeq,
                const SequenceIterator& lastSeq) const;

    void PropDx(const Handle& handle,
                ConstData_t w,
                ConstData_t workSpace,
                Data_t dx,
                SequenceDirection direction) const;

    void PropDx(const Handle& handle,
                ConstData_t w,
                ConstData_t workSpace,
                Data_t dx,
                SequenceDirection direction,
                size_t gemm_batch_offset,
                size_t gemm_batch_size) const;
    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

private:
    RNNBackwardDataModularAlgo(GeneralLstmRedBuffer rb_layout,
                               GeneralLstmTempBuffer workspace_info,
                               WeightsBufferDescriptor weights_layout,
                               HiddenBuffersDescriptor hidden_hxcx_info,
                               IOBufferDescriptor x_info,
                               IOBufferDescriptor y_info,
                               const RNNDescriptor& rnn_desc,
                               BatchController batch_controller)
        : reservLayout(std::move(rb_layout)),
          workspaceInfo(std::move(workspace_info)),
          weightsLayout(std::move(weights_layout)),
          hiddenHxCxInfo(std::move(hidden_hxcx_info)),
          xInfo(std::move(x_info)),
          yInfo(std::move(y_info)),
          rnnDesc(rnn_desc),
          batchController(std::move(batch_controller))
    {
    }

    template <typename BufType>
    inline miopen::TensorDescriptor BuildLstmTmpBlockDesc2D(const BufType& buf_info,
                                                            const size_t batch_size) const
    {
        const std::array<size_t, 4>& tmp_block_stride = buf_info.getGateBlockStride();
        const std::array<size_t, 4>& tmp_block_size   = buf_info.getGateBlockSize();

        // batch, gateBlock_elements
        return miopen::TensorDescriptor{rnnDesc.dataType,
                                        {batch_size, tmp_block_size[3]},
                                        {tmp_block_stride[1], tmp_block_stride[3]}};
    }

    inline miopen::TensorDescriptor BuildLstmFilterXDesc2D(int layer_id) const
    {
        assert(rnnDesc.inputMode == 0 || layer_id != 0);
        // TODO replace by stride
        auto x_vec = layer_id != 0 ? weightsLayout.xInVec : weightsLayout.inVec;

        // gateBlock_elements, ht_vec
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {weightsLayout.gatesCnt * weightsLayout.hVec, x_vec}, {x_vec, 1}};
    }

    inline miopen::TensorDescriptor BuildLstmFilterHidDesc2D() const
    {
        // TODO replace by stride
        auto h_vec = weightsLayout.hVec;

        // gateBlock_elements, ht_vec
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {weightsLayout.gatesCnt * weightsLayout.hVec, h_vec}, {h_vec, 1}};
    }

    inline miopen::TensorDescriptor BuildWsHtDesc2D(size_t batch_size) const
    {
        auto& ht_stride = workspaceInfo.getHiddenStateStride();
        auto& ht_size   = workspaceInfo.hStateSizes;

        // batch, gateBlock_elements
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {batch_size, ht_size[3]}, {ht_stride[1], ht_stride[3]}};
    }

    // 2 dims batch, vec
    inline miopen::TensorDescriptor BuildHxCxDesc2D(size_t batch_size) const
    {
        const std::vector<size_t> hx_size{batch_size, hiddenHxCxInfo.getHiddenSize()};
        const std::vector<size_t> hx_stride{hiddenHxCxInfo.getStrides()[1],
                                            hiddenHxCxInfo.getStrides()[2]};

        return miopen::TensorDescriptor{rnnDesc.dataType, hx_size, hx_stride};
    }

    // 3 dims layer, batch, vec
    inline miopen::TensorDescriptor BuildHxCxDesc3D(size_t layer_size, size_t batch_size) const
    {
        const std::vector<size_t> hx_accum_size{
            layer_size, batch_size, hiddenHxCxInfo.getHiddenSize()};

        return miopen::TensorDescriptor{
            rnnDesc.dataType, hx_accum_size, hiddenHxCxInfo.getStrides()};
    }

    // 3 dims layer, batch, vec
    inline miopen::TensorDescriptor BuildTempDhtDesc3D(size_t layer_size, size_t batch_size) const
    {
        const std::vector<size_t> dy_dhy_accum_size{
            layer_size, batch_size, hiddenHxCxInfo.getHiddenSize()};

        const auto ws_dy_stride = [](const auto& ws_4dim_strides) -> std::vector<size_t> {
            // convert 4dim stride to 3 dim without direction
            // TODO change hiddenBufferDesc
            return std::vector<size_t>{ws_4dim_strides[0], ws_4dim_strides[1], ws_4dim_strides[3]};
        }(workspaceInfo.getHiddenStateStride());

        return miopen::TensorDescriptor{rnnDesc.dataType, dy_dhy_accum_size, ws_dy_stride};
    }

    inline size_t getVirtualLayer(const size_t layer_id, SequenceDirection direction) const
    {
        return layer_id * (isBidirectSeq ? 2 : 1) +
               (direction == SequenceDirection::Forward ? 0 : 1);
    }

    const GeneralLstmRedBuffer reservLayout;
    // const WorkspaceBufferDescriptor workspaceInfo;
    const GeneralLstmTempBuffer workspaceInfo;

    const WeightsBufferDescriptor weightsLayout;
    const HiddenBuffersDescriptor hiddenHxCxInfo;
    const IOBufferDescriptor xInfo;
    const IOBufferDescriptor yInfo;

    const RNNDescriptor& rnnDesc;

    const ActivationDescriptor tanhDesc = {miopenActivationTANH, 1, 1, 1};
    const ActivationDescriptor sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
    const ActivationDescriptor reluDesc = {miopenActivationRELU, 1, 0, 1};

    const BatchController batchController;

    const bool isBidirectSeq = false;
};

class RNNModularSingleStreamBWD
{
public:
    RNNModularSingleStreamBWD(const RNNDescriptor& rnn,
                              const SeqTensorDescriptor& xDesc,
                              const SeqTensorDescriptor& yDesc,
                              const TensorDescriptor& hDesc)
        : rnnAlgoModules(RNNBackwardDataModularAlgo::create(rnn, xDesc, yDesc, hDesc)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    void ComputeBWD(Handle& handle,
                    ConstData_t dy,
                    ConstData_t dhy,
                    Data_t dhx,
                    ConstData_t cx,
                    ConstData_t dcy,
                    Data_t dcx,
                    Data_t dx,
                    ConstData_t w,
                    Data_t workSpace,
                    Data_t reserveSpace) const;

    const rnn_base::RNNBackwardDataModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

class RNNModularMultiStreamBWD
{
public:
    RNNModularMultiStreamBWD(const RNNDescriptor& rnn,
                             const SeqTensorDescriptor& xDesc,
                             const SeqTensorDescriptor& yDesc,
                             const TensorDescriptor& hDesc)
        : rnnAlgoModules(RNNBackwardDataModularAlgo::create(rnn, xDesc, yDesc, hDesc)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    struct runtimeArgsBwd
    {
        const Handle* handle;
        ConstData_t dy;
        ConstData_t dhy;
        Data_t dhx;
        ConstData_t cx;
        ConstData_t dcy;
        Data_t dcx;
        Data_t dx;
        ConstData_t w;
        Data_t workSpace;
        Data_t reserveSpace;
    };

    void ComputeBWD(Handle& handle,
                    ConstData_t dy,
                    ConstData_t dhy,
                    Data_t dhx,
                    ConstData_t cx,
                    ConstData_t dcy,
                    Data_t dcx,
                    Data_t dx,
                    ConstData_t w,
                    Data_t workSpace,
                    Data_t reserveSpace) const;

    bool ChunkDispatch(const runtimeArgsBwd& args,
                       size_t chunk_size,
                       size_t chunk_time_offset,
                       size_t chunk_layer_offset) const;

private:
    void PrologueDispatch(const runtimeArgsBwd& args) const;

    const rnn_base::RNNBackwardDataModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

class RNNBackwardWeightsModularAlgo
{
public:
    static RNNBackwardWeightsModularAlgo create(const RNNDescriptor& rnnDesc,
                                                const SeqTensorDescriptor& xDesc,
                                                const SeqTensorDescriptor& yDesc,
                                                const TensorDescriptor& hDesc)
    {
        auto [max_layers_hid, max_batch_hid, hidden_vec_sz] = miopen::tien<3>(hDesc.GetLengths());
        auto [max_batch_in, max_seq, input_vec_sz]          = miopen::tien<3>(xDesc.GetLengths());

        assert(max_batch_in <= max_batch_hid);

        size_t layers_cnt       = rnnDesc.nLayers;
        const bool is_seq_bidir = rnnDesc.dirMode == miopenRNNbidirection;

        assert(layers_cnt * (is_seq_bidir ? 2 : 1) <= max_layers_hid);

        auto gates_cnt = static_cast<int>(rnnDesc.nHiddenTensorsPerLayer);

        // class update req
        assert(!is_seq_bidir);
        const size_t seq_directions = is_seq_bidir ? 2 : 1;

        GeneralLstmRedBuffer rb_layout = GeneralLstmRedBuffer::build(
            layers_cnt, xDesc.GetTotalSequenceLen(), seq_directions, hidden_vec_sz);

        GeneralLstmTempBuffer workspace_info = GeneralLstmTempBuffer::build(
            layers_cnt, xDesc.GetTotalSequenceLen(), seq_directions, hidden_vec_sz);

        WeightsBufferDescriptor weights_layout = WeightsBufferDescriptor::create(input_vec_sz,
                                                                                 hidden_vec_sz,
                                                                                 layers_cnt,
                                                                                 rnnDesc.biasMode,
                                                                                 rnnDesc.inputMode,
                                                                                 gates_cnt,
                                                                                 is_seq_bidir);

        BatchController batch_controller = BatchController::Create(xDesc);

        HiddenBuffersDescriptor hidden_hxcx_info{hDesc};

        IOBufferDescriptor x_info{IOBufferDescriptor::build(xDesc)};
        IOBufferDescriptor y_info{IOBufferDescriptor::build(yDesc)};

        return {std::move(rb_layout),
                workspace_info,
                weights_layout,
                hidden_hxcx_info,
                x_info,
                y_info,
                rnnDesc,
                batch_controller};
    }

    void PrepareWriteBuffers(const Handle& handle, Data_t w) const;

    void PhisXInputWeights(const Handle& handle, Data_t dw, Data_t workSpace, ConstData_t x) const;

    void HiddenXInputWeights(const Handle& handle,
                             Data_t dw,
                             ConstData_t workSpace,
                             ConstData_t reserveSpace,
                             size_t layer) const;

    void BiasUpdate(const Handle& handle,
                    Data_t dw,
                    Data_t workSpace,
                    size_t layer,
                    size_t workSpaceSize) const;

    void HiddenHStateWeights(const Handle& handle,
                             Data_t dw,
                             ConstData_t workSpace,
                             ConstData_t reserveSpace,
                             const SequenceIterator& seq,
                             size_t layer,
                             SequenceDirection direction) const
    {
        const size_t gemm_batch_size = [&]() -> size_t {
            if(seq.isFirst())
                return 0;

            if(direction == SequenceDirection::Reverse)
                return batchController.getBatchSize(seq.getPhisVal());
            else
                return batchController.getBatchSize(seq.getPrev().getPhisVal());
        }();

        if(gemm_batch_size != 0)
            return HiddenHStateWeights_Unchecked(
                handle, dw, workSpace, reserveSpace, seq, layer, direction, gemm_batch_size);
    }

    void HiddenHStateWeights(const Handle& handle,
                             Data_t dw,
                             ConstData_t workSpace,
                             ConstData_t reserveSpace,
                             size_t layer,
                             size_t max_seq_len,
                             const SequenceDirection direction) const
    {
        size_t start_seq_id   = 0;
        const size_t last_seq = max_seq_len - 1;
        for(auto i = start_seq_id + 1; i <= last_seq; i++)
        {

            if(batchController.getBatchSize(i) != batchController.getBatchSize(start_seq_id) ||
               i == last_seq)
            {
                const size_t gemm_batch_size = (batchController.getBatchSum(i - 1) -
                                                batchController.getBatchSum(start_seq_id)) +
                                               batchController.getBatchSize(i);

                if(gemm_batch_size != 0)
                {
                    const auto first_logical_val = direction == SequenceDirection::Forward
                                                       ? start_seq_id
                                                       : (max_seq_len - 1) - start_seq_id - 1;
                    const auto seq =
                        SequenceIterator(first_logical_val, direction, max_seq_len, false);

                    HiddenHStateWeights_Unchecked(handle,
                                                  dw,
                                                  workSpace,
                                                  reserveSpace,
                                                  seq,
                                                  layer,
                                                  direction,
                                                  gemm_batch_size);
                }
                start_seq_id = i;
            }
        }
    }

    void PhisHStateWeights(const Handle& handle,
                           Data_t dw,
                           ConstData_t workSpace,
                           ConstData_t hx,
                           size_t layer,
                           size_t max_seq_len,
                           SequenceDirection direction) const
    {
        if(hx == nullptr)
            return;

        for(auto i = max_seq_len; i > 0; i--)
        {
            const auto seq = SequenceIterator(i - 1, direction, max_seq_len, false);

            PhisHStateWeights(handle, dw, workSpace, hx, seq, layer, direction);
        }
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

private:
    RNNBackwardWeightsModularAlgo(GeneralLstmRedBuffer rb_layout,
                                  GeneralLstmTempBuffer workspace_info,
                                  WeightsBufferDescriptor weights_layout,
                                  HiddenBuffersDescriptor hidden_hxcx_info,
                                  IOBufferDescriptor x_info,
                                  IOBufferDescriptor y_info,
                                  const RNNDescriptor& rnn_desc,
                                  BatchController batch_controller)
        : reservLayout(std::move(rb_layout)),
          workspaceInfo(std::move(workspace_info)),
          weightsLayout(std::move(weights_layout)),
          hiddenHxCxInfo(std::move(hidden_hxcx_info)),
          xInfo(std::move(x_info)),
          yInfo(std::move(y_info)),
          rnnDesc(rnn_desc),
          batchController(std::move(batch_controller))
    {
    }

    void HiddenHStateWeights_Unchecked(const Handle& handle,
                                       Data_t dw,
                                       ConstData_t workSpace,
                                       ConstData_t reserveSpace,
                                       const SequenceIterator& seq,
                                       size_t layer,
                                       SequenceDirection direction,
                                       size_t gemm_batch_size) const;

    void PhisHStateWeights(const Handle& handle,
                           Data_t dw,
                           ConstData_t workSpace,
                           ConstData_t hx,
                           const SequenceIterator& seq,
                           size_t layer,
                           SequenceDirection direction) const;

    template <typename BufType>
    inline miopen::TensorDescriptor BuildLstmTmpBlockDesc2D(const BufType& buf_info,
                                                            const size_t batch_size) const
    {
        const std::array<size_t, 4>& tmp_block_stride = buf_info.getGateBlockStride();
        const std::array<size_t, 4>& tmp_block_size   = buf_info.getGateBlockSize();

        // batch, gateBlock_elements
        return miopen::TensorDescriptor{rnnDesc.dataType,
                                        {batch_size, tmp_block_size[3]},
                                        {tmp_block_stride[1], tmp_block_stride[3]}};
    }

    inline miopen::TensorDescriptor BuildLstmFilterXDesc2D(int layer_id) const
    {
        assert(rnnDesc.inputMode == 0 || layer_id != 0);
        // TODO replace by stride
        auto x_vec = layer_id != 0 ? weightsLayout.xInVec : weightsLayout.inVec;

        // gateBlock_elements, ht_vec
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {weightsLayout.gatesCnt * weightsLayout.hVec, x_vec}, {x_vec, 1}};
    }

    inline miopen::TensorDescriptor BuildLstmFilterHidDesc2D() const
    {
        // TODO replace by stride
        auto h_vec = weightsLayout.hVec;

        // gateBlock_elements, ht_vec
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {weightsLayout.gatesCnt * weightsLayout.hVec, h_vec}, {h_vec, 1}};
    }

    template <typename BufType>
    inline miopen::TensorDescriptor BuildTmpHtDesc2D(const BufType& buf_info,
                                                     size_t batch_size) const
    {
        auto& ht_stride = buf_info.getHiddenStateStride();
        auto& ht_size   = buf_info.hStateSizes;

        // batch, gateBlock_elements
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {batch_size, ht_size[3]}, {ht_stride[1], ht_stride[3]}};
    }

    // 2 dims batch, vec
    inline miopen::TensorDescriptor BuildHxCxDesc2D(size_t batch_size) const
    {
        const std::vector<size_t> hx_size{batch_size, hiddenHxCxInfo.getHiddenSize()};
        const std::vector<size_t> hx_stride{hiddenHxCxInfo.getStrides()[1],
                                            hiddenHxCxInfo.getStrides()[2]};

        return miopen::TensorDescriptor{rnnDesc.dataType, hx_size, hx_stride};
    }

    // 3 dims layer, batch, vec
    inline miopen::TensorDescriptor BuildHxCxDesc3D(size_t layer_size, size_t batch_size) const
    {
        const std::vector<size_t> hx_accum_size{
            layer_size, batch_size, hiddenHxCxInfo.getHiddenSize()};

        return miopen::TensorDescriptor{
            rnnDesc.dataType, hx_accum_size, hiddenHxCxInfo.getStrides()};
    }

    // 3 dims layer, batch, vec
    inline miopen::TensorDescriptor BuildTempDhtDesc3D(size_t layer_size, size_t batch_size) const
    {
        const std::vector<size_t> dy_dhy_accum_size{
            layer_size, batch_size, hiddenHxCxInfo.getHiddenSize()};

        const auto ws_dy_stride = [](const auto& ws_4dim_strides) -> std::vector<size_t> {
            // convert 4dim stride to 3 dim without direction
            // TODO change hiddenBufferDesc
            return std::vector<size_t>{ws_4dim_strides[0], ws_4dim_strides[1], ws_4dim_strides[3]};
        }(workspaceInfo.getHiddenStateStride());

        return miopen::TensorDescriptor{rnnDesc.dataType, dy_dhy_accum_size, ws_dy_stride};
    }

    // 3 dims layer, batch, vec
    inline miopen::TensorDescriptor BuildWeiBiasDesc2D() const
    {
        const std::vector<size_t> bias_size = [](const auto& wei_4dim_size) -> std::vector<size_t> {
            // wei_4dim_size{layer, dir, gate, vec}
            return {1, wei_4dim_size[1] * wei_4dim_size[2] * wei_4dim_size[3]};
        }(weightsLayout.getBiasSize());

        const auto bias_stride = [](const auto& wei_4dim_strides) -> std::vector<size_t> {
            // convert 4dim stride to 2 dim without direction
            return std::vector<size_t>{wei_4dim_strides[0], wei_4dim_strides[3]};
        }(weightsLayout.getBiasStride());

        return miopen::TensorDescriptor{rnnDesc.dataType, bias_size, bias_stride};
    }

    inline size_t getVirtualLayer(const size_t layer_id, SequenceDirection direction) const
    {
        return layer_id * (isBidirectSeq ? 2 : 1) +
               (direction == SequenceDirection::Forward ? 0 : 1);
    }

    inline size_t getHxBatchSizeReadAtTime(const SequenceIterator& seq,
                                           SequenceDirection direction) const
    {
        if(seq.isLast())
            return batchController.getBatchSize(seq.getPhisVal());

        if(direction == SequenceDirection::Reverse)
        {
            return batchController.getBatchSize(seq.getPhisVal()) -
                   batchController.getBatchSize(seq.getPrev().getPhisVal());
        }
        return 0;
    }

    const GeneralLstmRedBuffer reservLayout;
    // const WorkspaceBufferDescriptor workspaceInfo;
    const GeneralLstmTempBuffer workspaceInfo;

    const WeightsBufferDescriptor weightsLayout;
    const HiddenBuffersDescriptor hiddenHxCxInfo;

    const IOBufferDescriptor xInfo;
    const IOBufferDescriptor yInfo;

    const RNNDescriptor& rnnDesc;

    const ActivationDescriptor tanhDesc = {miopenActivationTANH, 1, 1, 1};
    const ActivationDescriptor sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
    const ActivationDescriptor reluDesc = {miopenActivationRELU, 1, 0, 1};

    const BatchController batchController;

    const bool isBidirectSeq = false;
};

class RNNModularSingleStreamBWWeights
{
public:
    RNNModularSingleStreamBWWeights(const RNNDescriptor& rnn,
                                    const SeqTensorDescriptor& xDesc,
                                    const SeqTensorDescriptor& yDesc,
                                    const TensorDescriptor& hDesc)
        : rnnAlgoModules(RNNBackwardWeightsModularAlgo::create(rnn, xDesc, yDesc, hDesc)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    void Compute(const Handle& handle,
                 ConstData_t x,
                 ConstData_t hx,
                 Data_t dw,
                 Data_t workSpace,
                 size_t /*workSpaceSize*/,
                 ConstData_t reserveSpace,
                 size_t /*reserveSpaceSize*/) const;

    const rnn_base::RNNBackwardWeightsModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

class RNNModularMultiStreamBWWeights
{
public:
    RNNModularMultiStreamBWWeights(const RNNDescriptor& rnn,
                                   const SeqTensorDescriptor& xDesc,
                                   const SeqTensorDescriptor& yDesc,
                                   const TensorDescriptor& hDesc)
        : rnnAlgoModules(RNNBackwardWeightsModularAlgo::create(rnn, xDesc, yDesc, hDesc)),
          rnnDesc(rnn),
          max_seq_len(xDesc.GetMaxSequenceLength())
    {
    }

    static bool IsApplicable()
    {
#if MIOPEN_USE_GEMM && MIOPEN_BACKEND_HIP
        return true;
#else
        return false;
#endif // MIOPEN_USE_GEMM&& MIOPEN_BACKEND_HIP
    }

    // TODO
    static size_t GetWsSize() { return 0; };

    struct runtimeArgsBww
    {
        const Handle* handle;
        ConstData_t x;
        ConstData_t hx;
        Data_t dw;
        Data_t workSpace;
        ConstData_t reserveSpace;
    };

    void Compute(const Handle& handle,
                 ConstData_t x,
                 ConstData_t hx,
                 Data_t dw,
                 Data_t workSpace,
                 size_t /*workSpaceSize*/,
                 ConstData_t reserveSpace,
                 size_t /*reserveSpaceSize*/) const;

private:
    void PrologueDispatch(const runtimeArgsBww& args) const;

    const rnn_base::RNNBackwardWeightsModularAlgo rnnAlgoModules;
    const RNNDescriptor& rnnDesc;
    const size_t max_seq_len;
};

} // namespace rnn_base
} // namespace miopen
