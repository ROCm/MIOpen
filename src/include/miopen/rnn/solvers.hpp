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

        WeightsBufferDescriptor weights_layout{static_cast<int>(input_vec_sz),
                                               static_cast<int>(hidden_vec_sz),
                                               layers_cnt,
                                               rnnDesc.biasMode,
                                               gates_cnt};

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
        // TODO replace by stride
        auto x_vec = layer_id != 0 ? weightsLayout.x_in_vec : weightsLayout.in_vec;

        // gateBlock_elements, ht_vec
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {weightsLayout.gates_cnt * weightsLayout.h_vec, x_vec}, {x_vec, 1}};
    }

    inline miopen::TensorDescriptor BuildLstmFilterHidDesc2D() const
    {
        // TODO replace by stride
        auto h_vec = weightsLayout.h_vec;

        // gateBlock_elements, ht_vec
        return miopen::TensorDescriptor{
            rnnDesc.dataType, {weightsLayout.gates_cnt * weightsLayout.h_vec, h_vec}, {h_vec, 1}};
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

} // namespace rnn_base
} // namespace miopen
