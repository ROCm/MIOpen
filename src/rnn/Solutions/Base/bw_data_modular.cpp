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

#include <miopen/rnn/solvers.hpp>
#include <miopen/rnn/base_ops.hpp>
#include <miopen/handle.hpp>

namespace miopen {

namespace rnn_base {

void RNNBackwardDataModularAlgo::PrepareWriteBuffers(const Handle& handle,
                                                     Data_t dhx,
                                                     Data_t dcx,
                                                     Data_t workSpace) const
{
    float beta = 0.;

    auto rnn_data_type = rnnDesc.dataType;
    auto ws_size       = workspaceInfo.getBufferSize();
    if(ws_size > 0)
    {
        miopen::TensorDescriptor ws_desk{rnn_data_type, {1, ws_size}, {ws_size, 1}};
        SetTensor(handle, ws_desk, workSpace, &beta);
    }

    if(dhx != nullptr || (rnnDesc.rnnMode == miopenLSTM && dcx != nullptr))
    {
        auto cxhx_desc = BuildHxCxDesc3D(rnnDesc.nLayers, hiddenHxCxInfo.getMiniBatchSize());

        if(dhx != nullptr)
        {
            SetTensor(handle, cxhx_desc, dhx, &beta);
        }
        if(rnnDesc.rnnMode == miopenLSTM && dcx != nullptr)
        {
            SetTensor(handle, cxhx_desc, dcx, &beta);
        }
    }
}

void RNNBackwardDataModularAlgo::PropDhy(const Handle& handle,
                                         ConstData_t dhy,
                                         Data_t workSpace,
                                         unsigned int layer,
                                         const SequenceIterator& currentSeq,
                                         SequenceDirection direction) const
{
    if(dhy == nullptr)
        return;

    if(direction == SequenceDirection::Reverse && !currentSeq.isFirst())
        return;

    const auto [copy_batch_size, copy_batch_offset_id] = [](const SequenceIterator& current_seq,
                                                            const BatchController& b_c) {
        const auto cur_time_batch = b_c.getBatchSize(current_seq.getPhisVal());
        const auto prev_time_batch =
            current_seq.isFirst() ? 0 : b_c.getBatchSize(current_seq.getPrev().getPhisVal());

        size_t dst_batch_offset_id_ = prev_time_batch;
        size_t dst_batch_size_      = cur_time_batch - prev_time_batch;
        return std::make_tuple(dst_batch_size_, dst_batch_offset_id_);
    }(currentSeq, batchController);

    // no data so return
    if(copy_batch_size <= 0)
        return;

    // ws_dy + dhy
    const float alpha0 = 1;
    const float alpha1 = 1;
    const float beta_t = 0;

    // TODO remove virtual in implementation change getOffset
    auto virtual_layer      = getVirtualLayer(layer, direction);
    size_t dhy_layer_offset = hiddenHxCxInfo.getOffset(virtual_layer, copy_batch_offset_id);

    size_t time_batch_offset_id = batchController.getBatchSum(currentSeq.getPhisVal());
    size_t workspace_dy_offset  = workspaceInfo.getHiddenStateOffset(
        layer, time_batch_offset_id + copy_batch_offset_id, direction);

    const auto dhy_desc = BuildHxCxDesc3D(1, copy_batch_size);

    const auto workspace_dy_desc = BuildTempDhtDesc3D(1, copy_batch_size);

    OpTensor(handle,
             miopenTensorOpAdd,
             &alpha0,
             dhy_desc,
             dhy,
             &alpha1,
             workspace_dy_desc,
             workSpace,
             &beta_t,
             workspace_dy_desc,
             workSpace,
             dhy_layer_offset,
             workspace_dy_offset,
             workspace_dy_offset);
}

void RNNBackwardDataModularAlgo::PropHiddenDht(const Handle& handle,
                                               ConstData_t w,
                                               Data_t workSpace,
                                               int layer,
                                               const SequenceIterator& currentSeq,
                                               SequenceDirection direction) const
{
    // iterating over seq in descending order(from high to low)
    // take smallest batch
    const auto gemm_batch_size = batchController.getBatchSize(
        direction == SequenceDirection::Forward ? currentSeq.getPhisVal()
                                                : currentSeq.getNext().getPhisVal());

    // no gemm work
    if(gemm_batch_size == 0)
        return;

    const miopen::TensorDescriptor tmp_block_src_dsc =
        BuildLstmTmpBlockDesc2D(workspaceInfo, gemm_batch_size);

    const miopen::TensorDescriptor& filter_src_dsc = BuildLstmFilterHidDesc2D();

    const miopen::TensorDescriptor& ht_dest_dsc = BuildWsHtDesc2D(gemm_batch_size);

    RnnBaseFunctions::BWD_GEMM_Hidden_Prop(
        handle,
        workSpace,
        workspaceInfo.getGateBlockOffset(
            layer, batchController.getBatchSum(currentSeq.getPhisVal()), direction),
        tmp_block_src_dsc,
        w,
        weightsLayout.getMatrixHidOff(layer, static_cast<int>(direction)),
        filter_src_dsc,
        workSpace,
        workspaceInfo.getHiddenStateOffset(
            layer, batchController.getBatchSum(currentSeq.getNext().getPhisVal()), direction),
        ht_dest_dsc);
}

void RNNBackwardDataModularAlgo::UpdateHStatePerTimeSeq(const Handle& handle,
                                                        ConstData_t dcy,
                                                        ConstData_t cx,
                                                        Data_t,
                                                        Data_t workSpace,
                                                        Data_t reserveSpace,
                                                        int layer,
                                                        const SequenceIterator& seq,
                                                        SequenceDirection direction) const
{
    // Inited
    const size_t hidden_vec = rnnDesc.hsize;
    auto rnn_data_type      = rnnDesc.dataType;
    auto rnn_mode           = rnnDesc.rnnMode;
    auto rnn_algo_mode      = rnnDesc.algoMode;

    if(rnn_mode == miopenRNNRELU || rnn_mode == miopenRNNTANH)
    {
        // float alpha = 1;
        // float beta  = 0;
        //
        //// activation
        // auto& activDesc = rnn_mode == miopenRNNRELU ? reluDesc : tanhDesc;

        /*
        activDesc.Backward(handle,
                           &alpha,
                           dht_desc,
                           reserveSpace,
                           dht_desc,
                           workSpace,
                           dht_desc,
                           reserveSpace,
                           &beta,
                           sp_desc,
                           workSpace,
                           offset + static_cast<size_t>(ri) * wei_len +
                               static_cast<size_t>(nLayers) * batch_n * hy_stride,
                           offset + static_cast<size_t>(ri) * wei_len,
                           offset + static_cast<size_t>(ri) * wei_len,
                           offset + static_cast<size_t>(ri) * wei_len);
        */
    }
    else if(rnn_mode == miopenLSTM)
    {
        if(rnn_algo_mode == miopenRNNdefault)
        {

            size_t cur_batch = batchController.getBatchSize(seq.getPhisVal());

            const auto [dcy_use_batch, cx_use_batch] = [](const auto& seq,
                                                          const BatchController& batch_c,
                                                          const SequenceDirection dir) {
                auto current_batch = batch_c.getBatchSize(seq.getPhisVal());
                if(dir == SequenceDirection::Forward)
                {
                    const auto dcy_batch = seq.isFirst()
                                               ? current_batch
                                               : batch_c.getBatchSize(seq.getPrev().getPhisVal());
                    const auto cx_batch  = current_batch;
                    return std::make_tuple(dcy_batch, cx_batch);
                }
                else
                {
                    const auto dcy_batch = current_batch;
                    const auto cx_batch  = seq.isLast()
                                               ? current_batch
                                               : batch_c.getBatchSize(seq.getNext().getPhisVal());
                    return std::make_tuple(dcy_batch, cx_batch);
                }
            }(seq, batchController, direction);

            size_t cur_comb_dim  = batchController.getBatchSum(seq.getPhisVal());
            size_t prev_comb_dim = !seq.isFirst()
                                       ? batchController.getBatchSum(seq.getPrev().getPhisVal())
                                       : batchController.getBatchSum(seq.getPhisVal());
            size_t next_comb_dim = !seq.isLast()
                                       ? batchController.getBatchSum(seq.getNext().getPhisVal())
                                       : batchController.getBatchSum(seq.getPhisVal());

            LSTMBackwardHiddenStateUpdate(
                handle,
                rnn_data_type,
                seq.isLast(),  // ti == 0,
                seq.isFirst(), // ti == seqLen - 1,
                static_cast<int>(direction),
                batchController.getBatchSize(0),
                cur_batch,
                dcy_use_batch,
                cx_use_batch,
                hidden_vec,
                reservLayout.gateStride[1],
                -666, // unused
                -666, // unused
                cx,
                hiddenHxCxInfo.getOffset(getVirtualLayer(layer, direction), 0),
                reserveSpace,
                reservLayout.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::I),
                reservLayout.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::F),
                reservLayout.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::O),
                reservLayout.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::G),
                reservLayout.getActiveCellOffset(layer, cur_comb_dim, direction),
                reservLayout.getGasOffset( // TODO
                    layer,
                    next_comb_dim,
                    direction,
                    LstmGateAndState::St),
                dcy,
                hiddenHxCxInfo.getOffset(getVirtualLayer(layer, direction), 0),
                workSpace,
                workspaceInfo.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::I),
                workspaceInfo.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::F),
                workspaceInfo.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::O),
                workspaceInfo.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::G),
                workspaceInfo.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::St),
                workspaceInfo.getGasOffset(layer, prev_comb_dim, direction, LstmGateAndState::St),
                workspaceInfo.getGasOffset(layer, cur_comb_dim, direction, LstmGateAndState::Ht),
                workspaceInfo.getGasOffset(layer, prev_comb_dim, direction, LstmGateAndState::F));
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "TODO implementation algoMode != miopenRNNdefault");
            // TODO implementation
        }
    }
    else if(rnn_mode == miopenGRU)
    {
        MIOPEN_THROW(miopenStatusInternalError, "TODO implementation miopenGRU");
        // TODO implementation
    }
}

void RNNBackwardDataModularAlgo::PropDhxDcx(const Handle& handle,
                                            ConstData_t w,
                                            Data_t dhx,
                                            Data_t dcx,
                                            Data_t workSpace,
                                            Data_t reserveSpace,
                                            size_t layer,
                                            const SequenceIterator& currentSeq,
                                            SequenceDirection direction) const
{
    // dcx, dhx
    if(!(dhx != nullptr || (rnnDesc.rnnMode == miopenLSTM && dcx != nullptr)))
        return;

    if(direction == SequenceDirection::Forward && !currentSeq.isLast())
        return;

    const auto next_batch_size =
        currentSeq.isLast()
            ? 0 // forward and reverse
            : batchController.getBatchSize(currentSeq.getNext().getPhisVal()); // only reverse

    const auto batch_size = batchController.getBatchSize(currentSeq.getPhisVal()) - next_batch_size;

    if(batch_size > 0)
    {

        const size_t hx_cx_offset =
            hiddenHxCxInfo.getOffset(getVirtualLayer(layer, direction), next_batch_size);

        const size_t acc_batch_offset =
            batchController.getBatchSum(currentSeq.getPhisVal()) + next_batch_size;

        if(dhx != nullptr)
        {
            const miopen::TensorDescriptor hx_desc = BuildHxCxDesc2D(batch_size);

            const miopen::TensorDescriptor tmp_block_src_dsc =
                BuildLstmTmpBlockDesc2D(workspaceInfo, batch_size);

            const size_t tmp_block_src_offset =
                workspaceInfo.getGateBlockOffset(layer, acc_batch_offset, direction);

            const miopen::TensorDescriptor& filter_src_dsc = BuildLstmFilterHidDesc2D();
            const size_t filter_src_offset =
                weightsLayout.getMatrixHidOff(layer, static_cast<int>(direction));

            RnnBaseFunctions::BWD_GEMM_Hidden_Prop(handle,
                                                   workSpace,
                                                   tmp_block_src_offset,
                                                   tmp_block_src_dsc,
                                                   w,
                                                   filter_src_offset,
                                                   filter_src_dsc,
                                                   dhx,
                                                   hx_cx_offset,
                                                   hx_desc);
        }

        if(rnnDesc.rnnMode == miopenLSTM && dcx != nullptr)
        {
            miopen::TensorDescriptor cx_desc = BuildHxCxDesc3D(1, batch_size);
            const auto& temp_ct_desc         = BuildTempDhtDesc3D(1, batch_size);

            const float alpha0 = 1;
            const float alpha1 = 1;
            const float beta_t = 1;

            const auto bOffset = rnnDesc.algoMode == miopenRNNdefault
                                     ? reservLayout.getGasOffset(layer,
                                                                 acc_batch_offset,
                                                                 direction,
                                                                 LstmGateAndState::F)
                                     : reservLayout.getActiveCellOffset( // TODO double check
                                           layer,
                                           acc_batch_offset,
                                           direction);

            const auto a_offset = workspaceInfo.getGasOffset(
                layer, acc_batch_offset, direction, LstmGateAndState::St);

            OpTensor(handle,
                     miopenTensorOpMul,
                     &alpha0,
                     temp_ct_desc,
                     workSpace,
                     &alpha1,
                     temp_ct_desc,
                     reserveSpace,
                     &beta_t,
                     cx_desc,
                     dcx,
                     a_offset,
                     bOffset,
                     hx_cx_offset);
        }
    }
}

void RNNBackwardDataModularAlgo::PropDy(const Handle& handle,
                                        ConstData_t dy,
                                        Data_t workSpace) const
{
    const auto rnn_data_type = rnnDesc.dataType;
    const auto last_layer_id = rnnDesc.nLayers - 1;

    auto store_offset =
        workspaceInfo.getHiddenStateOffset(last_layer_id, 0, SequenceDirection::Forward);

    // bwd concat
    // currently supported only one type, but should be more
    auto [dy_src_desc, dy_data_ptr] =
        [](const IOBufferDescriptor& dyInfo, const RNNDescriptor& rnnD, ConstData_t dy) {
            const auto& dy_raw_size   = dyInfo.getFullSeqMajorSize();
            const auto& dy_raw_stride = dyInfo.getFullSeqMajorStrides();

            size_t direc_scale = rnnD.dirMode == miopenRNNbidirection ? 2 : 1;

            const auto dy_normalized_size =
                std::vector<size_t>{1, dy_raw_size[0], direc_scale, dy_raw_size[1] / direc_scale};

            const auto dy_normalized_stride =
                std::vector<size_t>{dy_normalized_size[1] * dy_raw_stride[0] /*unused*/,
                                    dy_raw_stride[0],
                                    dy_normalized_size[3] * dy_raw_stride[1],
                                    dy_raw_stride[1]};

            auto dy_desc =
                miopen::TensorDescriptor(rnnD.dataType, dy_normalized_size, dy_normalized_stride);

            return std::make_tuple(dy_desc, dy);
        }(yInfo, rnnDesc, dy);

    const std::vector<size_t> ws_dst_strides = [](const auto& full_stride_ref) {
        return std::vector<size_t>(full_stride_ref.begin(), full_stride_ref.end());
    }(workspaceInfo.getHiddenStateStride());

    const std::vector<size_t> ws_dst_size = [](const auto& full_size_ref) {
        std::vector<size_t> ws_ht_layer_size(full_size_ref.begin(), full_size_ref.end());

        ws_ht_layer_size[0] = 1;

        return ws_ht_layer_size;
    }(workspaceInfo.hStateSizes);

    auto ws_dy_dst_desc = miopen::TensorDescriptor(rnn_data_type, ws_dst_size, ws_dst_strides);

    if(store_offset <= INT32_MAX)
    {
        CopyTensor(handle,
                   dy_src_desc,
                   dy,
                   ws_dy_dst_desc,
                   workSpace,
                   0,
                   static_cast<int>(store_offset),
                   true);
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "store_offset > INT32_MAX");
    }
}

void RNNBackwardDataModularAlgo::PropHiddenDy(const Handle& handle,
                                              ConstData_t w,
                                              Data_t workSpace,
                                              Data_t reserveSpace,
                                              size_t layer,
                                              SequenceDirection direction) const
{
    if(layer == 0)
        return;

    const size_t gemm_batch_size = batchController.getTotalBatchSum();

    const size_t gemm_batch_offset = 0;

    return PropHiddenDy(
        handle, w, workSpace, reserveSpace, layer, direction, gemm_batch_size, gemm_batch_offset);
}

void RNNBackwardDataModularAlgo::PropHiddenDy(const Handle& handle,
                                              ConstData_t w,
                                              Data_t workSpace,
                                              Data_t reserveSpace,
                                              size_t layer,
                                              SequenceDirection direction,
                                              const SequenceIterator& firstSeq,
                                              const SequenceIterator& lastSeq) const
{
    if(layer == 0)
        return;

    auto start_phis_seq = std::min(firstSeq.getPhisVal(), lastSeq.getPhisVal());
    auto end_phis_seq   = std::max(firstSeq.getPhisVal(), lastSeq.getPhisVal());

    const size_t gemm_batch_size = batchController.getBatchSum(end_phis_seq) -
                                   batchController.getBatchSum(start_phis_seq) +
                                   batchController.getBatchSize(end_phis_seq);

    const size_t gemm_batch_offset = batchController.getBatchSum(start_phis_seq);

    return PropHiddenDy(
        handle, w, workSpace, reserveSpace, layer, direction, gemm_batch_size, gemm_batch_offset);
}

void RNNBackwardDataModularAlgo::PropHiddenDy(const Handle& handle,
                                              ConstData_t w,
                                              Data_t workSpace,
                                              Data_t reserveSpace,
                                              size_t layer,
                                              SequenceDirection direction,
                                              size_t gemm_batch_size,
                                              size_t gemm_batch_offset) const
{

    if(layer == 0)
        return;

    const auto rnn_data_type = rnnDesc.dataType;

    const auto ht_x_offset =
        workspaceInfo.getHiddenStateOffset(layer - 1, gemm_batch_offset, direction);

    const auto filter_offset = weightsLayout.getMatrixXinOff(layer, static_cast<int>(direction));

    if(rnnDesc.rnnMode == miopenLSTM)
    {
        const auto tmp_block_offset =
            workspaceInfo.getGateBlockOffset(layer, gemm_batch_offset, direction);

        const miopen::TensorDescriptor tmp_block_src_dsc =
            BuildLstmTmpBlockDesc2D(workspaceInfo, gemm_batch_size);

        const auto filter_src_dsc = BuildLstmFilterXDesc2D(layer);

        const auto ht_x_desc = BuildWsHtDesc2D(gemm_batch_size);

        RnnBaseFunctions::BWD_GEMM_Hidden_Prop(handle,
                                               workSpace,
                                               tmp_block_offset,
                                               tmp_block_src_dsc,
                                               w,
                                               filter_offset,
                                               filter_src_dsc,
                                               workSpace,
                                               ht_x_offset,
                                               ht_x_desc);
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Only lstm");
    }

    if(!float_equal(miopen::deref(rnnDesc.dropoutDesc).dropout, 0))
    {
        // size_t hid_shift   = layer * reservLayout.hStateSizes[1] * hy_stride;
        // auto dhd_off     = direction_mult * hy_h * 5;
        const size_t dst_data_offset = ht_x_offset; // hid_shift+dhd_off

        auto h_state_sizes = reservLayout.hStateSizes;

        // TODO 3 dim vec, add direction as dim
        std::vector<size_t> drop_size(2), drop_in_str(2, 1);
        drop_size[0] = h_state_sizes[1];                    // batch_n;
        drop_size[1] = h_state_sizes[2] * h_state_sizes[3]; // hy_h* direction_mult;

        drop_in_str[0] = reservLayout.getHiddenStateStride()[1]; // hy_stride;
        drop_in_str[1] = reservLayout.getHiddenStateStride()[3]; // 1;

        auto drop_in_desc = miopen::TensorDescriptor(rnn_data_type, drop_size, drop_in_str);

        size_t drop_rsv_size = drop_in_desc.GetElementSize();

        size_t drop_rsv_start = reservLayout.getBufferSize();

        size_t drop_rsv_offset = (drop_rsv_start + (rnnDesc.nLayers - 1) * drop_rsv_size) *
                                     (rnn_data_type == miopenFloat ? 4 : 2) +
                                 layer * drop_rsv_size;

        miopen::deref(rnnDesc.dropoutDesc)
            .DropoutBackward(handle,
                             drop_in_desc,
                             drop_in_desc,
                             workSpace,
                             drop_in_desc,
                             workSpace,
                             reserveSpace,
                             drop_rsv_size,
                             dst_data_offset,
                             dst_data_offset,
                             drop_rsv_offset);
    }
}

void RNNBackwardDataModularAlgo::PropDx(const Handle& handle,
                                        ConstData_t w,
                                        ConstData_t workSpace,
                                        Data_t dx,
                                        SequenceDirection direction,
                                        const SequenceIterator& firstSeq,
                                        const SequenceIterator& lastSeq) const
{

    auto start_phis_seq = std::min(firstSeq.getPhisVal(), lastSeq.getPhisVal());
    auto end_phis_seq   = std::max(firstSeq.getPhisVal(), lastSeq.getPhisVal());

    const size_t gemm_batch_size = batchController.getBatchSum(end_phis_seq) -
                                   batchController.getBatchSum(start_phis_seq) +
                                   batchController.getBatchSize(end_phis_seq);

    const size_t gemm_batch_offset = batchController.getBatchSum(start_phis_seq);

    return PropDx(handle, w, workSpace, dx, direction, gemm_batch_offset, gemm_batch_size);
}

void RNNBackwardDataModularAlgo::PropDx(const Handle& handle,
                                        ConstData_t w,
                                        ConstData_t workSpace,
                                        Data_t dx,
                                        SequenceDirection direction) const
{
    const size_t gemm_batch_offset = 0;

    const size_t gemm_batch_size = workspaceInfo.getGateBlockSize()[1];
    return PropDx(handle, w, workSpace, dx, direction, gemm_batch_offset, gemm_batch_size);
}

void RNNBackwardDataModularAlgo::PropDx(const Handle& handle,
                                        ConstData_t w,
                                        ConstData_t workSpace,
                                        Data_t dx,
                                        SequenceDirection direction,
                                        size_t gemm_batch_offset,
                                        size_t gemm_batch_size) const
{
    // TODO
    assert(rnnDesc.inputMode == miopenRNNlinear);

    constexpr size_t layer = 0;

    const auto tmp_block_offset =
        workspaceInfo.getGateBlockOffset(layer, gemm_batch_offset, direction);

    const auto filter_offset = weightsLayout.getMatrixXinOff(layer, static_cast<int>(direction));

    const auto ht_x_offset = xInfo.getPackedOffset(gemm_batch_offset);

    const miopen::TensorDescriptor tmp_block_src_dsc =
        BuildLstmTmpBlockDesc2D(workspaceInfo, gemm_batch_size);

    const auto filter_src_dsc = BuildLstmFilterXDesc2D(layer);

    const auto ht_x_desc =
        [](const miopenDataType_t dType, const auto& buf_info, const size_t batch_size) {
            const auto& ht_stride = buf_info.getFullSeqMajorStrides();
            const auto& ht_size   = buf_info.getFullSeqMajorSize();

            // batch, vec_elements
            return miopen::TensorDescriptor{dType, {batch_size, ht_size[1]}, ht_stride};
        }(rnnDesc.dataType, xInfo, gemm_batch_size);

    RnnBaseFunctions::BWD_GEMM_Hidden_Prop(handle,
                                           workSpace,
                                           tmp_block_offset,
                                           tmp_block_src_dsc,
                                           w,
                                           filter_offset,
                                           filter_src_dsc,
                                           dx,
                                           ht_x_offset,
                                           ht_x_desc,
                                           false);
}

} // namespace rnn_base
} // namespace miopen
