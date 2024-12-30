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

void RNNForwardDataModularAlgo::PrepareWriteBuffers(const Handle& handle,
                                                    const runtimeArgsFwd& runtimeArgs) const
{
    float beta   = 0.;
    auto rs_size = reservLayout.getBufferSize();

    if(rs_size > 0)
    {
        miopen::TensorDescriptor ws_desk{rnnDesc.dataType, {1, rs_size}, {rs_size, 1}};
        SetTensor(handle, ws_desk, runtimeArgs.reserveSpace, &beta);
    }

    if(runtimeArgs.hy != nullptr || (rnnDesc.rnnMode == miopenLSTM && runtimeArgs.cy != nullptr))
    {
        auto cxhx_desc = BuildHxCxDesc3D(rnnDesc.nLayers, hiddenHxCxInfo.getMiniBatchSize());

        if(runtimeArgs.hy != nullptr)
        {
            SetTensor(handle, cxhx_desc, runtimeArgs.hy, &beta);
        }
        if(rnnDesc.rnnMode == miopenLSTM && runtimeArgs.cy != nullptr)
        {
            SetTensor(handle, cxhx_desc, runtimeArgs.cy, &beta);
        }
    }
}

void RNNForwardDataModularAlgo::PropX(const Handle& handle, const runtimeArgsFwd& runtimeArgs) const
{
    const size_t gemm_batch_size = workspaceInfo.getGateBlockSize()[1];
    return PropX(handle, runtimeArgs, 0, gemm_batch_size);
}

void RNNForwardDataModularAlgo::PropX(const Handle& handle,
                                      const runtimeArgsFwd& runtimeArgs,
                                      size_t gemm_batch_offset,
                                      size_t gemm_batch_size) const
{
    constexpr size_t layer                = 0;
    constexpr SequenceDirection direction = SequenceDirection::Forward;
    // rnnocl 1512

    const auto tmp_block_offset =
        reservLayout.getGateBlockOffset(layer, gemm_batch_offset, direction);
    const auto ht_x_offset = xInfo.getPackedOffset(gemm_batch_offset);

    const miopen::TensorDescriptor tmp_block_src_dsc =
        BuildLstmTmpBlockDesc2D(reservLayout, gemm_batch_size);
    const auto ht_x_desc =
        [](const miopenDataType_t dType, const auto& buf_info, const size_t batch_size) {
            const auto& ht_stride = buf_info.getFullSeqMajorStrides();
            const auto& ht_size   = buf_info.getFullSeqMajorSize();

            // batch, vec_elements
            return miopen::TensorDescriptor{dType, {batch_size, ht_size[1]}, ht_stride};
        }(rnnDesc.dataType, xInfo, gemm_batch_size);

    if(rnnDesc.inputMode == miopenRNNskip)
    {
        // TODO
        assert(false);

        float alpha0 = 1;
        float alpha1 = 1;
        float beta_t = 0;

        OpTensor(handle,
                 miopenTensorOpAdd,
                 &alpha0,
                 tmp_block_src_dsc,
                 runtimeArgs.workSpace, // A
                 &alpha1,
                 ht_x_desc,
                 runtimeArgs.x, // B
                 &beta_t,
                 tmp_block_src_dsc,
                 runtimeArgs.workSpace, // C
                 tmp_block_offset,      // A offset
                 ht_x_offset,           // B offset
                 tmp_block_offset,      // C offset
                 true);

        // for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
        //{
        //    CopyTensor(handle, x_desc, x, sp_desc, workSpace, 0, gi * hy_h);
        //}
    }
    else
    {
        const auto filter_offset =
            weightsLayout.getMatrixXinOff(layer, static_cast<int>(direction));
        const auto filter_src_dsc = BuildLstmFilterXDesc2D(layer);

        RnnBaseFunctions::FWD_GEMM(handle,
                                   runtimeArgs.x,
                                   ht_x_offset,
                                   ht_x_desc,

                                   runtimeArgs.w,
                                   filter_offset,
                                   filter_src_dsc,
                                   runtimeArgs.reserveSpace,
                                   tmp_block_offset,
                                   tmp_block_src_dsc,
                                   true);
    }
}

void RNNForwardDataModularAlgo::PropHxCx(const Handle& handle,
                                         const runtimeArgsFwd& runtimeArgs,
                                         unsigned int layer,
                                         const SequenceIterator& currentSeq,
                                         SequenceDirection direction) const
{
    // 1834
    if(runtimeArgs.hx != nullptr)
    {
        const auto prev_batch =
            currentSeq.isFirst() ? 0
                                 : batchController.getBatchSize(currentSeq.getPrev().getPhisVal());

        const auto batch_offset = prev_batch;
        const auto gemm_batch_size =
            batchController.getBatchSize(currentSeq.getPhisVal()) - prev_batch;

        // rnnocl 1512

        const auto tmp_block_offset = reservLayout.getGateBlockOffset(
            layer, batchController.getBatchSum(currentSeq.getPhisVal()) + batch_offset, direction);

        const auto filter_offset =
            weightsLayout.getMatrixHidOff(layer, static_cast<int>(direction));

        const auto ht_x_offset = hiddenHxCxInfo.getOffset(layer, batch_offset);

        const miopen::TensorDescriptor tmp_block_src_dsc =
            BuildLstmTmpBlockDesc2D(reservLayout, gemm_batch_size);

        const miopen::TensorDescriptor filter_src_dsc = BuildLstmFilterHidDesc2D();

        const miopen::TensorDescriptor& ht_x_desc = BuildHxCxDesc2D(gemm_batch_size);

        RnnBaseFunctions::FWD_GEMM(handle,
                                   runtimeArgs.hx,
                                   ht_x_offset,
                                   ht_x_desc,
                                   runtimeArgs.w,
                                   filter_offset,
                                   filter_src_dsc,
                                   runtimeArgs.reserveSpace,
                                   tmp_block_offset,
                                   tmp_block_src_dsc,
                                   true);
    }
}

void RNNForwardDataModularAlgo::AddBias(const Handle& handle,
                                        const runtimeArgsFwd& runtimeArgs) const
{
    if(rnnDesc.biasMode == miopenRNNNoBias)
        return;

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    float alpha0 = 1;
    float alpha1 = 1;
    float beta_t = 0;

    // single layer, single direction
    const auto bias_desc = miopen::TensorDescriptor(
        rnnDesc.dataType,
        std::vector<size_t>{1, 1, weightsLayout.getBiasSize()[2] * weightsLayout.getBiasSize()[3]},
        std::vector<size_t>{weightsLayout.getBiasStride()[1],
                            weightsLayout.getBiasStride()[1],
                            weightsLayout.getBiasStride()[3]});

    const auto hidden_interim_desc = miopen::TensorDescriptor(
        rnnDesc.dataType,
        std::vector<size_t>{
            1, reservLayout.getGateBlockSizeImpl()[1], reservLayout.getGateBlockSizeImpl()[3]},
        std::vector<size_t>{reservLayout.getGateBlockStride()[0],
                            reservLayout.getGateBlockStride()[1],
                            reservLayout.getGateBlockStride()[3]});

    for(int layer = 0; layer < rnnDesc.nLayers; layer++)
    {
        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            const auto RB_layer_out_off = reservLayout.getGateBlockOffset(layer, 0, seq_dir);

            const auto w_bias_layer_start_off_h =
                weightsLayout.getBiasHidOff(layer, static_cast<int>(seq_dir), 0);

            const auto w_bias_layer_start_off_x =
                weightsLayout.getBiasXinOff(layer, static_cast<int>(seq_dir), 0);

            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     hidden_interim_desc,
                     runtimeArgs.reserveSpace, // A
                     &alpha1,
                     bias_desc,
                     runtimeArgs.w, // B
                     &beta_t,
                     hidden_interim_desc,
                     runtimeArgs.reserveSpace, // C
                     RB_layer_out_off,         // A offset
                     w_bias_layer_start_off_h, // B offset
                     RB_layer_out_off,         // C offset
                     true);

            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     hidden_interim_desc,
                     runtimeArgs.reserveSpace,
                     &alpha1,
                     bias_desc,
                     runtimeArgs.w,
                     &beta_t,
                     hidden_interim_desc,
                     runtimeArgs.reserveSpace,
                     RB_layer_out_off,
                     w_bias_layer_start_off_x,
                     RB_layer_out_off,
                     true);
        }
    }
}

void RNNForwardDataModularAlgo::PropHiddenHt(const Handle& handle,
                                             const runtimeArgsFwd& runtimeArgs,
                                             int layer,
                                             const SequenceIterator& currentSeq,
                                             SequenceDirection direction) const
{
    // iterating over seq in descending order(from high to low)
    // take smallest batch
    const auto gemm_batch_size = batchController.getBatchSize(
        direction == SequenceDirection::Forward ? currentSeq.getPhisVal()
                                                : currentSeq.getNext().getPhisVal());

    if(gemm_batch_size == 0)
        return;

    const auto ht_offset = reservLayout.getHiddenStateOffset(
        layer, batchController.getBatchSum(currentSeq.getPrev().getPhisVal()), direction);

    const auto tmp_block_offset = reservLayout.getGateBlockOffset(
        layer, batchController.getBatchSum(currentSeq.getPhisVal()), direction);

    const auto filter_offset = weightsLayout.getMatrixHidOff(layer, static_cast<int>(direction));

    const miopen::TensorDescriptor& ht_dest_dsc = BuildWsHtDesc2D(gemm_batch_size);

    const miopen::TensorDescriptor tmp_block_src_dsc =
        BuildLstmTmpBlockDesc2D(workspaceInfo, gemm_batch_size);

    const miopen::TensorDescriptor& filter_src_dsc = BuildLstmFilterHidDesc2D();

    RnnBaseFunctions::FWD_GEMM(handle,
                               runtimeArgs.reserveSpace,
                               ht_offset,
                               ht_dest_dsc,
                               runtimeArgs.w,
                               filter_offset,
                               filter_src_dsc,
                               runtimeArgs.reserveSpace,
                               tmp_block_offset,
                               tmp_block_src_dsc,
                               true);
}

void RNNForwardDataModularAlgo::UpdateHStatePerTimeSeq(const Handle& handle,
                                                       const runtimeArgsFwd& runtimeArgs,
                                                       int layer,
                                                       const SequenceIterator& currentSeq,
                                                       SequenceDirection direction) const
{
    size_t seq_batch_offset      = batchController.getBatchSum(currentSeq.getPhisVal());
    size_t seq_batch_offset_prev = batchController.getBatchSum(
        currentSeq.isFirst() ? currentSeq.getPhisVal() : currentSeq.getPrev().getPhisVal());

    LSTMForwardHiddenStateUpdate(
        handle,
        rnnDesc.dataType,
        fwdMode == miopenRNNFWDMode_t::miopenRNNTraining ? false : true,
        currentSeq.isFirst(),
        static_cast<int>(direction),
        batchController.getBatchSize(0),
        batchController.getBatchSize(currentSeq.getPhisVal()),
        batchController.getBatchSize(currentSeq.getPhisVal()),
        rnnDesc.hsize,
        reservLayout.gateStride[1],
        reservLayout.gateSizes[1],
        reservLayout.gateSizes[1],
        runtimeArgs.cx,
        hiddenHxCxInfo.getOffset(layer),
        runtimeArgs.reserveSpace,
        reservLayout.getGasOffset(layer, seq_batch_offset, direction, LstmGateAndState::I),
        reservLayout.getGasOffset(layer, seq_batch_offset, direction, LstmGateAndState::F),
        reservLayout.getGasOffset(layer, seq_batch_offset, direction, LstmGateAndState::O),
        reservLayout.getGasOffset(layer, seq_batch_offset, direction, LstmGateAndState::G),
        reservLayout.getGasOffset(layer, seq_batch_offset, direction, LstmGateAndState::St),
        reservLayout.getGasOffset(layer, seq_batch_offset_prev, direction, LstmGateAndState::St),
        reservLayout.getActiveCellOffset(layer, seq_batch_offset, direction),
        reservLayout.getGasOffset(layer, seq_batch_offset, direction, LstmGateAndState::Ht));
}

void RNNForwardDataModularAlgo::PropHyCy(const Handle& handle,
                                         const runtimeArgsFwd& runtimeArgs,
                                         size_t layer,
                                         const SequenceIterator& currentSeq,
                                         SequenceDirection direction) const
{
    if(runtimeArgs.hy != nullptr || (runtimeArgs.cy != nullptr))
    {
        const auto gap_batch_size = [&]() {
            if(currentSeq.isLast())
            {
                return batchController.getBatchSize(currentSeq.getPhisVal());
            }
            else
            {
                if(direction == SequenceDirection::Forward)
                {
                    return batchController.getBatchSize(currentSeq.getPhisVal()) -
                           batchController.getBatchSize(currentSeq.getNext().getPhisVal());
                }
                else
                    return static_cast<size_t>(0);
            }
        }();

        const auto gap_batch_offset = [&]() {
            if(currentSeq.isLast())
                return static_cast<size_t>(0);
            else
                return batchController.getBatchSize(currentSeq.getPhisVal()) - gap_batch_size;
        }();

        if(gap_batch_size > 0)
        {

            auto src_desc = BuildTempDhtDesc3D(1, gap_batch_size);

            auto dst_desc = BuildHxCxDesc3D(1, gap_batch_size);

            size_t tmp_batch_offset =
                batchController.getBatchSum(currentSeq.getPhisVal()) + gap_batch_offset;

            if(runtimeArgs.hy != nullptr)
            {
                CopyTensor(handle,
                           src_desc,
                           runtimeArgs.reserveSpace,
                           dst_desc,
                           runtimeArgs.hy,
                           reservLayout.getGasOffset(
                               layer, tmp_batch_offset, direction, LstmGateAndState::Ht),
                           hiddenHxCxInfo.getOffset(layer, gap_batch_offset));
            }

            if(runtimeArgs.cy != nullptr)
            {
                CopyTensor(handle,
                           src_desc,
                           runtimeArgs.reserveSpace,
                           dst_desc,
                           runtimeArgs.cy,
                           reservLayout.getGasOffset(
                               layer, tmp_batch_offset, direction, LstmGateAndState::St),
                           hiddenHxCxInfo.getOffset(layer, gap_batch_offset));
            }
        }
    }
}

void RNNForwardDataModularAlgo::PropHiddenY(const Handle& handle,
                                            const runtimeArgsFwd& runtimeArgs,
                                            size_t layer,
                                            SequenceDirection direction) const
{
    if(layer == 0)
        return;

    const auto gemm_batch_size   = batchController.getTotalBatchSum();
    const auto gemm_batch_offset = 0;

    // rnnocl 1512

    const auto tmp_block_offset =
        reservLayout.getGateBlockOffset(layer, gemm_batch_offset, direction);

    const auto tmp_ht_offset =
        reservLayout.getHiddenStateOffset(layer - 1, gemm_batch_offset, direction);

    const miopen::TensorDescriptor tmp_block_src_dsc =
        BuildLstmTmpBlockDesc2D(reservLayout, gemm_batch_size);

    const auto tmp_ht_desc = BuildWsHtDesc2D(gemm_batch_size);

    if(rnnDesc.rnnMode == miopenLSTM)
    {
        const auto filter_offset =
            weightsLayout.getMatrixXinOff(layer, static_cast<int>(direction));
        const auto filter_src_dsc = BuildLstmFilterXDesc2D(layer);

        RnnBaseFunctions::FWD_GEMM(handle,
                                   runtimeArgs.reserveSpace,
                                   tmp_ht_offset,
                                   tmp_ht_desc,

                                   runtimeArgs.w,
                                   filter_offset,
                                   filter_src_dsc,
                                   runtimeArgs.reserveSpace,
                                   tmp_block_offset,
                                   tmp_block_src_dsc,
                                   true);
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Only lstm");
    }
}

void RNNForwardDataModularAlgo::PropY(const Handle& handle, const runtimeArgsFwd& runtimeArgs) const
{
    const auto rnn_data_type = rnnDesc.dataType;
    const auto last_layer_id = rnnDesc.nLayers - 1;

    auto load_offset =
        reservLayout.getHiddenStateOffset(last_layer_id, 0, SequenceDirection::Forward);

    // bwd concat
    // currently supported only one type, but should be more
    auto [y_src_desc,
          y_data_ptr] = [](const IOBufferDescriptor& yInfo, const RNNDescriptor& rnnD, Data_t y) {
        const auto& dy_raw_size   = yInfo.getFullSeqMajorSize();
        const auto& dy_raw_stride = yInfo.getFullSeqMajorStrides();

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

        return std::make_tuple(dy_desc, y);
    }(yInfo, rnnDesc, runtimeArgs.y);

    const std::vector<size_t> tmp_y_strides = [](const auto& full_stride_ref) {
        return std::vector<size_t>(full_stride_ref.begin(), full_stride_ref.end());
    }(reservLayout.getHiddenStateStride());

    const std::vector<size_t> tmp_y_size = [](const auto& full_size_ref) {
        std::vector<size_t> ws_ht_layer_size(full_size_ref.begin(), full_size_ref.end());

        ws_ht_layer_size[0] = 1;

        return ws_ht_layer_size;
    }(reservLayout.hStateSizes);

    auto tmp_y_desc = miopen::TensorDescriptor(rnn_data_type, tmp_y_size, tmp_y_strides);

    if(load_offset <= INT32_MAX)
    {
        CopyTensor(handle,
                   tmp_y_desc,
                   runtimeArgs.reserveSpace,
                   y_src_desc,
                   y_data_ptr,
                   static_cast<int>(load_offset),
                   0,
                   true);
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "store_offset > INT32_MAX");
    }
}

//
//
//

} // namespace rnn_base
} // namespace miopen
