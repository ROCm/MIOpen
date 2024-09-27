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
#include <miopen/rnn_util.hpp>

namespace miopen {

namespace rnn_base {
miopenStatus_t ReducAddBias(const miopen::Handle& handle,
                            Data_t dw,
                            const Data_t workSpace,
                            const miopen::TensorDescriptor& dw_desc,
                            const miopen::TensorDescriptor& ws_desc,
                            size_t dw_bias_offset,
                            size_t ws_bias_offset,
                            Data_t red_workSpace,
                            size_t red_workSpace_size_bytes)
{
    assert(ws_desc.GetLengths().size() == 2);
    if(ws_desc.GetLengths()[0] != 1)
    {
        int algo = getReductionAlgo();

        switch(algo)
        {
        case 0: {
            float alpha0 = 0;
            float alpha1 = 1;
            float beta_t = 1;

            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     dw_desc,
                     dw,
                     &alpha1,
                     ws_desc,
                     workSpace,
                     &beta_t,
                     dw_desc,
                     dw,
                     dw_bias_offset,
                     ws_bias_offset,
                     dw_bias_offset,
                     true);
        }
        break;
        case 1: {
            float alpha1 = 1;
            float beta1  = 1;

            miopen::ReduceTensorDescriptor red_add{
                miopenReduceTensorOp_t::MIOPEN_REDUCE_TENSOR_ADD,
                miopenDataType_t::miopenFloat,
                miopenNanPropagation_t::MIOPEN_PROPAGATE_NAN,
                miopenReduceTensorIndices_t::MIOPEN_REDUCE_TENSOR_NO_INDICES,
                miopenIndicesType_t::MIOPEN_32BIT_INDICES};

            Data_t srcA_with_offset =
                static_cast<char*>(workSpace) + ws_bias_offset * GetTypeSize(dw_desc.GetType());

            Data_t dstC_with_offset =
                static_cast<char*>(dw) + dw_bias_offset * GetTypeSize(dw_desc.GetType());

            // WA CK bug
            Data_t red_workSpace_bugfix = red_workSpace;
            if(dw_desc.GetType() == miopenDataType_t::miopenHalf)
            {
                if(std::align(4,
                              red_workSpace_size_bytes - 4,
                              red_workSpace_bugfix,
                              red_workSpace_size_bytes) == nullptr)
                    MIOPEN_THROW(miopenStatusInternalError, "failed alignment.");
            }

            red_add.ReduceTensor(handle,
                                 nullptr,
                                 0,
                                 red_workSpace_bugfix,
                                 red_workSpace_size_bytes,
                                 &alpha1,
                                 ws_desc,
                                 srcA_with_offset,
                                 &beta1,
                                 dw_desc,
                                 dstC_with_offset);
        }
        break;

        default: break;
        }
    }
    else
    {
        // nothing to reduce
        // just copy data from workspace to dw
        CopyTensor(handle, ws_desc, workSpace, dw_desc, dw, ws_bias_offset, dw_bias_offset);
    }

    return miopenStatusSuccess;
}

void RNNBackwardWeightsModularAlgo::PrepareWriteBuffers(const Handle& handle, Data_t w) const
{
    const auto rnn_data_type = rnnDesc.dataType;

    const auto w_tensor_size =
        rnnDesc.GetParamsSize(xInfo.getHiddenSize()) / GetTypeSize(rnn_data_type);

    const auto w_desc =
        miopen::TensorDescriptor(rnn_data_type, {1, w_tensor_size}, {w_tensor_size, 1});

    const float beta = 0.;

    SetTensor(handle, w_desc, w, &beta);
}

void RNNBackwardWeightsModularAlgo::PhisXInputWeights(const Handle& handle,
                                                      Data_t dw,
                                                      Data_t workSpace,
                                                      ConstData_t x) const
{
    const size_t gemm_batch_size = xInfo.getFullSeqMajorSize()[0];

    assert(gemm_batch_size != 0);

    if(rnnDesc.inputMode == miopenRNNlinear)
    {
        constexpr int layer                   = 0;
        constexpr SequenceDirection direction = SequenceDirection::Forward;
        constexpr size_t gemm_batch_offset    = 0;

        // both directions in 1 call;

        const auto tmp_block_offset =
            workspaceInfo.getGateBlockOffset(layer, gemm_batch_offset, direction);

        const auto filter_offset =
            weightsLayout.getMatrixXinOff(layer, static_cast<int>(direction));

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

        RnnBaseFunctions::BWWei_GEMM(handle,
                                     workSpace,
                                     tmp_block_offset,
                                     tmp_block_src_dsc,
                                     x,
                                     ht_x_offset,
                                     ht_x_desc,
                                     dw,
                                     filter_offset,
                                     filter_src_dsc,
                                     true);
    }
}

void RNNBackwardWeightsModularAlgo::HiddenXInputWeights(const Handle& handle,
                                                        Data_t dw,
                                                        ConstData_t workSpace,
                                                        ConstData_t reserveSpace,
                                                        size_t layer) const
{
    const size_t gemm_batch_size = workspaceInfo.getGateBlockSize()[1];

    const size_t gemm_batch_offset    = 0;
    const size_t seq_start            = 0;
    const SequenceDirection direction = SequenceDirection::Forward;

    assert(gemm_batch_size != 0);
    assert(layer > 0);

    [[maybe_unused]] bool use_dropout = !float_equal(miopen::deref(rnnDesc.dropoutDesc).dropout, 0);

    assert(use_dropout == false);

    // both directions in 1 call;

    const auto tmp_block_offset =
        workspaceInfo.getGateBlockOffset(layer, gemm_batch_offset, direction);

    const auto filter_offset = weightsLayout.getMatrixXinOff(layer, static_cast<int>(direction));

    const auto ht_offset = reservLayout.getHiddenStateOffset(
        layer - 1, batchController.getBatchSum(seq_start), direction);

    const auto tmp_block_src_dsc = BuildLstmTmpBlockDesc2D(workspaceInfo, gemm_batch_size);

    const auto filter_src_dsc = BuildLstmFilterXDesc2D(layer);
    // TODO chage for dropout
    const auto ht_desc = BuildTmpHtDesc2D(reservLayout, gemm_batch_size);

    RnnBaseFunctions::BWWei_GEMM(handle,
                                 workSpace,
                                 tmp_block_offset,
                                 tmp_block_src_dsc,
                                 reserveSpace,
                                 ht_offset,
                                 ht_desc,
                                 dw,
                                 filter_offset,
                                 filter_src_dsc,
                                 true);
}

void RNNBackwardWeightsModularAlgo::BiasUpdate(
    const Handle& handle, Data_t dw, Data_t workSpace, size_t layer, size_t workSpaceSize) const
{
    if(rnnDesc.biasMode != 0u)
    {
        const auto batch_size = batchController.getTotalBatchSum();

        const TensorDescriptor block_dsc = BuildLstmTmpBlockDesc2D(workspaceInfo, batch_size);

        const miopen::TensorDescriptor dw_desc = BuildWeiBiasDesc2D();

        size_t main_ws_size = workspaceInfo.getBufferSize() * GetTypeSize(rnnDesc.dataType);

        size_t reduction_ws_size = workSpaceSize - main_ws_size;

        Data_t reduction_workSpace = static_cast<char*>(workSpace) + main_ws_size;
        size_t dw_bias_offset =
            weightsLayout.getBiasXinOff(layer, static_cast<int>(SequenceDirection::Forward), 0);

        ReducAddBias(handle,
                     dw,
                     workSpace,
                     dw_desc,
                     block_dsc,
                     dw_bias_offset,
                     workspaceInfo.getGateBlockOffset(layer, 0, SequenceDirection::Forward),
                     reduction_workSpace,
                     reduction_ws_size);

        // second dw bias equal to the first, so just copy reduction result
        size_t dw_bias_2_offset =
            weightsLayout.getBiasHidOff(layer, static_cast<int>(SequenceDirection::Forward), 0);
        CopyTensor(handle, dw_desc, dw, dw_desc, dw, dw_bias_offset, dw_bias_2_offset);
    }
}

void RNNBackwardWeightsModularAlgo::HiddenHStateWeights_Unchecked(const Handle& handle,
                                                                  Data_t dw,
                                                                  ConstData_t workSpace,
                                                                  ConstData_t reserveSpace,
                                                                  const SequenceIterator& seq,
                                                                  size_t layer,
                                                                  SequenceDirection direction,
                                                                  size_t gemm_batch_size) const
{
    if(gemm_batch_size == 0)
        return;

    const size_t blk_batch_shift = batchController.getBatchSum(seq.getPrev().getPhisVal());

    const size_t ht_batch_shift = batchController.getBatchSum(seq.getPhisVal());

    const auto block_offset  = workspaceInfo.getGateBlockOffset(layer, blk_batch_shift, direction);
    const auto ht_offset     = reservLayout.getHiddenStateOffset(layer, ht_batch_shift, direction);
    const auto filter_offset = weightsLayout.getMatrixHidOff(layer, static_cast<int>(direction));

    const TensorDescriptor block_dsc  = BuildLstmTmpBlockDesc2D(workspaceInfo, gemm_batch_size);
    const TensorDescriptor ht_desc    = BuildTmpHtDesc2D(reservLayout, gemm_batch_size);
    const TensorDescriptor filter_dsc = BuildLstmFilterHidDesc2D();

    RnnBaseFunctions::BWWei_GEMM(handle,
                                 workSpace,
                                 block_offset,
                                 block_dsc,
                                 reserveSpace,
                                 ht_offset,
                                 ht_desc,
                                 dw,
                                 filter_offset,
                                 filter_dsc,
                                 true);
}

void RNNBackwardWeightsModularAlgo::PhisHStateWeights(const Handle& handle,
                                                      Data_t dw,
                                                      ConstData_t workSpace,
                                                      ConstData_t hx,
                                                      const SequenceIterator& seq,
                                                      size_t layer,
                                                      SequenceDirection direction) const
{
    const size_t gemm_batch_size = getHxBatchSizeReadAtTime(seq, direction);

    if(gemm_batch_size == 0 || hx == nullptr)
        return;

    const size_t batch_shift = batchController.getBatchSum(seq.getPhisVal()) +
                               (batchController.getBatchSize(seq.getPhisVal()) - gemm_batch_size);

    const auto virt_layer = getVirtualLayer(layer, direction);

    const size_t block_offset  = workspaceInfo.getGateBlockOffset(layer, batch_shift, direction);
    const size_t hx_offset     = hiddenHxCxInfo.getOffset(virt_layer, batch_shift);
    const size_t filter_offset = weightsLayout.getMatrixHidOff(layer, static_cast<int>(direction));

    const TensorDescriptor block_dsc  = BuildLstmTmpBlockDesc2D(workspaceInfo, gemm_batch_size);
    const TensorDescriptor hx_desc    = BuildHxCxDesc2D(gemm_batch_size);
    const TensorDescriptor filter_dsc = BuildLstmFilterHidDesc2D();

    RnnBaseFunctions::BWWei_GEMM(handle,
                                 workSpace,
                                 block_offset,
                                 block_dsc,
                                 hx,
                                 hx_offset,
                                 hx_desc,
                                 dw,
                                 filter_offset,
                                 filter_dsc,
                                 true);
}

} // namespace rnn_base
} // namespace miopen
