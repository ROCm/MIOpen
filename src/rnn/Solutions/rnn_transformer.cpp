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

#include <miopen/handle.hpp>
#include <miopen/rnn.hpp>
#include <miopen/rnn_util.hpp>

#include <cassert>
#include <cstddef>
#include <numeric>
#include <ostream>

namespace miopen {

namespace {

size_t RNNLayoutTransformTotalTmpSpace(miopenRNNMode_t rnnMode,
                                       miopenRNNFWDMode_t fwdMode,
                                       size_t batch,
                                       size_t seq_len,
                                       size_t in_vec,
                                       size_t out_vec,
                                       size_t h_vec,
                                       size_t c_vec,
                                       size_t num_layers)
{
    const auto xin_tmp_size  = batch * seq_len * in_vec;
    const auto yout_tmp_size = batch * seq_len * out_vec;

    const auto transform_workspace = 2 * std::max(xin_tmp_size, yout_tmp_size);

    const auto cnt_mult = fwdMode == miopenRNNTraining ? 3 : 2;

    const auto h_tmp_size = cnt_mult * batch * h_vec * num_layers;
    const auto c_tmp_size = cnt_mult * batch * c_vec * num_layers;

    switch(rnnMode)
    {
    case miopenRNNTANH:
    case miopenRNNRELU:
    case miopenGRU:
        return std::max((h_tmp_size), transform_workspace) + xin_tmp_size + yout_tmp_size;
    case miopenLSTM:
        return std::max((h_tmp_size + c_tmp_size), transform_workspace) + xin_tmp_size +
               yout_tmp_size;
    default: MIOPEN_THROW(miopenStatusInternalError, "unknown rnnDesc.rnnMode");
    }
}

size_t RNNLayoutTransformTotalTmpSpace(miopenRNNMode_t rnnMode,
                                       miopenRNNFWDMode_t fwdMode,
                                       const SeqTensorDescriptor& xDesc,
                                       const SeqTensorDescriptor& yDesc,
                                       const TensorDescriptor& hDesc,
                                       const TensorDescriptor& cDesc)
{
    if(!xDesc.IsPaddedSeqLayout())
        MIOPEN_THROW(miopenStatusInternalError, "wrong SeqLayout ");

    const auto [batch, seq_len, in_vec] = miopen::tien<3>(xDesc.GetLengths());
    const auto out_vec                  = yDesc.GetLengths()[2];
    const auto c_size                   = cDesc.GetLengths()[2];
    const auto h_size                   = hDesc.GetLengths()[2];
    const auto num_layers               = hDesc.GetLengths()[0];

    return RNNLayoutTransformTotalTmpSpace(
        rnnMode, fwdMode, batch, seq_len, in_vec, out_vec, h_size, c_size, num_layers);
}

inline Data_t PostIncBytePtr(Data_t& tmp_prt, size_t inc_size)
{
    auto ret_ptr = tmp_prt;
    tmp_prt      = static_cast<void*>(reinterpret_cast<char*>(tmp_prt) + inc_size);
    return ret_ptr;
}

} // namespace

size_t RNNDescriptor::RNNTransformerWorkspaceSize(const SeqTensorDescriptor& xDesc,
                                                  miopenRNNFWDMode_t fwdMode) const
{
    if(!xDesc.IsPaddedSeqLayout())
        MIOPEN_THROW(miopenStatusInternalError, "wrong SeqLayout ");
    if(!xDesc.IsZeroBytePadding())
        MIOPEN_THROW(miopenStatusInternalError, "wrong BytePadding ");

    auto [batch, seq_len, in_vec] = miopen::tien<3>(xDesc.GetLengths());
    return RNNLayoutTransformTotalTmpSpace(rnnMode,
                                           fwdMode,
                                           batch,
                                           seq_len,
                                           in_vec,
                                           hsize * (dirMode == miopenRNNunidirection ? 1 : 2),
                                           hsize,
                                           hsize,
                                           nLayers * (dirMode == miopenRNNunidirection ? 1 : 2)) *
           typeSize;
}

void RNNDescriptor::RNNTransformerForward(const Handle& handle,
                                          miopenRNNFWDMode_t fwdMode,
                                          ConstData_t w,
                                          const SeqTensorDescriptor& xDesc,
                                          ConstData_t x,
                                          const TensorDescriptor& hDesc,
                                          ConstData_t hx,
                                          Data_t hy,
                                          const TensorDescriptor& cDesc,
                                          ConstData_t cx,
                                          Data_t cy,
                                          const SeqTensorDescriptor& yDesc,
                                          Data_t y,
                                          Data_t workSpace,
                                          size_t workSpaceSize,
                                          Data_t reserveSpace,
                                          size_t reserveSpaceSize) const
{

    if(workSpaceSize < GetMaxWorkspaceSize(handle, xDesc, fwdMode))
    {
        MIOPEN_THROW("Workspace is required");
    }

    if(xDesc.GetType() != hDesc.GetType() || hDesc.GetType() != cDesc.GetType() ||
       cDesc.GetType() != yDesc.GetType())
    {
        MIOPEN_THROW("Error unsupported config");
    }

    const auto xDesc_base_layout = RNNDescriptor::getBaseLayoutFromDataTensor(xDesc);

    if(xDesc_base_layout == miopenRNNDataSeqMajorNotPadded)
    {
        RNNVanillaForward(handle,
                          fwdMode,
                          w,
                          xDesc,
                          x,
                          hDesc,
                          hx,
                          hy,
                          cDesc,
                          cx,
                          cy,
                          yDesc,
                          y,
                          workSpace,
                          workSpaceSize,
                          reserveSpace,
                          reserveSpaceSize);
    }
    else
    {
        auto dataTypeSize                    = GetTypeSize(xDesc.GetType());
        const std::vector<size_t> sorted_seq = RNNTensorBaseLayoutConverter::GetSortedLens(xDesc);
        const auto [layout_dims_order, layout_seq_padding] =
            convertRNNBaseLayout(miopenRNNDataSeqMajorNotPadded);

        const SeqTensorDescriptor xDesc_packed_SeqMajor(xDesc.GetType(),
                                                        layout_dims_order,
                                                        xDesc.GetLengths(),
                                                        sorted_seq,
                                                        {},
                                                        true,
                                                        layout_seq_padding);

        const SeqTensorDescriptor yDesc_packed_SeqMajor(yDesc.GetType(),
                                                        layout_dims_order,
                                                        yDesc.GetLengths(),
                                                        sorted_seq,
                                                        {},
                                                        true,
                                                        layout_seq_padding);

        size_t packedXInSize  = xDesc_packed_SeqMajor.GetTensorMaxByteSpace(),
               packedYOutSize = yDesc_packed_SeqMajor.GetTensorMaxByteSpace();

        Data_t tmp_space = workSpace;

        Data_t packedXIn_data  = PostIncBytePtr(tmp_space, packedXInSize);
        Data_t packedYOut_data = PostIncBytePtr(tmp_space, packedYOutSize);

        Data_t converter_workSpace = tmp_space;
        RNNTensorBaseLayoutConverter::ConvertInputTensorGPUData(
            handle, xDesc, x, xDesc_packed_SeqMajor, packedXIn_data, converter_workSpace, false);

        {
            Data_t tmp_layout_hx =
                PostIncBytePtr(tmp_space, hDesc.GetElementSpace() * dataTypeSize);
            Data_t tmp_layout_hy =
                PostIncBytePtr(tmp_space, hDesc.GetElementSpace() * dataTypeSize);
            Data_t tmp_layout_cx =
                PostIncBytePtr(tmp_space, cDesc.GetElementSpace() * dataTypeSize);
            Data_t tmp_layout_cy =
                PostIncBytePtr(tmp_space, cDesc.GetElementSpace() * dataTypeSize);

            const std::vector<size_t> input_reorder_index =
                RNNTensorBaseLayoutConverter::GetSamplesDescendingOrder(xDesc);
            if(hx != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, input_reorder_index, hx, tmp_layout_hx);
            }
            if(cx != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, input_reorder_index, cx, tmp_layout_cx);
            }

            auto workSpace_shift_size =
                dataTypeSize *
                RNNLayoutTransformTotalTmpSpace(rnnMode, fwdMode, xDesc, yDesc, hDesc, cDesc);

            auto workSpace_fwd = workSpace;
            PostIncBytePtr(workSpace_fwd, workSpace_shift_size);
            auto workSpace_fwd_size = workSpaceSize - workSpace_shift_size;

            RNNDescriptor packedRnnDesc(*this);
            packedRnnDesc.SetPaddingmode(miopenRNNIONotPadded);
            packedRnnDesc.RNNVanillaForward(handle,
                                            fwdMode,
                                            w,
                                            xDesc_packed_SeqMajor,
                                            packedXIn_data,
                                            hDesc,
                                            hx != nullptr ? tmp_layout_hx : nullptr,
                                            hy != nullptr ? tmp_layout_hy : nullptr,
                                            cDesc,
                                            cx != nullptr ? tmp_layout_cx : nullptr,
                                            cy != nullptr ? tmp_layout_cy : nullptr,
                                            yDesc_packed_SeqMajor,
                                            packedYOut_data,
                                            workSpace_fwd,
                                            workSpace_fwd_size,

                                            reserveSpace,
                                            reserveSpaceSize);

            const std::vector<size_t> output_reorder_index =
                RNNTensorBaseLayoutConverter::GetSamplesDescendingOrder(xDesc, true);
            if(hy != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, output_reorder_index, tmp_layout_hy, hy);
            }
            if(cy != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, cDesc, 1, output_reorder_index, tmp_layout_cy, cy);
            }
        }

        RNNTensorBaseLayoutConverter::ReverseConvertInputTensorGPUData(
            handle, yDesc_packed_SeqMajor, packedYOut_data, yDesc, y, converter_workSpace);
    }
}

void RNNDescriptor::RNNTransformerBackwardData(const Handle& handle,
                                               const SeqTensorDescriptor& yDesc,
                                               ConstData_t dy,
                                               const TensorDescriptor& hDesc,
                                               ConstData_t hx,
                                               ConstData_t dhy,
                                               Data_t dhx,
                                               const TensorDescriptor& cDesc,
                                               ConstData_t cx,
                                               ConstData_t dcy,
                                               Data_t dcx,
                                               const SeqTensorDescriptor& xDesc,
                                               Data_t dx,
                                               ConstData_t w,
                                               Data_t workSpace,
                                               size_t workSpaceSize,
                                               Data_t reserveSpace,
                                               size_t reserveSpaceSize) const
{
    if(workSpaceSize < GetMaxWorkspaceSize(handle, xDesc, miopenRNNFWDMode_t::miopenRNNTraining))
    {
        MIOPEN_THROW("Workspace is required");
    }

    const auto xDesc_base_layout = RNNDescriptor::getBaseLayoutFromDataTensor(xDesc);

    if(xDesc_base_layout == miopenRNNDataSeqMajorNotPadded)
    {
        RNNVanillaBackwardData(handle,
                               yDesc,
                               dy,
                               hDesc,
                               hx,
                               dhy,
                               dhx,
                               cDesc,
                               cx,
                               dcy,
                               dcx,
                               xDesc,
                               dx,
                               w,
                               workSpace,
                               workSpaceSize,
                               reserveSpace,
                               reserveSpaceSize);
    }
    else
    {
        auto dataTypeSize                    = GetTypeSize(xDesc.GetType());
        const std::vector<size_t> sorted_seq = RNNTensorBaseLayoutConverter::GetSortedLens(xDesc);
        const auto [layout_dims_order, layout_seq_padding] =
            convertRNNBaseLayout(miopenRNNDataSeqMajorNotPadded);

        const SeqTensorDescriptor xDesc_packed_SeqMajor(xDesc.GetType(),
                                                        layout_dims_order,
                                                        xDesc.GetLengths(),
                                                        sorted_seq,
                                                        {},
                                                        true,
                                                        layout_seq_padding);
        const SeqTensorDescriptor yDesc_packed_SeqMajor(yDesc.GetType(),
                                                        layout_dims_order,
                                                        yDesc.GetLengths(),
                                                        sorted_seq,
                                                        {},
                                                        true,
                                                        layout_seq_padding);

        size_t packedDXInSize  = xDesc_packed_SeqMajor.GetTensorMaxByteSpace(),
               packedDYOutSize = yDesc_packed_SeqMajor.GetTensorMaxByteSpace();

        Data_t tmp_space = workSpace;

        const Data_t packedDYOut_data = PostIncBytePtr(tmp_space, packedDYOutSize);
        const Data_t packedDXIn_data  = PostIncBytePtr(tmp_space, packedDXInSize);

        Data_t converter_workSpace = tmp_space;
        RNNTensorBaseLayoutConverter::ConvertInputTensorGPUData(
            handle, yDesc, dy, yDesc_packed_SeqMajor, packedDYOut_data, converter_workSpace, false);

        {
            const Data_t tmp_layout_hx =
                PostIncBytePtr(tmp_space, hDesc.GetElementSpace() * dataTypeSize);
            const Data_t tmp_layout_dhy =
                PostIncBytePtr(tmp_space, hDesc.GetElementSpace() * dataTypeSize);
            const Data_t tmp_layout_dhx =
                PostIncBytePtr(tmp_space, hDesc.GetElementSpace() * dataTypeSize);
            const Data_t tmp_layout_cx =
                PostIncBytePtr(tmp_space, cDesc.GetElementSpace() * dataTypeSize);
            const Data_t tmp_layout_dcy =
                PostIncBytePtr(tmp_space, cDesc.GetElementSpace() * dataTypeSize);
            const Data_t tmp_layout_dcx =
                PostIncBytePtr(tmp_space, cDesc.GetElementSpace() * dataTypeSize);

            const std::vector<size_t> input_reorder_index =
                RNNTensorBaseLayoutConverter::GetSamplesDescendingOrder(xDesc);
            if(hx != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, input_reorder_index, hx, tmp_layout_hx);
            }
            if(dhy != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, input_reorder_index, dhy, tmp_layout_dhy);
            }
            if(cx != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, input_reorder_index, cx, tmp_layout_cx);
            }
            if(dcy != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, input_reorder_index, dcy, tmp_layout_dcy);
            }

            auto workSpace_shift_size =
                dataTypeSize * RNNLayoutTransformTotalTmpSpace(
                                   rnnMode, miopenRNNTraining, xDesc, yDesc, hDesc, cDesc);

            auto workSpace_bwd = workSpace;
            PostIncBytePtr(workSpace_bwd, workSpace_shift_size);

            auto workSpace_bwd_size = workSpaceSize - workSpace_shift_size;

            RNNDescriptor packedRnnDesc(*this);
            packedRnnDesc.SetPaddingmode(miopenRNNIONotPadded);
            packedRnnDesc.RNNVanillaBackwardData(handle,
                                                 yDesc_packed_SeqMajor,
                                                 packedDYOut_data,
                                                 hDesc,
                                                 hx != nullptr ? tmp_layout_hx : nullptr,
                                                 dhy != nullptr ? tmp_layout_dhy : nullptr,
                                                 dhx != nullptr ? tmp_layout_dhx : nullptr,
                                                 cDesc,
                                                 cx != nullptr ? tmp_layout_cx : nullptr,
                                                 dcy != nullptr ? tmp_layout_dcy : nullptr,
                                                 dcx != nullptr ? tmp_layout_dcx : nullptr,
                                                 xDesc_packed_SeqMajor,
                                                 packedDXIn_data,
                                                 w,
                                                 workSpace_bwd,
                                                 workSpace_bwd_size,
                                                 reserveSpace,
                                                 reserveSpaceSize);

            const std::vector<size_t> output_reorder_index =
                RNNTensorBaseLayoutConverter::GetSamplesDescendingOrder(xDesc, true);

            if(dhx != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, output_reorder_index, tmp_layout_dhx, dhx);
            }
            if(dcx != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, cDesc, 1, output_reorder_index, tmp_layout_dcx, dcx);
            }
        }

        RNNTensorBaseLayoutConverter::ReverseConvertInputTensorGPUData(
            handle, xDesc_packed_SeqMajor, packedDXIn_data, xDesc, dx, converter_workSpace);
    }
}

void RNNDescriptor::RNNTransformerBackwardWeights(const Handle& handle,
                                                  const SeqTensorDescriptor& xDesc,
                                                  ConstData_t x,
                                                  const TensorDescriptor& hDesc,
                                                  ConstData_t hx,
                                                  const SeqTensorDescriptor& yDesc,
                                                  Data_t dw,
                                                  Data_t workSpace,
                                                  size_t workSpaceSize,
                                                  ConstData_t reserveSpace,
                                                  size_t reserveSpaceSize) const
{
    if(workSpaceSize < GetMaxWorkspaceSize(handle, xDesc, miopenRNNFWDMode_t::miopenRNNTraining))
    {
        MIOPEN_THROW("Workspace is required");
    }

    const auto xDesc_base_layout = RNNDescriptor::getBaseLayoutFromDataTensor(xDesc);

    if(xDesc_base_layout == miopenRNNDataSeqMajorNotPadded)
    {
        RNNVanillaBackwardWeights(handle,
                                  xDesc,
                                  x,
                                  hDesc,
                                  hx,
                                  yDesc,
                                  dw,
                                  workSpace,
                                  workSpaceSize,
                                  reserveSpace,
                                  reserveSpaceSize);
    }
    else
    {
        auto dataTypeSize                    = GetTypeSize(xDesc.GetType());
        const std::vector<size_t> sorted_seq = RNNTensorBaseLayoutConverter::GetSortedLens(xDesc);
        const auto [layout_dims_order, layout_seq_padding] =
            convertRNNBaseLayout(miopenRNNDataSeqMajorNotPadded);

        const SeqTensorDescriptor xDesc_packed_SeqMajor(xDesc.GetType(),
                                                        layout_dims_order,
                                                        xDesc.GetLengths(),
                                                        sorted_seq,
                                                        {},
                                                        true,
                                                        layout_seq_padding);
        const SeqTensorDescriptor yDesc_packed_SeqMajor(yDesc.GetType(),
                                                        layout_dims_order,
                                                        yDesc.GetLengths(),
                                                        sorted_seq,
                                                        {},
                                                        true,
                                                        layout_seq_padding);

        size_t packedXInSize = xDesc_packed_SeqMajor.GetTensorMaxByteSpace();

        Data_t tmp_space = workSpace;

        const Data_t packedXIn_data = PostIncBytePtr(tmp_space, packedXInSize);

        Data_t converter_workSpace = tmp_space;
        RNNTensorBaseLayoutConverter::ConvertInputTensorGPUData(
            handle, xDesc, x, xDesc_packed_SeqMajor, packedXIn_data, converter_workSpace, false);

        {
            const Data_t tmp_layout_hx =
                PostIncBytePtr(tmp_space, hDesc.GetElementSpace() * dataTypeSize);

            const std::vector<size_t> input_reorder_index =
                RNNTensorBaseLayoutConverter::GetSamplesDescendingOrder(xDesc);

            if(hx != nullptr)
            {
                RNNTensorBaseLayoutConverter::ReorderHiddenTensorGPUData(
                    handle, hDesc, 1, input_reorder_index, hx, tmp_layout_hx);
            }

            auto workSpace_shift_size =
                dataTypeSize * RNNLayoutTransformTotalTmpSpace(
                                   rnnMode, miopenRNNTraining, xDesc, yDesc, hDesc, hDesc);

            {
                auto shifted_workSpace = workSpace;
                PostIncBytePtr(shifted_workSpace, workSpace_shift_size);

                auto shifted_workSpace_size = workSpaceSize - workSpace_shift_size;

                RNNDescriptor packedRnnDesc(*this);
                packedRnnDesc.SetPaddingmode(miopenRNNIONotPadded);
                packedRnnDesc.RNNVanillaBackwardWeights(handle,
                                                        xDesc_packed_SeqMajor,
                                                        packedXIn_data,
                                                        hDesc,
                                                        hx != nullptr ? tmp_layout_hx : nullptr,
                                                        yDesc_packed_SeqMajor,
                                                        dw,
                                                        shifted_workSpace,
                                                        shifted_workSpace_size,
                                                        reserveSpace,
                                                        reserveSpaceSize);
            }
        }
    }
}

} // namespace miopen
