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
#include <miopen/reducetensor.hpp>

namespace miopen {

struct RnnHipAutoProfiler
{
    RnnHipAutoProfiler(const Handle& handle) : is_profiling_active(handle.IsProfilingEnabled())
    {
        if(is_profiling_active)
        {
#if MIOPEN_BACKEND_HIP
            attached_handle = &handle;
#endif
            RNNProfilingBegin();
        }
    }
    ~RnnHipAutoProfiler()
    {
        if(is_profiling_active)
        {
            RNNProfilingEnd();
        }
    }

    void abortProfiling()
    {
        if(is_profiling_active)
        {
#if MIOPEN_BACKEND_HIP
            attached_handle->EnableProfiling(true);
#endif
            is_profiling_active = false;
        }
    }

private:
    void RNNProfilingBegin()
    {
#if MIOPEN_BACKEND_HIP
        attached_handle->EnableProfiling(false);
        start = miopen::make_hip_event();
        stop  = miopen::make_hip_event();
        hipEventRecord(start.get(), attached_handle->GetStream());
#endif
    }

    void RNNProfilingEnd()
    {
#if MIOPEN_BACKEND_HIP
        hipEventRecord(stop.get(), attached_handle->GetStream());
        hipEventSynchronize(stop.get());
        float eventTime_mS = 0;
        hipEventElapsedTime(&eventTime_mS, start.get(), stop.get());

        attached_handle->EnableProfiling(true);
        attached_handle->ResetKernelTime();
        attached_handle->AccumKernelTime(eventTime_mS);
#endif
    }

#if MIOPEN_BACKEND_HIP
    const Handle* attached_handle = nullptr;
    HipEventPtr start             = nullptr;
    HipEventPtr stop              = nullptr;
#endif
    bool is_profiling_active = false;
};

#if MIOPEN_BACKEND_HIP
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
            0ULL,
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
            0ULL,
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

int getReductionAlgo();

inline size_t ReductionWorkspaceSize(const Handle& handle,
                                     size_t batchLenSum,
                                     size_t nHiddenTensorsPerLayer,
                                     size_t workspaceScale,
                                     size_t hsize,
                                     bool is_bidirect,
                                     miopenDataType_t rnn_data_t)
{
    int red_algo = getReductionAlgo();

    size_t reduction_ws = 0;

    // nothing to reduce,
    if(batchLenSum == 1)
        return 0;

    if(red_algo == 1)
    {
        miopen::ReduceTensorDescriptor red_add{
            miopenReduceTensorOp_t::MIOPEN_REDUCE_TENSOR_ADD,
            miopenDataType_t::miopenFloat, // compute in float for fp16
            miopenNanPropagation_t::MIOPEN_PROPAGATE_NAN,
            miopenReduceTensorIndices_t::MIOPEN_REDUCE_TENSOR_NO_INDICES,
            miopenIndicesType_t::MIOPEN_32BIT_INDICES};

        int bidirect_mp = is_bidirect ? 2 : 1;

        size_t hy_stride = hsize * bidirect_mp * workspaceScale;

        size_t bias_total_cnt = hsize * bidirect_mp * nHiddenTensorsPerLayer;

        const std::vector<size_t> ws_bias_strides{
            batchLenSum * workspaceScale * hsize * bidirect_mp, hy_stride, 1};

        const miopen::TensorDescriptor ws_desc{
            rnn_data_t, {1, batchLenSum, bias_total_cnt}, ws_bias_strides};

        const std::vector<size_t> dw_bias_strides{bias_total_cnt, bias_total_cnt, 1};
        const miopen::TensorDescriptor dw_desc{rnn_data_t, {1, 1, bias_total_cnt}, dw_bias_strides};

        reduction_ws = red_add.GetWorkspaceSize(handle, ws_desc, dw_desc) + // WA CK bug
                       (rnn_data_t == miopenDataType_t::miopenHalf ? 4 : 0);
    }
    else
    {
        if(red_algo == 2 || red_algo == 3)
        {
            reduction_ws = batchLenSum * GetTypeSize(rnn_data_t);
        }
    }
    return reduction_ws;
}

} // namespace miopen

#endif // GUARD_MIOPEN_RNN_UTIL_HPP_
