/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/logger.hpp>
#include <miopen/datatype.hpp>
#include <miopen/rnn_util.hpp>
#include <cassert>

namespace miopen {

void LSTMForwardHiddenStateUpdate(Handle& handle,
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
                                  std::size_t hidden_offset)
{
    std::string program_name = "MIOpenRNNHiddenStateUpdate.cl";
    std::string kernel_name  = "LSTMFwdHidUpdate";

    size_t max_active_threads = handle.GetMaxComputeUnits() * handle.GetWavefrontWidth() * 32;

    std::string network_config =
        "lstmfwdhid-" + std::string(rnn_data_type == miopenHalf ? "fp16-" : "fp32-") +
        std::to_string(max_active_threads) + "x" + std::to_string(static_cast<int>(is_inference)) +
        "x" + std::to_string(max_batch) + "x" + std::to_string(hy_h) + "x" +
        std::to_string(hy_stride);

    bool use_cx = cx != nullptr;

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(cx,
               reserve_space,
               static_cast<long long>(cx_offset),
               static_cast<long long>(i_offset),
               static_cast<long long>(f_offset),
               static_cast<long long>(o_offset),
               static_cast<long long>(c_offset),
               static_cast<long long>(cell_offset),
               static_cast<long long>(cell_offset_pre),
               static_cast<long long>(activ_cell_offset),
               static_cast<long long>(hidden_offset),
               static_cast<char>(use_cx),
               static_cast<char>(is_seq_begin),
               direction,
               cur_batch,
               use_batch);
    }
    else
    {
        std::string params = " -DLSTM_FWD_HID=1";

        size_t total_work = max_batch * hy_h;

        size_t RD_BLCK = (total_work >= 4 * max_active_threads && hy_h % 4 == 0)
                             ? 4
                             : ((total_work >= 2 * max_active_threads && hy_h % 2 == 0) ? 2 : 1);
        const std::string data_type = GetDataType(rnn_data_type);
        const std::string READ_TYPE =
            (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK);

        size_t total_item   = std::max(size_t(total_work / RD_BLCK), size_t(1));
        size_t item_per_grp = total_item <= 64 ? 64 : total_item <= 128 ? 128 : 256;
        size_t glb_sz       = total_item < max_active_threads ? total_item : max_active_threads;

        params += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_TYPE=" + READ_TYPE;

        params += " -DHY_H=" + std::to_string(hy_h) + " -DHY_STRIDE=" + std::to_string(hy_stride) +
                  " -DWEI_LEN=" + std::to_string(wei_len) + " -DWEI_STRIDE=" +
                  std::to_string(wei_stride);

        if(rnn_data_type == miopenHalf)
            params += " -DMIOPEN_USE_FP16=1";
        else
            params += " -DMIOPEN_USE_FP32=1";

        if(is_inference)
            params += " -DINFERENCE_MODE=1";

        const std::vector<size_t> vld{item_per_grp, 1, 1};
        const std::vector<size_t> vgd{glb_sz, 1, 1};

        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            cx,
            reserve_space,
            static_cast<long long>(cx_offset),
            static_cast<long long>(i_offset),
            static_cast<long long>(f_offset),
            static_cast<long long>(o_offset),
            static_cast<long long>(c_offset),
            static_cast<long long>(cell_offset),
            static_cast<long long>(cell_offset_pre),
            static_cast<long long>(activ_cell_offset),
            static_cast<long long>(hidden_offset),
            static_cast<char>(use_cx),
            static_cast<char>(is_seq_begin),
            direction,
            cur_batch,
            use_batch);
    }
}

void LSTMBackwardHiddenStateUpdate(Handle& handle,
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
                                   std::size_t f_offset_pre)
{
    std::string program_name = "MIOpenRNNHiddenStateUpdate.cl";
    std::string kernel_name  = "LSTMBwdHidUpdate";

    size_t max_active_threads = handle.GetMaxComputeUnits() * handle.GetWavefrontWidth() * 32;

    std::string network_config =
        "lstmbwdhid-" + std::string(rnn_data_type == miopenHalf ? "fp16-" : "fp32-") +
        std::to_string(max_active_threads) + "x" + std::to_string(max_batch) + "x" +
        std::to_string(hy_h) + "x" + std::to_string(hy_stride);

    bool use_cx  = cx != nullptr;
    bool use_dcy = dcy != nullptr;

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(cx,
               dcy,
               reserve_space,
               work_space,
               static_cast<long long>(cx_offset),
               static_cast<long long>(dcy_offset),
               static_cast<long long>(i_offset),
               static_cast<long long>(f_offset),
               static_cast<long long>(o_offset),
               static_cast<long long>(c_offset),
               static_cast<long long>(activ_cell_offset),
               static_cast<long long>(cell_offset_pre),
               static_cast<long long>(di_offset),
               static_cast<long long>(df_offset),
               static_cast<long long>(do_offset),
               static_cast<long long>(dc_offset),
               static_cast<long long>(dcell_offset),
               static_cast<long long>(dcell_offset_pre),
               static_cast<long long>(dhidden_offset),
               static_cast<long long>(f_offset_pre),
               static_cast<char>(use_cx),
               static_cast<char>(use_dcy),
               static_cast<char>(is_seq_begin),
               static_cast<char>(is_seq_end),
               direction,
               cur_batch,
               use_batch,
               use_batch2);
    }
    else
    {
        std::string params = " -DLSTM_BWD_HID=1";

        size_t total_work = max_batch * hy_h;

        size_t RD_BLCK = (total_work >= 4 * max_active_threads && hy_h % 4 == 0)
                             ? 4
                             : ((total_work >= 2 * max_active_threads && hy_h % 2 == 0) ? 2 : 1);
        const std::string data_type = GetDataType(rnn_data_type);
        const std::string READ_TYPE =
            (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK);

        size_t total_item   = std::max(size_t(total_work / RD_BLCK), size_t(1));
        size_t item_per_grp = total_item <= 64 ? 64 : total_item <= 128 ? 128 : 256;
        size_t glb_sz       = total_item < max_active_threads ? total_item : max_active_threads;

        params += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_TYPE=" + READ_TYPE;

        params += " -DHY_H=" + std::to_string(hy_h) + " -DHY_STRIDE=" + std::to_string(hy_stride) +
                  " -DWEI_LEN=" + std::to_string(wei_len) + " -DWEI_STRIDE=" +
                  std::to_string(wei_stride);

        if(rnn_data_type == miopenHalf)
            params += " -DMIOPEN_USE_FP16=1";
        else
            params += " -DMIOPEN_USE_FP32=1";

        const std::vector<size_t> vld{item_per_grp, 1, 1};
        const std::vector<size_t> vgd{glb_sz, 1, 1};

        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            cx,
            dcy,
            reserve_space,
            work_space,
            static_cast<long long>(cx_offset),
            static_cast<long long>(dcy_offset),
            static_cast<long long>(i_offset),
            static_cast<long long>(f_offset),
            static_cast<long long>(o_offset),
            static_cast<long long>(c_offset),
            static_cast<long long>(activ_cell_offset),
            static_cast<long long>(cell_offset_pre),
            static_cast<long long>(di_offset),
            static_cast<long long>(df_offset),
            static_cast<long long>(do_offset),
            static_cast<long long>(dc_offset),
            static_cast<long long>(dcell_offset),
            static_cast<long long>(dcell_offset_pre),
            static_cast<long long>(dhidden_offset),
            static_cast<long long>(f_offset_pre),
            static_cast<char>(use_cx),
            static_cast<char>(use_dcy),
            static_cast<char>(is_seq_begin),
            static_cast<char>(is_seq_end),
            direction,
            cur_batch,
            use_batch,
            use_batch2);
    }
}
} // namespace miopen
