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
#pragma once

#include "InputFlags.hpp"
#include "driver.hpp"
#include "gru_verify_gemm.hpp"
#include "lstm_verify_gemm.hpp"
#include "random.hpp"
#include "rnn_verify_gemm.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include "util_file.hpp"

#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/rnn.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

std::vector<size_t> get_default_time_strides(int vec_size, const std::vector<int>& seq_array)
{
    std::vector<size_t> sum_v(seq_array.size());
    sum_v[0] = 0;
    std::partial_sum(seq_array.begin(), std::prev(seq_array.end()), std::next(sum_v.begin()));
    for(auto& it : sum_v)
        it *= vec_size;
    return sum_v;
};

template <typename Tgpu>
void TransformIODefaultLayaoutToTarget(std::vector<Tgpu>& def_array,
                                       std::vector<Tgpu>& target_array,
                                       const std::vector<int>& def_batch_sz_per_time,
                                       const std::vector<int>& target_batch_order,
                                       size_t seq_len_padded,
                                       size_t io_vec_size,
                                       miopenRNNBaseLayout_t target_layout,
                                       bool is_dst_target)
{
    assert(target_layout == miopenRNNDataBatchMajorPadded ||
           target_layout == miopenRNNDataSeqMajorPadded);

    assert(def_batch_sz_per_time[0] == target_batch_order.size());

    size_t non_zero_seq_len = def_batch_sz_per_time.size();
    size_t batch_size       = def_batch_sz_per_time[0];

    const std::vector<size_t> default_time_strides =
        get_default_time_strides(io_vec_size, def_batch_sz_per_time);
    const size_t default_batch_stride = io_vec_size;

    size_t target_batch_stride =
        io_vec_size * (target_layout == miopenRNNDataBatchMajorPadded ? seq_len_padded : 1);
    size_t target_time_stride =
        io_vec_size * (target_layout == miopenRNNDataBatchMajorPadded ? 1 : batch_size);

    for(size_t time_it = 0; time_it < non_zero_seq_len; time_it++)
    {
        for(size_t batch_it = 0; batch_it < def_batch_sz_per_time[time_it]; batch_it++)
        {
            const size_t def_time_offset = default_time_strides[time_it];
            const size_t target_time_off = time_it * target_time_stride;

            const size_t def_offset = def_time_offset + default_batch_stride * batch_it;
            const size_t target_offset =
                target_time_off + target_batch_stride * target_batch_order[batch_it];

            if(is_dst_target)
            {
                std::copy(&def_array[def_offset],
                          &def_array[def_offset + io_vec_size],
                          &target_array[target_offset]);
            }
            else
            {
                std::copy(&target_array[target_offset],
                          &target_array[target_offset + io_vec_size],
                          &def_array[def_offset]);
            }
        }
    }
}

template <typename Tgpu>
void HiddenTensorReorder(std::vector<Tgpu>& src_array,
                         std::vector<Tgpu>& dst_array,
                         const std::vector<int>& batch_order,
                         const std::vector<int> hid_len,
                         bool is_dst_direct_order)
{
    const size_t copy_size = hid_len[2];

    const size_t batch_stride = hid_len[2];
    const size_t layer_stride = batch_stride * hid_len[1];

    for(size_t batch_id = 0; batch_id < hid_len[1]; batch_id++)
    {
        const auto src_batch_off =
            batch_stride * (is_dst_direct_order ? batch_order[batch_id] : batch_id);
        const auto dst_batch_off =
            batch_stride * (is_dst_direct_order ? batch_id : batch_order[batch_id]);

        for(size_t layer_id = 0; layer_id < hid_len[0]; layer_id++)
        {
            const auto dst_offset = dst_batch_off + layer_id * layer_stride;
            const auto src_offset = src_batch_off + layer_id * layer_stride;

            std::copy(src_array.begin() + src_offset,
                      src_array.begin() + src_offset + copy_size,
                      dst_array.begin() + dst_offset);
        }
    }
}

template <typename Tgpu, typename Tref>
class RNNSeqDriver : public Driver
{
public:
    RNNSeqDriver() : Driver()
    {
        miopenCreateSeqTensorDescriptor(&inputSeqTensor);
        miopenCreateTensorDescriptor(&hiddenTensor);
        miopenCreateTensorDescriptor(&inputTensor_dims);
        miopenCreateSeqTensorDescriptor(&outputSeqTensor);

        miopenCreateRNNDescriptor(&rnnDesc);
        miopenCreateDropoutDescriptor(&DropoutDesc);
        workspace_dev    = nullptr;
        reservespace_dev = nullptr;

        InitDataType<Tgpu>();
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;

    InputFlags& GetInputFlags() override { return inflags; }

    int CheckDescriptor(miopenSeqTensorDescriptor_t desc,
                        miopenRNNBaseLayout_t layout,
                        const std::vector<int>& src_lens,
                        const std::vector<int>& src_Slens);
    int CheckDescriptor(miopenTensorDescriptor_t src_desc, const std::vector<int>& src_lens);

    int GetandSetData() override;

    std::vector<int> GetSeqLengthsFromCmdLine();
    miopenRNNBaseLayout_t GetIODataLayoutFromCmdLine();
    miopenRNNFWDMode_t GetRNNFwdModeFromCmdLine();

    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetHiddenTensorLengthsFromCmdLine();
    std::vector<int> GetOutputTensorLengthsFromCmdLine();

    int SetRNNDescriptorFromCmdLineArgs();
    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();
    int RunBackwardGPU() override;
    int RunBackwardDataCPU();
    int RunBackwardWeightsCPU();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~RNNSeqDriver() override
    {
        miopenDestroySeqTensorDescriptor(outputSeqTensor);
        miopenDestroyTensorDescriptor(inputTensor_dims);
        miopenDestroyTensorDescriptor(hiddenTensor);
        miopenDestroySeqTensorDescriptor(inputSeqTensor);

        miopenDestroyRNNDescriptor(rnnDesc);
    }

private:
    InputFlags inflags;

    miopenSeqTensorDescriptor_t inputSeqTensor;
    miopenSeqTensorDescriptor_t outputSeqTensor;
    miopenTensorDescriptor_t hiddenTensor;
    miopenTensorDescriptor_t inputTensor_dims;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> wei_dev;
    std::unique_ptr<GPUMem> dwei_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dout_dev;
    std::unique_ptr<GPUMem> hx_dev;
    std::unique_ptr<GPUMem> cx_dev;
    std::unique_ptr<GPUMem> hy_dev;
    std::unique_ptr<GPUMem> cy_dev;
    std::unique_ptr<GPUMem> dhx_dev;
    std::unique_ptr<GPUMem> dcx_dev;
    std::unique_ptr<GPUMem> dhy_dev;
    std::unique_ptr<GPUMem> dcy_dev;
    std::unique_ptr<GPUMem> workspace_dev;
    std::unique_ptr<GPUMem> reservespace_dev;
    std::unique_ptr<GPUMem> dropout_states_dev;

    // for debug
    std::vector<Tgpu> tmp_gpu_in;
    std::vector<Tgpu> tmp_gpu_hx;
    std::vector<Tgpu> tmp_gpu_cx;
    std::vector<Tgpu> from_gpu_out;

    std::vector<Tgpu> in;
    std::vector<Tgpu> din;
    std::vector<Tgpu> wei;
    std::vector<Tgpu> dwei;
    std::vector<Tgpu> out;
    std::vector<Tgpu> dout;
    std::vector<Tgpu> hx;
    std::vector<Tgpu> cx;
    std::vector<Tgpu> hy;
    std::vector<Tgpu> cy;
    std::vector<Tgpu> dhx;
    std::vector<Tgpu> dcx;
    std::vector<Tgpu> dhy;
    std::vector<Tgpu> dcy;
    std::vector<Tgpu> workspace;
    std::vector<Tgpu> reservespace;

    std::vector<Tref> outhost;
    std::vector<Tref> workspace_host;
    std::vector<Tref> reservespace_host;
    std::vector<Tref> din_host;
    std::vector<Tref> dwei_host;
    std::vector<Tref> hy_host;
    std::vector<Tref> cy_host;
    std::vector<Tref> dhx_host;
    std::vector<Tref> dcx_host;
    std::vector<rocrand_state_xorwow> dropout_states_host;

    miopenRNNDescriptor_t rnnDesc;

    ///////////////

    std::vector<int> sorted_seq_lens;
    std::vector<int> unsorted_seq_lens;

    miopenRNNFWDMode_t fwd_type;
    miopenRNNBaseLayout_t io_layout;
    ///////////////

    miopenDropoutDescriptor_t DropoutDesc;
    float dropout_rate;
    unsigned long long dropout_seed;
};

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    int nn_dir = inflags.GetValueInt("forw");
    if(inflags.GetValueInt("fwdtype") == 1 && !(nn_dir == 0 || nn_dir == 1))
    {
        MIOPEN_THROW(
            "Incorrect input, In Inference only fwd direction is allowed ((forw=0) OR (forw=1)).");
    }

    if(inflags.GetValueInt("iter") > 1 && inflags.GetValueInt("verify") != 0)
        MIOPEN_THROW(
            "To use non default Number of Iterations >1 need to disable Verification -V 0.");

    if((nn_dir & 4) && !(nn_dir & 2) && !(nn_dir & 1))
        MIOPEN_THROW("Incorrect input, calculation of BackwardWeights require BackwardData and "
                     "ForwardData.");
    if((nn_dir & 2) && !(nn_dir & 1))
        MIOPEN_THROW("Incorrect input, calculation of BackwardData require ForwardData.");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::CheckDescriptor(miopenSeqTensorDescriptor_t src_desc,
                                              miopenRNNBaseLayout_t src_layout,
                                              const std::vector<int>& src_lens,
                                              const std::vector<int>& src_Slens)
{
    miopenDataType_t dt;
    miopenRNNBaseLayout_t layout;
    std::vector<int> lens(src_lens.size());
    std::vector<int> slens(src_Slens.size());

    miopenGetRNNDataSeqTensorDescriptor(
        src_desc, &dt, &layout, &lens[1], &lens[0], &lens[2], slens.size(), slens.data(), nullptr);

    if(!std::equal(std::begin(src_lens), std::end(src_lens), std::begin(lens)) ||
       !std::equal(std::begin(src_Slens), std::end(src_Slens), std::begin(slens)) ||
       layout != src_layout)
    {
        return miopenStatusInternalError;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::CheckDescriptor(miopenTensorDescriptor_t src_desc,
                                              const std::vector<int>& src_lens)
{
    const std::vector<int> lens = GetTensorLengths(src_desc);

    if(lens.size() != src_lens.size() ||
       !std::equal(src_lens.begin(), src_lens.end(), lens.begin()))
    {
        return miopenStatusInternalError;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::GetandSetData()
{
    int status = 0;
    io_layout  = GetIODataLayoutFromCmdLine();

    unsorted_seq_lens = GetSeqLengthsFromCmdLine();
    sorted_seq_lens   = unsorted_seq_lens;
    std::sort(sorted_seq_lens.begin(), sorted_seq_lens.end(), std::greater<int>());

    const std::vector<int> in_lens  = GetInputTensorLengthsFromCmdLine();
    const std::vector<int> out_lens = GetOutputTensorLengthsFromCmdLine();
    const std::vector<int> hid_len  = GetHiddenTensorLengthsFromCmdLine();

    {
        const std::vector<int> in_dims = {in_lens[0], in_lens[2]};
        miopenSetTensorDescriptor(inputTensor_dims, data_type, 2, in_dims.data(), nullptr);
        status = CheckDescriptor(inputTensor_dims, in_dims);
    }

    if(status != miopenStatusSuccess)
    {
        printf("Error checking TensorDescriptor:%s content.\n", "inputTensor_dims");
        return miopenStatusInternalError;
    }

    Tgpu alfa = static_cast<Tgpu>(-1);

    miopenSetRNNDataSeqTensorDescriptor(inputSeqTensor,
                                        data_type,
                                        io_layout,
                                        in_lens[1],
                                        in_lens[0],
                                        in_lens[2],
                                        unsorted_seq_lens.data(),
                                        &alfa);

    status = CheckDescriptor(inputSeqTensor, io_layout, in_lens, unsorted_seq_lens);
    if(status != miopenStatusSuccess)
    {
        printf("Error checking SeqTensorDescriptor:%s content.\n", "inputSeqTensor");
        return status;
    }

    miopenSetRNNDataSeqTensorDescriptor(outputSeqTensor,
                                        data_type,
                                        io_layout,
                                        out_lens[1],
                                        out_lens[0],
                                        out_lens[2],
                                        unsorted_seq_lens.data(),
                                        &alfa);

    status = CheckDescriptor(outputSeqTensor, io_layout, out_lens, unsorted_seq_lens);
    if(status != miopenStatusSuccess)
    {
        printf("Error checking SeqTensorDescriptor:%s content.\n", "outputSeqTensor");
        return status;
    }

    miopenSetTensorDescriptor(hiddenTensor, data_type, 3, hid_len.data(), nullptr);
    status = CheckDescriptor(hiddenTensor, hid_len);
    if(status != miopenStatusSuccess)
    {
        printf("Error checking TensorDescriptor:%s content.\n", "hiddenTensor");
        return miopenStatusInternalError;
    }

    SetRNNDescriptorFromCmdLineArgs();
    fwd_type = GetRNNFwdModeFromCmdLine();

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run only Forward RNN == 1 or only Backward Data RNN == 2, Backward "
                         "Weights = 4 or both == 0 (Default=0)",
                         "int");
    inflags.AddInputFlag("fwdtype",
                         'c',
                         "0",
                         "RNN forward being training or inference, Default training (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "mode", 'm', "tanh", "RNN Mode (relu, tanh, lstm, gru) (Default=tanh)", "str");

    inflags.AddInputFlag("datatype", 'f', "1", "16-bit or 32-bit fp (Default=1)", "int");
    inflags.AddInputFlag("io_layout",
                         'I',
                         "1",
                         "IO data layout"
                         "SeqMajorNotPadded = 1"
                         "SeqMajorPadded = 2"
                         "BatchMajorPadded = 3 (Default=1)",
                         "int");

    inflags.AddInputFlag("num_layer", 'l', "1", "Number of hidden stacks (Default=1)", "int");
    inflags.AddInputFlag("batch_size", 'n', "4", "Mini-batch size (Default=4)", "int");
    inflags.AddInputFlag(
        "seq_len", 'k', "10", "Max number of iterations to unroll over (Default=10)", "int");
    inflags.AddInputFlag("seq_len_array",
                         'K',
                         "",
                         "Array of SeqLength for each sample in batch (Default=4)",
                         "vector");

    inflags.AddInputFlag("hid_h", 'H', "32", "Hidden State Length (Default=32)", "int");
    inflags.AddInputFlag("in_vec", 'W', "32", "Input Length (Default=32)", "int");

    inflags.AddInputFlag(
        "bidirection", 'r', "0", "uni- or bi-direction, default uni- (Default=0)", "int");
    inflags.AddInputFlag("bias", 'b', "", "Use Bias (Default=0)", "int");
    inflags.AddInputFlag("inputmode", 'p', "0", "linear == 0 or skip == 1, (Default=0)", "int");

    inflags.AddInputFlag("rnnalgo", 'a', "0", "default, fundamental (Default=0)", "int");

    inflags.AddInputFlag(
        "use_dropout", 'U', "0", "Use dropout: 1; Not use dropout: 0 (Default=0)", "int");
    inflags.AddInputFlag("dropout", 'P', "0.0", "Dropout rate (Default=0.0)", "float");
    inflags.AddInputFlag(
        "seed_low", 'L', "0", "Least significant 32 bits of seed (Default=0)", "int");
    inflags.AddInputFlag(
        "seed_high", 'M', "0", "Most significant 32 bits of seed (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");

    inflags.AddInputFlag(
        "wall",
        'w',
        "0",
        "Wall-clock, for host and gpu, Time Each Layer,       Disabled                = 0,\
        OldWallClock            = 1,\
        SeparateClocksSynced    = 2,\
        SeparateClocksNotSynced = 3 \
        (Default = 0) ",
        "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    /*  // DL: These have not been implemented. Removing them for now.
        inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");
        inflags.AddInputFlag("weights", 'e', "", "Input weights filename (Default=)", "string");*/

    return 0;
}

template <typename Tgpu, typename Tref>
miopenRNNBaseLayout_t RNNSeqDriver<Tgpu, Tref>::GetIODataLayoutFromCmdLine()
{
    int layout = inflags.GetValueInt("io_layout");
    switch(layout)
    {
    case 1: return miopenRNNDataSeqMajorNotPadded;
    case 2: return miopenRNNDataSeqMajorPadded;
    case 3: return miopenRNNDataBatchMajorPadded;
    default: MIOPEN_THROW("Incorrect input, unsupported RNNLayout.");
    }
}

template <typename Tgpu, typename Tref>
miopenRNNFWDMode_t RNNSeqDriver<Tgpu, Tref>::GetRNNFwdModeFromCmdLine()
{
    int fwdtype = inflags.GetValueInt("fwdtype");
    switch(fwdtype)
    {
    case 0: return miopenRNNFWDMode_t::miopenRNNTraining;
    case 1: return miopenRNNFWDMode_t::miopenRNNInference;
    default: MIOPEN_THROW("Incorrect input, unsupported fwdtype.");
    }
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNSeqDriver<Tgpu, Tref>::GetSeqLengthsFromCmdLine()
{
    int batch_size  = inflags.GetValueInt("batch_size");
    int seq_len_max = inflags.GetValueInt("seq_len");

    std::vector<int> data_seq_lens(batch_size, 0);

    std::string s_lens = inflags.GetValueStr("seq_len_array");
    std::stringstream ss(s_lens);

    auto seq_it = data_seq_lens.begin();
    int element;

    while(ss >> element)
    {
        if(seq_it == data_seq_lens.end())
            MIOPEN_THROW("Incorrect input, seq_len_array bigger than provided batch_size.\n");

        if(element > seq_len_max)
            MIOPEN_THROW("Length of data sequence is longer than required unrolled time sequence "
                         "provided.\n"
                         "The data sequence will be truncated to match unrolled time sequence.\n");

        *(seq_it++) = element;

        if(ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }
    }

    if(io_layout == miopenRNNDataSeqMajorNotPadded && (seq_it != data_seq_lens.begin()) &&
       (!std::is_sorted(data_seq_lens.begin(), seq_it, std::greater<>{})))
    {
        MIOPEN_THROW("Incorrect input, seq_lens should not to increase with "
                     "miopenRNNDataSeqMajorNotPadded layout\n");
    }

    if(seq_it != data_seq_lens.end())
    {
        auto padding_val = (seq_it != data_seq_lens.begin()) ? *(seq_it - 1) : seq_len_max;
        std::cout << "sampl_lens size == " << std::distance(data_seq_lens.begin(), seq_it)
                  << " is smaller than time batch_size == " << batch_size
                  << ", padding the rest of data with " << padding_val << "\n";

        for(; seq_it != data_seq_lens.end(); seq_it++)
            *seq_it = padding_val;
    }

    return data_seq_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNSeqDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int batch_size = inflags.GetValueInt("batch_size");
    int seq_len    = inflags.GetValueInt("seq_len");
    int in_vec     = inflags.GetValueInt("in_vec");

    return std::vector<int>({batch_size, seq_len, in_vec});
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNSeqDriver<Tgpu, Tref>::GetOutputTensorLengthsFromCmdLine()
{
    int batch_size = inflags.GetValueInt("batch_size");
    int seq_len    = inflags.GetValueInt("seq_len");

    int hid_h   = inflags.GetValueInt("hid_h");
    int bi      = (inflags.GetValueInt("bidirection") == 1) ? 2 : 1;
    int out_vec = hid_h * bi;

    return std::vector<int>({batch_size, seq_len, out_vec});
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNSeqDriver<Tgpu, Tref>::GetHiddenTensorLengthsFromCmdLine()
{
    int n_layer = inflags.GetValueInt("num_layer");
    if((inflags.GetValueInt("bidirection")) == 1)
        n_layer *= 2;

    int batch_size = inflags.GetValueInt("batch_size");
    int hid_vec    = inflags.GetValueInt("hid_h");

    return std::vector<int>({n_layer, batch_size, hid_vec});
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::SetRNNDescriptorFromCmdLineArgs()
{

    const int layer       = inflags.GetValueInt("num_layer");
    const int hidden_size = inflags.GetValueInt("hid_h");

    miopenRNNMode_t mode;

    if((inflags.GetValueStr("mode")) == "relu")
    {
        mode = miopenRNNRELU;
    }
    else if((inflags.GetValueStr("mode")) == "tanh")
    {
        mode = miopenRNNTANH;
    }
    else if((inflags.GetValueStr("mode")) == "lstm")
    {
        mode = miopenLSTM;
    }
    else if((inflags.GetValueStr("mode")) == "gru")
    {
        mode = miopenGRU;
    }
    else
    {
        printf("Incorrect RNN Mode\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    miopenRNNBiasMode_t biasMode;
    if((inflags.GetValueInt("bias")) == 0)
    {
        biasMode = miopenRNNNoBias;
    }
    else if((inflags.GetValueInt("bias")) == 1)
    {
        biasMode = miopenRNNwithBias;
    }
    else
    {
        printf("Incorrect bias Mode\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    miopenRNNDirectionMode_t directionMode;
    if((inflags.GetValueInt("bidirection")) == 0)
    {
        directionMode = miopenRNNunidirection;
    }
    else if((inflags.GetValueInt("bidirection")) == 1)
    {
        directionMode = miopenRNNbidirection;
    }
    else
    {
        printf("Incorrect direction Mode\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    miopenRNNInputMode_t inMode;
    if((inflags.GetValueInt("inputmode")) == 0)
    {
        inMode = miopenRNNlinear;
    }
    else if((inflags.GetValueInt("inputmode")) == 1)
    {
        inMode = miopenRNNskip;
    }
    else
    {
        printf("Incorrect input Mode\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    miopenRNNAlgo_t algo;
    if((inflags.GetValueInt("rnnalgo")) == 0)
    {
        algo = miopenRNNdefault;
    }
    else if((inflags.GetValueInt("rnnalgo")) == 1)
    {
        algo = miopenRNNfundamental;
    }
    else
    {
        printf("Incorrect RNN algorithm\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    if(inflags.GetValueInt("use_dropout"))
    {
        dropout_rate = static_cast<float>(inflags.GetValueDouble("dropout"));
        auto dropout_seed_low =
            static_cast<unsigned long long>(std::max(inflags.GetValueInt("seed_low"), 0));
        auto dropout_seed_high =
            static_cast<unsigned long long>(std::max(inflags.GetValueInt("seed_high"), 0));
        dropout_seed = dropout_seed_high << 32 | dropout_seed_low;

        size_t statesSizeInBytes = 0;
        miopenDropoutGetStatesSize(GetHandle(), &statesSizeInBytes);
        size_t states_size = statesSizeInBytes / sizeof(rocrand_state_xorwow);

        DEFINE_CONTEXT(ctx);
#if MIOPEN_BACKEND_OPENCL
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif

        dropout_states_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, states_size, sizeof(rocrand_state_xorwow)));

        miopenSetDropoutDescriptor(DropoutDesc,
                                   GetHandle(),
                                   dropout_rate,
                                   dropout_states_dev->GetMem(),
                                   dropout_states_dev->GetSize(),
                                   dropout_seed,
                                   false,
                                   false,
                                   MIOPEN_RNG_PSEUDO_XORWOW);

        miopenSetRNNDescriptor_V2(rnnDesc,
                                  hidden_size,
                                  layer,
                                  DropoutDesc,
                                  inMode,
                                  directionMode,
                                  mode,
                                  biasMode,
                                  algo,
                                  data_type);
    }
    else
    {
        miopenSetRNNDescriptor(
            rnnDesc, hidden_size, layer, inMode, directionMode, mode, biasMode, algo, data_type);
    }

    if(io_layout != miopenRNNDataSeqMajorNotPadded)
    {
        miopenSetRNNPaddingMode(rnnDesc, miopenRNNPaddingMode_t::miopenRNNIOWithPadding);
    }

    return miopenStatusSuccess;
}

// GetTensorSize broken So this is WA
inline size_t Get3DNoVECTensorSize(miopenTensorDescriptor_t& tensor)
{
    assert(miopen::deref(tensor).IsPacked() &&
           "GetTensorSize should not be used on an unpacked tensor.");
    const auto len = GetTensorLengths(tensor);
    size_t sz      = std::accumulate(len.begin(), len.end(), 1ULL, std::multiplies<size_t>());
    return sz;
}

std::vector<int> GetSamplesIndexDescendingOrder(const std::vector<int>& unsorted_seq_lens,
                                                bool reverse)
{
    const auto sample_count = unsorted_seq_lens.size();

    std::vector<int> index_v(sample_count);
    std::iota(index_v.begin(), index_v.end(), 0);

    auto seq_len_cmp = [&unsorted_seq_lens](unsigned a_id, unsigned b_id) {
        return unsorted_seq_lens[a_id] > unsorted_seq_lens[b_id];
    };

    std::stable_sort(index_v.begin(), index_v.end(), seq_len_cmp);

    auto get_reverse_index = [](const std::vector<int>& base_index) {
        std::vector<int> reverse_index(base_index.size());
        unsigned next_rev_index = 0;
        for(auto id : base_index)
            reverse_index[id] = next_rev_index++;
        return reverse_index;
    };

    return !reverse ? index_v : get_reverse_index(index_v);
}

std::vector<int> GetBatchesPerTime(const std::vector<int>& sorted_sequence_len)
{
    std::vector<int> batches;
    auto block_begin = sorted_sequence_len.rbegin();
    auto sample_ptr  = sorted_sequence_len.rbegin();
    auto batch_size  = sorted_sequence_len.size();

    batches.insert(batches.end(), *block_begin, batch_size);

    while(sample_ptr != sorted_sequence_len.rend())
    {
        if(*sample_ptr != *block_begin)
        {
            batch_size           = batch_size - (sample_ptr - block_begin);
            const auto seq_count = *sample_ptr - *block_begin;
            batches.insert(batches.end(), seq_count, batch_size);
            block_begin = sample_ptr;
        }
        sample_ptr++;
    }
    return batches;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    int status = 0;

    const std::vector<int> in_lens  = GetInputTensorLengthsFromCmdLine();
    const std::vector<int> out_lens = GetOutputTensorLengthsFromCmdLine();

    const size_t vectors_cnt_host =
        std::accumulate(sorted_seq_lens.begin(), sorted_seq_lens.end(), 0ULL);
    const size_t vectors_cnt_gpu =
        io_layout == miopenRNNDataSeqMajorNotPadded ? vectors_cnt_host : in_lens[0] * in_lens[1];

    const size_t in_host_sz  = vectors_cnt_host * in_lens[2];
    const size_t out_host_sz = vectors_cnt_host * out_lens[2];

    const size_t in_gpu_sz  = vectors_cnt_gpu * in_lens[2];
    const size_t out_gpu_sz = vectors_cnt_gpu * out_lens[2];

    const size_t hid_sz = Get3DNoVECTensorSize(hiddenTensor);

    size_t workSpace_sz;
    size_t reserveSpace_sz;
    status |= miopenGetRNNTempSpaceSizes(
        GetHandle(), rnnDesc, inputSeqTensor, fwd_type, &workSpace_sz, &reserveSpace_sz);

    workSpace_sz /= sizeof(Tgpu);
    if(inflags.GetValueInt("use_dropout"))
        reserveSpace_sz = (reserveSpace_sz + sizeof(Tgpu) - 1) / sizeof(Tgpu);
    else
        reserveSpace_sz /= sizeof(Tgpu);

    size_t wei_sz = 0;
    status |= miopenGetRNNParamsSize(GetHandle(), rnnDesc, inputTensor_dims, &wei_sz, data_type);
    wei_sz /= sizeof(Tgpu);

    if(status != HIP_SUCCESS)
    {
        printf("Error at getting required space for RNN rnnDesc\n");
        return miopenStatusNotInitialized;
    }

    const uint32_t ctx = 0; // opencl legacy

    in_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_gpu_sz, sizeof(Tgpu)));
    hx_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
    out_dev          = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_gpu_sz, sizeof(Tgpu)));
    wei_dev          = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
    cx_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
    workspace_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpace_sz, sizeof(Tgpu)));
    reservespace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, reserveSpace_sz, sizeof(Tgpu)));

    if(inflags.GetValueInt("forw") != 2)
    {
        hy_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
        cy_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
    }

    if(inflags.GetValueInt("forw") != 1)
    {
        din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_gpu_sz, sizeof(Tgpu)));
        dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
        dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_gpu_sz, sizeof(Tgpu)));
        dhx_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
        dcx_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
        dhy_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
        dcy_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hid_sz, sizeof(Tgpu)));
    }

    in  = std::vector<Tgpu>(in_host_sz);
    hx  = std::vector<Tgpu>(hid_sz);
    wei = std::vector<Tgpu>(wei_sz);
    out = std::vector<Tgpu>(out_host_sz, static_cast<Tgpu>(0));
    cx  = std::vector<Tgpu>(hid_sz);

    if(inflags.GetValueInt("forw") != 2)
    {
        hy      = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
        cy      = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
        hy_host = std::vector<Tref>(hid_sz, static_cast<Tref>(0));
        cy_host = std::vector<Tref>(hid_sz, static_cast<Tref>(0));
    }

    workspace      = std::vector<Tgpu>(workSpace_sz, static_cast<Tgpu>(0));
    reservespace   = std::vector<Tgpu>(reserveSpace_sz, static_cast<Tgpu>(0));
    outhost        = std::vector<Tref>(out_host_sz, static_cast<Tref>(0));
    workspace_host = std::vector<Tref>(workSpace_sz, static_cast<Tref>(0));

    /// dropout legacy format
    const std::size_t inputBatchLenSum = vectors_cnt_host;

    int hid_h = inflags.GetValueInt("hid_h");
    int layer = inflags.GetValueInt("num_layer");
    int bidir = inflags.GetValueInt("bidirection");

    size_t reserveSpaceHost_sz =
        2 *
        (inflags.GetValueStr("mode") == "lstm" ? 6
                                               : (inflags.GetValueStr("mode") == "gru" ? 4 : 1)) *
        layer * inputBatchLenSum * hid_h * (bidir + 1);
    if(inflags.GetValueInt("use_dropout"))
    {
        reserveSpaceHost_sz += (layer - 1) * inputBatchLenSum * hid_h * (bidir + 1);
        reserveSpaceHost_sz *= sizeof(Tref);
        reserveSpaceHost_sz += (layer - 1) * inputBatchLenSum * hid_h * (bidir + 1);
        reserveSpaceHost_sz = (reserveSpaceHost_sz + sizeof(Tref) - 1) / sizeof(Tref);
    }
    reservespace_host = std::vector<Tref>(reserveSpaceHost_sz, static_cast<Tref>(0));
    // end dropout

    if(inflags.GetValueInt("forw") != 1)
    {
        din       = std::vector<Tgpu>(in_host_sz, static_cast<Tgpu>(0));
        dwei      = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
        dout      = std::vector<Tgpu>(out_host_sz, static_cast<Tgpu>(0));
        dhx       = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
        dcx       = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
        dhy       = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
        dcy       = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
        din_host  = std::vector<Tref>(in_host_sz, static_cast<Tref>(0));
        dwei_host = std::vector<Tref>(wei_sz, static_cast<Tref>(0));
        dhx_host  = std::vector<Tref>(hid_sz, static_cast<Tref>(0));
        dcx_host  = std::vector<Tref>(hid_sz, static_cast<Tref>(0));
    }

    // Unless seed is persistent between runs validation using cache stored in file is impossible.
    prng::reset_seed();

    auto fill_array_via_gen = [](auto& dst, size_t dst_sz, double range_l, double range_r) {
        for(size_t it = 0; it < dst_sz; it++)
            dst[it] = prng::gen_A_to_B(static_cast<Tgpu>(range_l), static_cast<Tgpu>(range_r));
    };

    const double scale = 0.01;
    fill_array_via_gen(in, in_host_sz, 0.0, 1.0 * scale);
    fill_array_via_gen(hx, hid_sz, 0.0, 1.0 * scale);
    fill_array_via_gen(wei, wei_sz, -0.5 * scale, 0.5 * scale);

    if((inflags.GetValueStr("mode")) == "lstm")
        fill_array_via_gen(cx, hid_sz, 0.0, 1.0 * scale);

    if(inflags.GetValueInt("forw") != 1)
    {
        fill_array_via_gen(dout, out_host_sz, 0.0, 1.0 * scale);

        fill_array_via_gen(dhy, hid_sz, 0.0, 1.0 * scale);

        if((inflags.GetValueStr("mode")) == "lstm")
            fill_array_via_gen(dcy, hid_sz, 0.0, 1.0 * scale);
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_in.bin", in.data(), in_host_sz);
        dumpBufferToFile("dump_wei.bin", wei.data(), wei_sz);
    }

    const std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    if(io_layout != miopenRNNDataSeqMajorNotPadded)
    {
        const std::vector<int> batches_per_time = GetBatchesPerTime(sorted_seq_lens);
        const std::vector<int> order_idxs =
            GetSamplesIndexDescendingOrder(unsorted_seq_lens, false);

        tmp_gpu_in = std::vector<Tgpu>(in_gpu_sz, static_cast<Tgpu>(0));
        tmp_gpu_hx = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));

        TransformIODefaultLayaoutToTarget(
            in, tmp_gpu_in, batches_per_time, order_idxs, in_lens[1], in_lens[2], io_layout, true);

        HiddenTensorReorder(hx, tmp_gpu_hx, order_idxs, hid_len, false);

        status |= in_dev->ToGPU(q, tmp_gpu_in.data());
        status |= hx_dev->ToGPU(q, tmp_gpu_hx.data());

        if((inflags.GetValueStr("mode")) == "lstm")
        {
            tmp_gpu_cx = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
            HiddenTensorReorder(cx, tmp_gpu_cx, order_idxs, hid_len, false);
            status |= cx_dev->ToGPU(q, tmp_gpu_cx.data());
        }

        if(inflags.GetValueInt("forw") != 1)
        {
            {
                std::vector<Tgpu> tmp_gpu_dout =
                    std::vector<Tgpu>(out_gpu_sz, static_cast<Tgpu>(0));
                TransformIODefaultLayaoutToTarget(dout,
                                                  tmp_gpu_dout,
                                                  batches_per_time,
                                                  order_idxs,
                                                  out_lens[1],
                                                  out_lens[2],
                                                  io_layout,
                                                  true);
                status |= dout_dev->ToGPU(q, tmp_gpu_dout.data());
            }
            {
                std::vector<Tgpu> tmp_gpu_dhy = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
                HiddenTensorReorder(dhy, tmp_gpu_dhy, order_idxs, hid_len, false);
                status |= dhy_dev->ToGPU(q, tmp_gpu_dhy.data());
            }
            if((inflags.GetValueStr("mode")) == "lstm")
            {
                std::vector<Tgpu> tmp_gpu_dcy = std::vector<Tgpu>(hid_sz, static_cast<Tgpu>(0));
                HiddenTensorReorder(dcy, tmp_gpu_dcy, order_idxs, hid_len, false);
                status |= dcy_dev->ToGPU(q, tmp_gpu_dcy.data());
            }
        }
        if(status != HIP_SUCCESS)
        {
            printf("Error copying data to GPU\n");
            return miopenStatusNotInitialized;
        }
    }
    else
    {
        status |= in_dev->ToGPU(q, in.data());
        status |= hx_dev->ToGPU(q, hx.data());
        status |= cx_dev->ToGPU(q, cx.data());
        if(inflags.GetValueInt("forw") != 1)
        {
            status |= dout_dev->ToGPU(q, dout.data());
            status |= dhy_dev->ToGPU(q, dhy.data());
            if((inflags.GetValueStr("mode")) == "lstm")
                status |= dcy_dev->ToGPU(q, dcy.data());
        }

        if(status != HIP_SUCCESS)
        {
            printf("Error copying data to GPU\n");
            return miopenStatusNotInitialized;
        }
    }

    status |= wei_dev->ToGPU(q, wei.data());
    // status |= workspace_dev->ToGPU(q, workspace.data());
    // status |= reservespace_dev->ToGPU(q, reservespace.data());

    if(status != HIP_SUCCESS)
    {
        printf("Error copying data to GPU\n");
        return miopenStatusNotInitialized;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::RunForwardGPU()
{

    if(inflags.GetValueInt("forw") != 0 && !(inflags.GetValueInt("forw") & 1))
        return miopenStatusSuccess;

    RNNCombTimeLoger t(GetStream(), inflags.GetValueInt("iter"), inflags.GetValueInt("wall"));

    from_gpu_out = std::vector<Tgpu>(out_dev->GetSize() / sizeof(Tgpu), static_cast<Tgpu>(0));

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        out_dev->ToGPU(q, from_gpu_out.data());
        workspace_dev->ToGPU(q, workspace.data());
        reservespace_dev->ToGPU(q, reservespace.data());

        t.Start();
        miopenRNNForward(GetHandle(),
                         rnnDesc,
                         fwd_type,
                         inputSeqTensor,
                         in_dev->GetMem(),
                         hiddenTensor,
                         hx_dev->GetMem(),
                         hy_dev->GetMem(),
                         hiddenTensor, // cdesc
                         cx_dev->GetMem(),
                         cy_dev->GetMem(),
                         outputSeqTensor,
                         out_dev->GetMem(),
                         wei_dev->GetMem(),
                         wei_dev->GetSize(),
                         workspace_dev->GetMem(),
                         workspace_dev->GetSize(),
                         reservespace_dev->GetMem(),
                         reservespace_dev->GetSize());

        t.StopAndPush();
    }

    miopen::deref(GetHandle()).Finish();
    if(WALL_CLOCK)
    {
        printf("Forward RNN time results:\n");
        t.Print();
    }

    if(io_layout != miopenRNNDataSeqMajorNotPadded)
    {
        auto from_gpu_hy = std::vector<Tgpu>(hy_dev->GetSize() / sizeof(Tgpu));
        auto from_gpu_cy = std::vector<Tgpu>(cy_dev->GetSize() / sizeof(Tgpu));

        out_dev->FromGPU(GetStream(), from_gpu_out.data());
        hy_dev->FromGPU(GetStream(), from_gpu_hy.data());
        cy_dev->FromGPU(GetStream(), from_gpu_cy.data());

        const std::vector<int> batches_per_time = GetBatchesPerTime(sorted_seq_lens);
        const std::vector<int> order_idxs =
            GetSamplesIndexDescendingOrder(unsorted_seq_lens, false);

        const std::vector<int> hid_lens = GetHiddenTensorLengthsFromCmdLine();
        const std::vector<int> out_lens = GetOutputTensorLengthsFromCmdLine();

        TransformIODefaultLayaoutToTarget(out,
                                          from_gpu_out,
                                          batches_per_time,
                                          order_idxs,
                                          out_lens[1],
                                          out_lens[2],
                                          io_layout,
                                          false);

        HiddenTensorReorder(from_gpu_hy, hy, order_idxs, hid_lens, true);
        HiddenTensorReorder(from_gpu_cy, cy, order_idxs, hid_lens, true);
    }
    else
    {
        out_dev->FromGPU(GetStream(), out.data());
        hy_dev->FromGPU(GetStream(), hy.data());
        cy_dev->FromGPU(GetStream(), cy.data());
    }
    reservespace_dev->FromGPU(GetStream(), reservespace.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::RunBackwardGPU()
{
    int ret = 0;
    if(inflags.GetValueInt("fwdtype") == 1 || inflags.GetValueInt("forw") == 1)
    {
        return ret;
    }

    if((inflags.GetValueInt("forw") & 2) || (inflags.GetValueInt("forw") == 0))
    {
        RNNCombTimeLoger t(GetStream(), inflags.GetValueInt("iter"), inflags.GetValueInt("wall"));

        workspace_dev->ToGPU(q, workspace.data());
        if(inflags.GetValueInt("inputmode") == 1)
        {
            // skip mode bug or feature, but din=F(...)+din
            auto tmp_gpu_din =
                std::vector<Tgpu>(din_dev->GetSize() / sizeof(Tgpu), static_cast<Tgpu>(0));
            din_dev->ToGPU(GetStream(), tmp_gpu_din.data());
        }

        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            t.Start();
            ret = miopenRNNBackwardSeqData(GetHandle(),
                                           rnnDesc,
                                           outputSeqTensor,
                                           out_dev->GetMem(),
                                           dout_dev->GetMem(),
                                           hiddenTensor,
                                           hx_dev->GetMem(),
                                           dhy_dev->GetMem(),
                                           dhx_dev->GetMem(),
                                           hiddenTensor,
                                           cx_dev->GetMem(),
                                           dcy_dev->GetMem(),
                                           dcx_dev->GetMem(),
                                           inputSeqTensor,
                                           din_dev->GetMem(),
                                           wei_dev->GetMem(),
                                           wei_dev->GetSize(),
                                           workspace_dev->GetMem(),
                                           workspace_dev->GetSize(),
                                           reservespace_dev->GetMem(),
                                           reservespace_dev->GetSize());
            t.StopAndPush();
        }

        miopen::deref(GetHandle()).Finish();
        if(WALL_CLOCK)
        {
            printf("Backward Data RNN time results:\n");
            t.Print();
        }

        if(io_layout != miopenRNNDataSeqMajorNotPadded)
        {
            auto from_gpu_din = std::vector<Tgpu>(din_dev->GetSize() / sizeof(Tgpu));
            auto from_gpu_dhx = std::vector<Tgpu>(dhx_dev->GetSize() / sizeof(Tgpu));
            auto from_gpu_dcx = std::vector<Tgpu>(dcx_dev->GetSize() / sizeof(Tgpu));

            din_dev->FromGPU(GetStream(), from_gpu_din.data());
            dhx_dev->FromGPU(GetStream(), from_gpu_dhx.data());
            dcx_dev->FromGPU(GetStream(), from_gpu_dcx.data());

            const std::vector<int> batches_per_time = GetBatchesPerTime(sorted_seq_lens);
            const std::vector<int> order_idxs =
                GetSamplesIndexDescendingOrder(unsorted_seq_lens, false);

            const std::vector<int> hid_lens = GetHiddenTensorLengthsFromCmdLine();
            const std::vector<int> in_lens  = GetInputTensorLengthsFromCmdLine();

            TransformIODefaultLayaoutToTarget(din,
                                              from_gpu_din,
                                              batches_per_time,
                                              order_idxs,
                                              in_lens[1],
                                              in_lens[2],
                                              io_layout,
                                              false);

            HiddenTensorReorder(from_gpu_dhx, dhx, order_idxs, hid_lens, true);
            HiddenTensorReorder(from_gpu_dcx, dcx, order_idxs, hid_lens, true);
        }
        else
        {
            din_dev->FromGPU(GetStream(), din.data());
            dhx_dev->FromGPU(GetStream(), dhx.data());
            dcx_dev->FromGPU(GetStream(), dcx.data());
        }
        workspace_dev->FromGPU(GetStream(), workspace.data());
    }

    if((inflags.GetValueInt("forw") & 4) || (inflags.GetValueInt("forw") == 0))
    {
        RNNCombTimeLoger t(GetStream(), inflags.GetValueInt("iter"), inflags.GetValueInt("wall"));

        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            t.Start();
            ret = miopenRNNBackwardWeightsSeqTensor(GetHandle(),
                                                    rnnDesc,
                                                    inputSeqTensor,
                                                    in_dev->GetMem(),
                                                    hiddenTensor,
                                                    hx_dev->GetMem(),
                                                    outputSeqTensor,
                                                    dout_dev->GetMem(),
                                                    dwei_dev->GetMem(),
                                                    dwei_dev->GetSize(),
                                                    workspace_dev->GetMem(),
                                                    workspace_dev->GetSize(),
                                                    reservespace_dev->GetMem(),
                                                    reservespace_dev->GetSize());
            t.StopAndPush();
        }

        miopen::deref(GetHandle()).Finish();

        if(WALL_CLOCK)
        {
            printf("Backward Weights RNN time results:\n");
            t.Print();
        }
        dwei_dev->FromGPU(GetStream(), dwei.data());
    }

    /*
       if(inflags.GetValueInt("dump_output"))
       {
       dumpBufferToFile("dump_bwd_din_gpu.bin", din.data(), din.size());
       dumpBufferToFile("dump_bwd_dwei_gpu.bin", dwei.data(), dwei.size());
       }
       */

    return ret;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(inflags.GetValueInt("forw") != 0 && !(inflags.GetValueInt("forw") & 1))
        return miopenStatusSuccess;

    const std::vector<int> in_lens  = GetInputTensorLengthsFromCmdLine();
    const std::vector<int> out_lens = GetOutputTensorLengthsFromCmdLine();
    const int in_vec                = in_lens[2];
    const int out_h                 = out_lens[2];

    const std::vector<int> hid_lens = GetHiddenTensorLengthsFromCmdLine();
    const int n_layer = hid_lens[0], batch_size = hid_lens[1], hid_vec = hid_lens[2];

    const auto batchs = GetBatchesPerTime(sorted_seq_lens);
    std::vector<int> in_n(batchs.begin(), batchs.end());

    bool bidirection, biased;
    int layer;

    miopenRNNMode_t mode;
    miopenRNNAlgo_t algoMode;
    miopenRNNInputMode_t inputMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNBiasMode_t biasMode;
    int hiddenSize;
    miopenDropoutDescriptor_t drop_desc;

    if(inflags.GetValueInt("use_dropout"))
    {
        miopenGetRNNDescriptor_V2(rnnDesc,
                                  &hiddenSize,
                                  &layer,
                                  &drop_desc,
                                  &inputMode,
                                  &dirMode,
                                  &mode,
                                  &biasMode,
                                  &algoMode,
                                  nullptr);
    }
    else
    {
        miopenGetRNNDescriptor(
            rnnDesc, &mode, &algoMode, &inputMode, &dirMode, &biasMode, &hiddenSize, &layer);
    }

    bidirection = (dirMode == miopenRNNbidirection);
    biased      = (biasMode == miopenRNNwithBias);

    std::vector<Tgpu>* in_packed  = &in;
    std::vector<Tref>* out_packed = &outhost;

    if(mode == miopenRNNRELU || mode == miopenRNNTANH)
    {
        printf("verify rnn fwd \n");
        RunRNNForwardGEMMCPUVerify(GetHandle(),
                                   *in_packed,
                                   wei,
                                   hy_host,
                                   hx,
                                   *out_packed,
                                   in_n,
                                   in_vec,
                                   sorted_seq_lens[0],
                                   bidirection,
                                   biased,
                                   n_layer,
                                   batch_size,
                                   hid_vec,
                                   out_h,
                                   mode,
                                   inputMode,
                                   reservespace_host,
                                   bool(inflags.GetValueInt("use_dropout")),
                                   DropoutDesc);
    }
    else if(mode == miopenLSTM)
    {
        printf("verify lstm fwd \n");

        RunLSTMForwardGEMMCPUVerify(GetHandle(),
                                    *in_packed,
                                    wei,
                                    hy_host,
                                    hx,
                                    cy_host,
                                    cx,
                                    *out_packed,
                                    in_n,
                                    in_vec,
                                    sorted_seq_lens[0],
                                    bidirection,
                                    biased,
                                    n_layer,
                                    batch_size,
                                    hid_vec,
                                    out_h,
                                    inputMode,
                                    reservespace_host,
                                    bool(inflags.GetValueInt("use_dropout")),
                                    DropoutDesc);
    }
    else if(mode == miopenGRU)
    {
        printf("verify gru fwd \n");

        RunGRUForwardGEMMCPUVerify(GetHandle(),
                                   *in_packed,
                                   wei,
                                   hy_host,
                                   hx,
                                   *out_packed,
                                   in_n,
                                   in_vec,
                                   sorted_seq_lens[0],
                                   bidirection,
                                   biased,
                                   n_layer,
                                   batch_size,
                                   hid_vec,
                                   out_h,
                                   inputMode,
                                   reservespace_host,
                                   bool(inflags.GetValueInt("use_dropout")),
                                   DropoutDesc);
    }
    else
    {
        printf("illegal RNN mode");
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_fwd_out_cpu.bin", outhost.data(), outhost.size());
    }

    //    TrySaveVerificationCache("fwd_out", outhost);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::RunBackwardWeightsCPU()
{
    const std::vector<int> in_lens  = GetInputTensorLengthsFromCmdLine();
    const std::vector<int> out_lens = GetOutputTensorLengthsFromCmdLine();
    const int in_vec                = in_lens[2];
    const int out_h                 = out_lens[2];

    const std::vector<int> hid_lens = GetHiddenTensorLengthsFromCmdLine();
    const int n_layer = hid_lens[0], batch_size = hid_lens[1], hid_vec = hid_lens[2];

    const auto batchs = GetBatchesPerTime(sorted_seq_lens);
    std::vector<int> in_n(batchs.begin(), batchs.end());

    bool bidirection, biased;
    int layer;
    miopenRNNMode_t mode;
    miopenRNNAlgo_t algoMode;
    miopenRNNInputMode_t inputMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNBiasMode_t biasMode;
    int hiddenSize;
    miopenDropoutDescriptor_t drop_desc;

    if(inflags.GetValueInt("use_dropout"))
    {
        miopenGetRNNDescriptor_V2(rnnDesc,
                                  &hiddenSize,
                                  &layer,
                                  &drop_desc,
                                  &inputMode,
                                  &dirMode,
                                  &mode,
                                  &biasMode,
                                  &algoMode,
                                  nullptr);
    }
    else
    {
        miopenGetRNNDescriptor(
            rnnDesc, &mode, &algoMode, &inputMode, &dirMode, &biasMode, &hiddenSize, &layer);
    }

    bidirection = (dirMode == miopenRNNbidirection);
    biased      = (biasMode == miopenRNNwithBias);

    std::vector<Tgpu>* in_packed   = &in;
    std::vector<Tgpu>* dout_packed = &dout;

    if(mode == miopenRNNRELU || mode == miopenRNNTANH)
    {
        printf("verify rnn bwdwei \n");

        RunRNNBackwardWeightGEMMCPUVerify(*in_packed,
                                          dwei_host,
                                          hx,
                                          *dout_packed,
                                          in_n,
                                          in_vec,
                                          sorted_seq_lens[0],
                                          bidirection,
                                          biased,
                                          n_layer,
                                          batch_size,
                                          hid_vec,
                                          out_h,
                                          mode,
                                          inputMode,
                                          reservespace_host,
                                          workspace_host,
                                          bool(inflags.GetValueInt("use_dropout")));
    }
    else if(mode == miopenLSTM)
    {
        printf("verify lstm bwdwei \n");

        RunLSTMBackwardWeightGEMMCPUVerify(*in_packed,
                                           dwei_host,
                                           hx,
                                           *dout_packed,
                                           in_n,
                                           in_vec,
                                           sorted_seq_lens[0],
                                           bidirection,
                                           biased,
                                           n_layer,
                                           batch_size,
                                           hid_vec,
                                           out_h,
                                           inputMode,
                                           reservespace_host,
                                           workspace_host,
                                           bool(inflags.GetValueInt("use_dropout")));
    }
    else if(mode == miopenGRU)
    {
        printf("verify gru bwdwei \n");

        RunGRUBackwardWeightGEMMCPUVerify(*in_packed,
                                          dwei_host,
                                          hx,
                                          *dout_packed,
                                          in_n,
                                          in_vec,
                                          sorted_seq_lens[0],
                                          bidirection,
                                          biased,
                                          n_layer,
                                          batch_size,
                                          hid_vec,
                                          out_h,
                                          inputMode,
                                          reservespace_host,
                                          workspace_host,
                                          bool(inflags.GetValueInt("use_dropout")));
    }
    else
    {
        printf("illegal RNN mode");
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_bwd_dwei_cpu.bin", dwei_host.data(), dwei_host.size());
    }

    //    TrySaveVerificationCache("bwd_wei", dwei_host);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::RunBackwardDataCPU()
{
    const std::vector<int> in_lens  = GetInputTensorLengthsFromCmdLine();
    const std::vector<int> out_lens = GetOutputTensorLengthsFromCmdLine();
    const int in_vec                = in_lens[2];
    const int out_h                 = out_lens[2];

    const std::vector<int> hid_lens = GetHiddenTensorLengthsFromCmdLine();
    const int n_layer = hid_lens[0], batch_size = hid_lens[1], hid_vec = hid_lens[2];

    const auto batchs = GetBatchesPerTime(sorted_seq_lens);
    std::vector<int> in_n(batchs.begin(), batchs.end());

    bool bidirection, biased;
    int layer;
    miopenRNNMode_t mode;
    miopenRNNAlgo_t algoMode;
    miopenRNNInputMode_t inputMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNBiasMode_t biasMode;
    int hiddenSize;
    miopenDropoutDescriptor_t drop_desc;

    if(inflags.GetValueInt("use_dropout"))
    {
        miopenGetRNNDescriptor_V2(rnnDesc,
                                  &hiddenSize,
                                  &layer,
                                  &drop_desc,
                                  &inputMode,
                                  &dirMode,
                                  &mode,
                                  &biasMode,
                                  &algoMode,
                                  nullptr);
    }
    else
    {
        miopenGetRNNDescriptor(
            rnnDesc, &mode, &algoMode, &inputMode, &dirMode, &biasMode, &hiddenSize, &layer);
    }

    bidirection = (dirMode == miopenRNNbidirection);
    biased      = (biasMode == miopenRNNwithBias);

    std::vector<Tref>* din_packed  = &din_host;
    std::vector<Tgpu>* dout_packed = &dout;

    if(mode == miopenRNNRELU || mode == miopenRNNTANH)
    {
        printf("verify rnn bwddata \n");

        RunRNNBackwardDataGEMMCPUVerify(*din_packed,
                                        wei,
                                        dhy,
                                        dhx_host,
                                        hx,
                                        *dout_packed,
                                        in_n,
                                        in_vec,
                                        sorted_seq_lens[0],
                                        bidirection,
                                        biased,
                                        n_layer,
                                        batch_size,
                                        hid_vec,
                                        out_h,
                                        mode,
                                        inputMode,
                                        reservespace_host,
                                        workspace_host,
                                        bool(inflags.GetValueInt("use_dropout")),
                                        DropoutDesc);
    }
    else if(mode == miopenLSTM)
    {
        printf("verify lstm bwddata \n");

        RunLSTMBackwardDataGEMMCPUVerify(*din_packed,
                                         wei,
                                         dhy,
                                         dhx_host,
                                         hx,
                                         dcy,
                                         dcx_host,
                                         cx,
                                         *dout_packed,
                                         in_n,
                                         in_vec,
                                         sorted_seq_lens[0],
                                         bidirection,
                                         biased,
                                         n_layer,
                                         batch_size,
                                         hid_vec,
                                         out_h,
                                         inputMode,
                                         reservespace_host,
                                         workspace_host,
                                         bool(inflags.GetValueInt("use_dropout")),
                                         DropoutDesc);
    }
    else if(mode == miopenGRU)
    {
        printf("verify gru bwddata \n");

        RunGRUBackwardDataGEMMCPUVerify(*din_packed,
                                        wei,
                                        dhy,
                                        dhx_host,
                                        hx,
                                        *dout_packed,
                                        in_n,
                                        in_vec,
                                        sorted_seq_lens[0],
                                        bidirection,
                                        biased,
                                        n_layer,
                                        batch_size,
                                        hid_vec,
                                        out_h,
                                        inputMode,
                                        reservespace_host,
                                        workspace_host,
                                        bool(inflags.GetValueInt("use_dropout")),
                                        DropoutDesc);
    }
    else
    {
        printf("illegal RNN mode");
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_bwd_din_cpu.bin", din_host.data(), din_host.size());
    }

    //    TrySaveVerificationCache("bwd_dat", din_host);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    auto error = miopen::rms_range(outhost, out);

    Tref tolerance = (sizeof(Tgpu) == 4 ? static_cast<Tref>(1e-6) : static_cast<Tref>(5e-2));

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << std::string("Forward RNN FAILED: ") << error << std::endl;
    }
    else
    {
        printf("Forward RNN Verifies on CPU and GPU\n");
    }

    auto error2 = miopen::rms_range(hy_host, hy);

    if(!std::isfinite(error2) || error2 > tolerance)
    {
        std::cout << std::string("final hidden state FAILED: ") << error2 << std::endl;
    }
    else
    {
        printf("final hidden Verifies on CPU and GPU\n");
    }

    if((inflags.GetValueStr("mode")) == "lstm")
    {
        auto error3 = miopen::rms_range(cy_host, cy);

        if(!std::isfinite(error3) || error3 > tolerance)
        {
            std::cout << std::string("final cell state FAILED: ") << error3 << std::endl;
        }
        else
        {
            printf("final cell Verifies on CPU and GPU\n");
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNSeqDriver<Tgpu, Tref>::VerifyBackward()
{

    if(inflags.GetValueInt("fwdtype") == 1 && inflags.GetValueInt("forw") != 1)
    {
        return miopenStatusSuccess;
    }

    Tref tolerance = (sizeof(Tgpu) == 4 ? static_cast<Tref>(1e-6) : static_cast<Tref>(5e-2));

    //   if(!TryReadVerificationCache("bwd_dat", inputTensor, din_host.data()))
    if((inflags.GetValueInt("forw") & 2) || (inflags.GetValueInt("forw") == 0))
    {
        {
            RunBackwardDataCPU();
        }

        auto error_data = miopen::rms_range(din_host, din);

        if(!std::isfinite(error_data) || error_data > tolerance)
        {
            std::cout << std::string("Backward RNN Data FAILED: ") << error_data << std::endl;
        }
        else
        {
            printf("Backward RNN Data Verifies on CPU and GPU\n");
        }

        auto error_data2 = miopen::rms_range(dhx_host, dhx);

        if(!std::isfinite(error_data2) || error_data2 > tolerance)
        {
            std::cout << std::string("difference at inital hidden state FAILED: ") << error_data2
                      << std::endl;
        }
        else
        {
            printf("initial hidden state Verifies on CPU and GPU\n");
        }

        if((inflags.GetValueStr("mode")) == "lstm")
        {
            auto error_data3 = miopen::rms_range(dcx_host, dcx);

            if(!std::isfinite(error_data3) || error_data3 > tolerance)
            {
                std::cout << std::string("difference at inital cell state FAILED: ") << error_data3
                          << std::endl;
            }
            else
            {
                printf("inital cell state Verifies on CPU and GPU\n");
            }
        }
    }

    //    if(!TryReadVerificationCache("bwd_wei", weightTensor, dwei_host.data()))
    if((inflags.GetValueInt("forw") & 4) || (inflags.GetValueInt("forw") == 0))
    {
        {
            RunBackwardWeightsCPU();
        }

        auto error_weights = miopen::rms_range(dwei_host, dwei);
        if(!std::isfinite(error_weights) || error_weights > tolerance)
        {
            std::cout << std::string("Backward RNN Weights FAILED: ") << error_weights << std::endl;
        }
        else
        {
            printf("Backward RNN Weights Verifies on CPU and GPU\n");
        }
    }

    return miopenStatusSuccess;
}
