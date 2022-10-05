/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_RNN_DRIVER_HPP
#define GUARD_MIOPEN_RNN_DRIVER_HPP

#include "InputFlags.hpp"
#include "rnn_verify_gemm.hpp"
#include "lstm_verify_gemm.hpp"
#include "gru_verify_gemm.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include "random.hpp"
#include <../test/verify.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/rnn.hpp>
#include <miopen/tensor.hpp>
#include <miopen/env.hpp>
#include <numeric>
#include <sstream>
#include <vector>
#include <array>

template <typename Tgpu, typename Tref>
class RNNDriver : public Driver
{
public:
    RNNDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&hiddenTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateRNNDescriptor(&rnnDesc);
        workspace_dev    = nullptr;
        reservespace_dev = nullptr;
        data_type        = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;

        miopenCreateDropoutDescriptor(&DropoutDesc);
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetHiddenTensorLengthsFromCmdLine();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();
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
    ~RNNDriver() override
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(weightTensor);
        miopenDestroyTensorDescriptor(hiddenTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyRNNDescriptor(rnnDesc);
    }

private:
    InputFlags inflags;

    std::vector<miopenTensorDescriptor_t> inputTensors;
    std::vector<miopenTensorDescriptor_t> outputTensors;
    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t hiddenTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t outputTensor;

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
    std::vector<prngStates> dropout_states_host;

    miopenRNNDescriptor_t rnnDesc;

    int batchsize;
    int adjustedSeqLen;
    std::vector<int> batchseq;

    miopenDropoutDescriptor_t DropoutDesc;
    float dropout_rate;
    unsigned long long dropout_seed;

    //    std::string GetVerificationCacheFileName() const;
    //    bool TryReadVerificationCache(const std::string& file_name,
    //                                  miopenTensorDescriptor_t& tensorDesc,
    //                                  Tgpu* data) const;
    //    void TrySaveVerificationCache(const std::string& file_name, std::vector<Tgpu>& data)
    //    const;
};

static inline bool CheckGuard(const int& in_h,
                              const int& out_h,
                              const int& hy_d,
                              const int& hy_n,
                              const int& hy_h,
                              const miopenRNNDirectionMode_t& dirMode,
                              const miopenRNNInputMode_t& inputMode)
{
    return (in_h == 0 || out_h == 0 || hy_d == 0 || hy_n == 0 || hy_h == 0 ||
            out_h != ((dirMode + 1) * hy_h) || (inputMode == miopenRNNskip && in_h != hy_h));
}

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();

    for(int i = 0; i < in_len.size() - 1; i++)
    {
        std::array<int, 2> in_lens = {{in_len[i], in_len.back()}};
        miopenCreateTensorDescriptor(&inputTensor);
        miopenSetTensorDescriptor(inputTensor, data_type, 2, in_lens.data(), nullptr);
        inputTensors.push_back(inputTensor);

        std::array<int, 2> out_lens = {{in_len[i], out_len[0]}};
        miopenCreateTensorDescriptor(&outputTensor);
        miopenSetTensorDescriptor(outputTensor, data_type, 2, out_lens.data(), nullptr);
        outputTensors.push_back(outputTensor);
    }

    std::array<int, 3> hid_lens = {{hid_len[0], in_len[0], hid_len[1]}};
    miopenSetTensorDescriptor(hiddenTensor, data_type, 3, hid_lens.data(), nullptr);

    SetRNNDescriptorFromCmdLineArgs();
    miopenGetRNNParamsDescriptor(handle, rnnDesc, inputTensor, weightTensor, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run only Forward RNN == 1 or only Backward Data RNN == 2, Backward "
                         "Weights = 4 or both == 0 (Default=0)",
                         "int");
    inflags.AddInputFlag("num_layer", 'l', "1", "Number of hidden stacks (Default=1)", "int");
    inflags.AddInputFlag(
        "seq_len", 'k', "1", "Number of iterations to unroll over (Default=10)", "int");
    inflags.AddInputFlag(
        "bidirection", 'r', "0", "uni- or bi-direction, default uni- (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "4", "Mini-batch size (Default=4)", "vector");
    inflags.AddInputFlag("hid_h", 'H', "32", "Hidden State Length (Default=32)", "int");
    inflags.AddInputFlag("in_h", 'W', "32", "Input Length (Default=32)", "int");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    /*
    // DL: This is disabled below, so I'm turninig it off here
    inflags.AddInputFlag("verification_cache",
                         'C',
                         "",
                         "Use specified directory to cache verification data. Off by default.",
                         "string");
    */
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    /*  // DL: These have not been implemented. Removing them for now.
        inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");
        inflags.AddInputFlag("weights", 'e', "", "Input weights filename (Default=)", "string");*/
    inflags.AddInputFlag("bias", 'b', "", "Use Bias (Default=0)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "tanh", "RNN Mode (relu, tanh, lstm, gru) (Default=tanh)", "str");
    inflags.AddInputFlag("inputmode", 'p', "0", "linear == 0 or skip == 1, (Default=0)", "int");
    inflags.AddInputFlag("rnnalgo", 'a', "0", "default, fundamental (Default=0)", "int");
    inflags.AddInputFlag("fwdtype",
                         'c',
                         "0",
                         "RNN forward being training or inference, Default training (Default=0)",
                         "int");
    inflags.AddInputFlag("datatype", 'f', "1", "16-bit or 32-bit fp (Default=1)", "int");

    inflags.AddInputFlag(
        "use_dropout", 'U', "0", "Use dropout: 1; Not use dropout: 0 (Default=0)", "int");
    inflags.AddInputFlag("dropout", 'P', "0.0", "Dropout rate (Default=0.0)", "float");
    inflags.AddInputFlag(
        "seed_low", 'L', "0", "Least significant 32 bits of seed (Default=0)", "int");
    inflags.AddInputFlag(
        "seed_high", 'M', "0", "Most significant 32 bits of seed (Default=0)", "int");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int nseq = inflags.GetValueInt("seq_len");
    int in_h = inflags.GetValueInt("in_h");
    std::vector<int> in_n(nseq, 0);
    std::string batchstr = inflags.GetValueStr("batchsize");

    std::stringstream ss(batchstr);
    int cont = 0;

    int element;
    while(ss >> element)
    {
        if(cont >= nseq)
        {
            printf("Length of data sequence is longer than required unrolled time sequence "
                   "provided.\n"
                   "The data sequence will be truncated to match unrolled time sequence.\n");
            break;
        }

        if(ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }

        if(cont > 0 && in_n[cont] > in_n[cont - 1])
        {
            printf("Incorrect input batch size at time %d\n", cont);
            return std::vector<int>({0});
        }
        else
        {
            in_n[cont] = element;
            cont++;
        }
    }

    adjustedSeqLen = nseq;

    if(nseq > cont)
    {
        printf("length of data sequence == %d is short than time sequence == %d, padding the rest "
               "of data sequence with %d\n",
               cont,
               nseq,
               in_n[cont - 1]);
        for(int i = cont; i < nseq; i++)
        {
            in_n[i] = in_n[cont - 1];
        }
    }

    in_n.push_back(in_h);

    return in_n;
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNDriver<Tgpu, Tref>::GetHiddenTensorLengthsFromCmdLine()
{
    int hid_h = inflags.GetValueInt("hid_h");
    int hid_l = inflags.GetValueInt("num_layer");
    if((inflags.GetValueInt("bidirection")) == 1)
        hid_l *= 2;

    return std::vector<int>({hid_l, hid_h});
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNDriver<Tgpu, Tref>::GetWeightTensorLengthsFromCmdLine()
{
    int wei_ih = inflags.GetValueInt("in_h");
    int wei_hh = inflags.GetValueInt("hid_h");
    int wei_oh;
    int wei_l  = inflags.GetValueInt("num_layer");
    int wei_bi = 1;
    if((inflags.GetValueInt("bidirection")) == 1)
        wei_bi = 2;
    wei_oh = wei_hh * wei_bi;

    int wei_sc = 1;
    if((inflags.GetValueStr("mode")) == "lstm")
        wei_sc = 4;
    else if((inflags.GetValueStr("mode")) == "gru")
        wei_sc = 3;

    return std::vector<int>({wei_bi, wei_l, wei_ih, wei_hh, wei_oh, wei_sc});
}

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::SetRNNDescriptorFromCmdLineArgs()
{

    int layer  = inflags.GetValueInt("num_layer");
    int wei_hh = inflags.GetValueInt("hid_h"); // hidden state size

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
        size_t states_size = statesSizeInBytes / sizeof(prngStates);

#if MIOPEN_BACKEND_OPENCL
        cl_context ctx;

        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
        uint32_t ctx = 0;
#endif

        dropout_states_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, states_size, sizeof(prngStates)));

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
                                  wei_hh,
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
            rnnDesc, wei_hh, layer, inMode, directionMode, mode, biasMode, algo, data_type);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> RNNDriver<Tgpu, Tref>::GetOutputTensorLengthsFromCmdLine()
{
    int hid_h = inflags.GetValueInt("hid_h");
    int bi    = (inflags.GetValueInt("bidirection") == 1) ? 2 : 1;

    int out_h = hid_h * bi;
    return std::vector<int>({out_h});
}

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{

    size_t in_sz  = 0;
    size_t out_sz = 0;
    size_t wei_sz = 0;
    size_t hy_sz  = 0;
    size_t workSpace_sz;
    size_t reserveSpace_sz;

    miopenGetRNNInputTensorSize(GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), &in_sz);
    miopenGetRNNInputTensorSize(
        GetHandle(), rnnDesc, adjustedSeqLen, outputTensors.data(), &out_sz);
    miopenGetRNNHiddenTensorSize(GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), &hy_sz);
    miopenGetRNNWorkspaceSize(
        GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), &workSpace_sz);
    miopenGetRNNTrainingReserveSize(
        GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), &reserveSpace_sz);
    miopenGetRNNParamsSize(GetHandle(), rnnDesc, inputTensors[0], &wei_sz, data_type);

    in_sz /= sizeof(Tgpu);
    out_sz /= sizeof(Tgpu);
    hy_sz /= sizeof(Tgpu);
    wei_sz /= sizeof(Tgpu);
    workSpace_sz /= sizeof(Tgpu);
    reserveSpace_sz = (reserveSpace_sz + sizeof(Tgpu) - 1) / sizeof(Tgpu);

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    in_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    hx_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
    out_dev          = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    wei_dev          = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
    cx_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
    workspace_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpace_sz, sizeof(Tgpu)));
    reservespace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, reserveSpace_sz, sizeof(Tgpu)));

    if(inflags.GetValueInt("forw") != 2)
    {
        hy_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
        cy_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
    }

    if(inflags.GetValueInt("forw") != 1)
    {
        din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
        dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        dhx_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
        dcx_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
        dhy_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
        dcy_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(Tgpu)));
    }

    in  = std::vector<Tgpu>(in_sz);
    hx  = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
    wei = std::vector<Tgpu>(wei_sz);
    out = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    cx  = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));

    if(inflags.GetValueInt("forw") != 2)
    {
        hy      = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        cy      = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        hy_host = std::vector<Tref>(hy_sz, static_cast<Tref>(0));
        cy_host = std::vector<Tref>(hy_sz, static_cast<Tref>(0));
    }

    workspace      = std::vector<Tgpu>(workSpace_sz, static_cast<Tgpu>(0));
    reservespace   = std::vector<Tgpu>(reserveSpace_sz, static_cast<Tgpu>(0));
    outhost        = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    workspace_host = std::vector<Tref>(workSpace_sz, static_cast<Tref>(0));

    int nseq              = inflags.GetValueInt("seq_len");
    std::vector<int> in_n = GetInputTensorLengthsFromCmdLine();
    std::size_t inputBatchLenSum;
    inputBatchLenSum = std::accumulate(in_n.begin(), in_n.begin() + nseq, 0);

    int hid_h = inflags.GetValueInt("hid_h");
    int layer = inflags.GetValueInt("num_layer");
    int bidir = inflags.GetValueInt("bidirection");

    reserveSpace_sz =
        2 *
        (inflags.GetValueStr("mode") == "lstm" ? 6
                                               : (inflags.GetValueStr("mode") == "gru" ? 4 : 1)) *
        layer * inputBatchLenSum * hid_h * (bidir + 1);
    if(inflags.GetValueInt("use_dropout"))
    {
        reserveSpace_sz += (layer - 1) * inputBatchLenSum * hid_h * (bidir + 1);
        reserveSpace_sz *= sizeof(Tref);
        reserveSpace_sz += (layer - 1) * inputBatchLenSum * hid_h * (bidir + 1);
        reserveSpace_sz = (reserveSpace_sz + sizeof(Tref) - 1) / sizeof(Tref);
    }
    reservespace_host = std::vector<Tref>(reserveSpace_sz, static_cast<Tref>(0));

    if(inflags.GetValueInt("forw") != 1)
    {
        din       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        dwei      = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
        dout      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        dhx       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        dcx       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        dhy       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        dcy       = std::vector<Tgpu>(hy_sz, static_cast<Tgpu>(0));
        din_host  = std::vector<Tref>(in_sz, static_cast<Tref>(0));
        dwei_host = std::vector<Tref>(wei_sz, static_cast<Tref>(0));
        dhx_host  = std::vector<Tref>(hy_sz, static_cast<Tref>(0));
        dcx_host  = std::vector<Tref>(hy_sz, static_cast<Tref>(0));
    }

    /*  // Not implemented.
        std::string inFileName  = inflags.GetValueStr("in_data");
        std::string weiFileName = inflags.GetValueStr("weights");*/

    // Unless seed is persistent between runs validation using cache stored in file is impossible.
    srand(0);
    double scale = 0.01;

    /*    bool dataRead = false;
        bool weiRead = false;
        if(!inFileName.empty())
        {
            std::cerr << "Loading input data from file is a feature which has not been implemented
       in MIOpenDriver." << std::endl;
            // Not implemented.
            //dataRead = readBufferFromFile(in.data(), in_sz, inFileName.c_str());
        }
    */

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = static_cast<Tgpu>((static_cast<double>(scale * GET_RAND()) * (1.0 / RAND_MAX)));
    }

    for(int i = 0; i < hy_sz; i++)
    {
        hx[i] = static_cast<Tgpu>((scale * static_cast<double>(GET_RAND()) * (1.0 / RAND_MAX)));
    }

    if((inflags.GetValueStr("mode")) == "lstm")
    {
        for(int i = 0; i < hy_sz; i++)
        {
            cx[i] = static_cast<Tgpu>((scale * static_cast<double>(GET_RAND()) * (1.0 / RAND_MAX)));
        }
    }

    if(inflags.GetValueInt("forw") != 1)
    {
        for(int i = 0; i < out_sz; i++)
        {
            dout[i] =
                static_cast<Tgpu>((scale * static_cast<double>(GET_RAND()) * (1.0 / RAND_MAX)));
        }

        for(int i = 0; i < hy_sz; i++)
        {
            dhy[i] =
                static_cast<Tgpu>((scale * static_cast<double>(GET_RAND()) * (1.0 / RAND_MAX)));
        }

        if((inflags.GetValueStr("mode")) == "lstm")
        {
            for(int i = 0; i < hy_sz; i++)
            {
                dcy[i] =
                    static_cast<Tgpu>((scale * static_cast<double>(GET_RAND()) * (1.0 / RAND_MAX)));
            }
        }
    }

    /*
        if(!weiFileName.empty())
        {
            std::cerr << "Loading weights from file is a feature which has not been implemented in
       MIOpenDriver." << std::endl;
            // Not implemented.
            // weiRead = readBufferFromFile(wei.data(), wei_sz, weiFileName.c_str());
        }
    */

    for(int i = 0; i < wei_sz; i++)
    {
        wei[i] =
            static_cast<Tgpu>((scale * static_cast<double>((GET_RAND()) * (1.0 / RAND_MAX) - 0.5)));
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_in.bin", in.data(), in_sz);
        dumpBufferToFile("dump_wei.bin", wei.data(), wei_sz);
    }

#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
#define CL_SUCCESS 0
    int status;
#endif

    status = in_dev->ToGPU(q, in.data());
    status |= wei_dev->ToGPU(q, wei.data());
    status |= out_dev->ToGPU(q, out.data());
    status |= cx_dev->ToGPU(q, cx.data());
    status |= hx_dev->ToGPU(q, hx.data());
    status |= workspace_dev->ToGPU(q, workspace.data());
    status |= reservespace_dev->ToGPU(q, reservespace.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    if(inflags.GetValueInt("forw") != 2)
    {
        status = hy_dev->ToGPU(q, hy.data());
        status |= cy_dev->ToGPU(q, cy.data());

        if(status != CL_SUCCESS)
            printf("Error copying data to GPU\n");
    }

    if(inflags.GetValueInt("forw") != 1)
    {
        status = din_dev->ToGPU(q, din.data());
        status |= dwei_dev->ToGPU(q, dwei.data());
        status |= dout_dev->ToGPU(q, dout.data());
        status |= dhx_dev->ToGPU(q, dhx.data());
        status |= dcx_dev->ToGPU(q, dcx.data());
        status |= dhy_dev->ToGPU(q, dhy.data());
        status |= dcy_dev->ToGPU(q, dcy.data());

        if(status != CL_SUCCESS)
            printf("Error copying data to GPU\n");
    }

    return miopenStatusSuccess;
}

#include <array>
#include <initializer_list>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::RunForwardGPU()
{

    if(inflags.GetValueInt("forw") != 0 && !(inflags.GetValueInt("forw") & 1))
        return miopenStatusSuccess;

    Timer t;
    float wl_time_forward = 0.0;
    float kl_time_forward = 0.0;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        std::fill(out.begin(), out.end(), static_cast<Tgpu>(0));
        out_dev->ToGPU(GetStream(), out.data());

        if(i > 0)
        {
            std::fill(reservespace.begin(), reservespace.end(), 0.);
            std::fill(workspace.begin(), workspace.end(), 0.);

            workspace_dev->ToGPU(q, workspace.data());
            reservespace_dev->ToGPU(q, reservespace.data());
        }

        START_TIME
        if(inflags.GetValueInt("fwdtype") == 0)
        {
            miopenRNNForwardTraining(GetHandle(),
                                     rnnDesc,
                                     adjustedSeqLen,
                                     inputTensors.data(),
                                     in_dev->GetMem(),
                                     hiddenTensor,
                                     hx_dev->GetMem(),
                                     hiddenTensor,
                                     cx_dev->GetMem(),
                                     weightTensor,
                                     wei_dev->GetMem(),
                                     outputTensors.data(),
                                     out_dev->GetMem(),
                                     hiddenTensor,
                                     hy_dev->GetMem(),
                                     hiddenTensor,
                                     cy_dev->GetMem(),
                                     workspace_dev->GetMem(),
                                     workspace_dev->GetSize(),
                                     reservespace_dev->GetMem(),
                                     reservespace_dev->GetSize());
        }
        else if(inflags.GetValueInt("fwdtype") == 1)
        {
            if(inflags.GetValueInt("forw") != 1)
            {
                printf("Warning: Inference type is only valid for Forward RNN! \n");
            }

            miopenRNNForwardInference(GetHandle(),
                                      rnnDesc,
                                      adjustedSeqLen,
                                      inputTensors.data(),
                                      in_dev->GetMem(),
                                      hiddenTensor,
                                      hx_dev->GetMem(),
                                      hiddenTensor,
                                      cx_dev->GetMem(),
                                      weightTensor,
                                      wei_dev->GetMem(),
                                      outputTensors.data(),
                                      out_dev->GetMem(),
                                      hiddenTensor,
                                      hy_dev->GetMem(),
                                      hiddenTensor,
                                      cy_dev->GetMem(),
                                      workspace_dev->GetMem(),
                                      workspace_dev->GetSize());
        }
        miopen::deref(GetHandle()).Finish();
        STOP_TIME

        if(i > 0 || inflags.GetValueInt("iter") == 1)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            // printf("wall time: %f\n", t.gettime_ms());
            wl_time_forward += t.gettime_ms();
            kl_time_forward += time;
        }
    }

    if(inflags.GetValueInt("time") == 1)
    {
        int n_iter = inflags.GetValueInt("iter") > 1 ? inflags.GetValueInt("iter") - 1
                                                     : inflags.GetValueInt("iter");
        printf("GPU Kernel Time Forward RNN Elapsed: %f ms\n", kl_time_forward / n_iter);
    }

    if(WALL_CLOCK)
    {
        int n_iter = inflags.GetValueInt("iter") > 1 ? inflags.GetValueInt("iter") - 1
                                                     : inflags.GetValueInt("iter");
        printf("Wall-clock Time Forward RNN Elapsed: %f ms\n", wl_time_forward / n_iter);
    }

    out_dev->FromGPU(GetStream(), out.data());
    hy_dev->FromGPU(GetStream(), hy.data());
    cy_dev->FromGPU(GetStream(), cy.data());
    reservespace_dev->FromGPU(GetStream(), reservespace.data());

    /*
       if(inflags.GetValueInt("dump_output"))
       {
       dumpBufferToFile("dump_fwd_out_gpu.bin", out.data(), out.size());
       }
       */

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(inflags.GetValueInt("forw") != 0 && !(inflags.GetValueInt("forw") & 1))
        return miopenStatusSuccess;
    std::vector<int> in_n    = GetInputTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();
    std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    int in_h                 = in_n.back();
    int out_h                = out_len[0];
    int hy_d = hid_len[0], hy_n = in_n[0], hy_h = hid_len[1];
    in_n.pop_back();

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

    if(CheckGuard(in_h, out_h, hy_d, hy_n, hy_h, dirMode, inputMode))
    {
        return miopenStatusBadParm;
    }

    if(mode == miopenRNNRELU || mode == miopenRNNTANH)
    {
        printf("verify rnn fwd \n");
        RunRNNForwardGEMMCPUVerify(GetHandle(),
                                   in,
                                   wei,
                                   hy_host,
                                   hx,
                                   outhost,
                                   in_n,
                                   in_h,
                                   adjustedSeqLen,
                                   bidirection,
                                   biased,
                                   hy_d,
                                   hy_n,
                                   hy_h,
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
                                    in,
                                    wei,
                                    hy_host,
                                    hx,
                                    cy_host,
                                    cx,
                                    outhost,
                                    in_n,
                                    in_h,
                                    adjustedSeqLen,
                                    bidirection,
                                    biased,
                                    hy_d,
                                    hy_n,
                                    hy_h,
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
                                   in,
                                   wei,
                                   hy_host,
                                   hx,
                                   outhost,
                                   in_n,
                                   in_h,
                                   adjustedSeqLen,
                                   bidirection,
                                   biased,
                                   hy_d,
                                   hy_n,
                                   hy_h,
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
int RNNDriver<Tgpu, Tref>::RunBackwardGPU()
{
    int ret = 0;

    if(inflags.GetValueInt("forw") == 1)
        return ret; // forward only

    if((inflags.GetValueInt("forw") & 2) || (inflags.GetValueInt("forw") == 0))
    {
        Timer t;
        float wl_time_backward_data = 0.0;
        float kl_time_backward_data = 0.0;

        workspace_dev->ToGPU(q, workspace.data());

        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            START_TIME
            ret = miopenRNNBackwardData(GetHandle(),
                                        rnnDesc,
                                        adjustedSeqLen,
                                        outputTensors.data(),
                                        out_dev->GetMem(),
                                        outputTensors.data(),
                                        dout_dev->GetMem(),
                                        hiddenTensor,
                                        dhy_dev->GetMem(),
                                        hiddenTensor,
                                        dcy_dev->GetMem(),
                                        weightTensor,
                                        wei_dev->GetMem(),
                                        hiddenTensor,
                                        hx_dev->GetMem(),
                                        hiddenTensor,
                                        cx_dev->GetMem(),
                                        inputTensors.data(),
                                        din_dev->GetMem(),
                                        hiddenTensor,
                                        dhx_dev->GetMem(),
                                        hiddenTensor,
                                        dcx_dev->GetMem(),
                                        workspace_dev->GetMem(),
                                        workspace_dev->GetSize(),
                                        reservespace_dev->GetMem(),
                                        reservespace_dev->GetSize());
            miopen::deref(GetHandle()).Finish();
            STOP_TIME
            if(i > 0 || inflags.GetValueInt("iter") == 1)
            {
                float time = 0.0;
                miopenGetKernelTime(GetHandle(), &time);
                wl_time_backward_data += t.gettime_ms();
                kl_time_backward_data += time;
            }
        }

        if(inflags.GetValueInt("time") == 1)
        {
            int n_iter = inflags.GetValueInt("iter") > 1 ? inflags.GetValueInt("iter") - 1
                                                         : inflags.GetValueInt("iter");
            printf("GPU Kernel Time Backward Data RNN Elapsed: %f ms\n",
                   kl_time_backward_data / n_iter);
        }

        if(WALL_CLOCK)
        {
            int n_iter = inflags.GetValueInt("iter") > 1 ? inflags.GetValueInt("iter") - 1
                                                         : inflags.GetValueInt("iter");
            printf("Wall-clock Time Backward Data RNN Elapsed: %f ms\n",
                   wl_time_backward_data / n_iter);
        }

        din_dev->FromGPU(GetStream(), din.data());
        dhx_dev->FromGPU(GetStream(), dhx.data());
        dcx_dev->FromGPU(GetStream(), dcx.data());
        workspace_dev->FromGPU(GetStream(), workspace.data());
    }

    if((inflags.GetValueInt("forw") & 4) || (inflags.GetValueInt("forw") == 0))
    {
        Timer t;
        float wl_time_backward_weight = 0.0;
        float kl_time_backward_weight = 0.0;

        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            START_TIME
            ret = miopenRNNBackwardWeights(GetHandle(),
                                           rnnDesc,
                                           adjustedSeqLen,
                                           inputTensors.data(),
                                           in_dev->GetMem(),
                                           hiddenTensor,
                                           hx_dev->GetMem(),
                                           outputTensors.data(),
                                           dout_dev->GetMem(),
                                           weightTensor,
                                           dwei_dev->GetMem(),
                                           workspace_dev->GetMem(),
                                           workspace_dev->GetSize(),
                                           reservespace_dev->GetMem(),
                                           reservespace_dev->GetSize());
            miopen::deref(GetHandle()).Finish();
            STOP_TIME
            if(i > 0 || inflags.GetValueInt("iter") == 1)
            {
                float time = 0.0;
                miopenGetKernelTime(GetHandle(), &time);
                wl_time_backward_weight += t.gettime_ms();
                kl_time_backward_weight += time;
            }
        }

        if(inflags.GetValueInt("time") == 1)
        {
            int n_iter = inflags.GetValueInt("iter") > 1 ? inflags.GetValueInt("iter") - 1
                                                         : inflags.GetValueInt("iter");
            printf("GPU Kernel Time Backward Weights RNN Elapsed: %f ms\n",
                   kl_time_backward_weight / n_iter);
        }

        if(WALL_CLOCK)
        {
            int n_iter = inflags.GetValueInt("iter") > 1 ? inflags.GetValueInt("iter") - 1
                                                         : inflags.GetValueInt("iter");
            printf("Wall-clock Time Backward Weights RNN Elapsed: %f ms\n",
                   wl_time_backward_weight / n_iter);
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
int RNNDriver<Tgpu, Tref>::RunBackwardWeightsCPU()
{
    std::vector<int> in_n    = GetInputTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();
    std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    int in_h                 = in_n.back();
    int out_h                = out_len[0];
    int hy_d = hid_len[0], hy_n = in_n[0], hy_h = hid_len[1];
    in_n.pop_back();

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

    if(CheckGuard(in_h, out_h, hy_d, hy_n, hy_h, dirMode, inputMode))
    {
        return miopenStatusBadParm;
    }

    if(mode == miopenRNNRELU || mode == miopenRNNTANH)
    {
        printf("verify rnn bwdwei \n");

        RunRNNBackwardWeightGEMMCPUVerify(in,
                                          dwei_host,
                                          hx,
                                          dout,
                                          in_n,
                                          in_h,
                                          adjustedSeqLen,
                                          bidirection,
                                          biased,
                                          hy_d,
                                          hy_n,
                                          hy_h,
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

        RunLSTMBackwardWeightGEMMCPUVerify(in,
                                           dwei_host,
                                           hx,
                                           dout,
                                           in_n,
                                           in_h,
                                           adjustedSeqLen,
                                           bidirection,
                                           biased,
                                           hy_d,
                                           hy_n,
                                           hy_h,
                                           out_h,
                                           inputMode,
                                           reservespace_host,
                                           workspace_host,
                                           bool(inflags.GetValueInt("use_dropout")));
    }
    else if(mode == miopenGRU)
    {
        printf("verify gru bwdwei \n");

        RunGRUBackwardWeightGEMMCPUVerify(in,
                                          dwei_host,
                                          hx,
                                          dout,
                                          in_n,
                                          in_h,
                                          adjustedSeqLen,
                                          bidirection,
                                          biased,
                                          hy_d,
                                          hy_n,
                                          hy_h,
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
int RNNDriver<Tgpu, Tref>::RunBackwardDataCPU()
{
    std::vector<int> in_n    = GetInputTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();
    std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    int in_h                 = in_n.back();
    int out_h                = out_len[0];
    int hy_d = hid_len[0], hy_n = in_n[0], hy_h = hid_len[1];
    in_n.pop_back();

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

    if(CheckGuard(in_h, out_h, hy_d, hy_n, hy_h, dirMode, inputMode))
    {
        return miopenStatusBadParm;
    }

    if(mode == miopenRNNRELU || mode == miopenRNNTANH)
    {
        printf("verify rnn bwddata \n");

        RunRNNBackwardDataGEMMCPUVerify(din_host,
                                        wei,
                                        dhy,
                                        dhx_host,
                                        hx,
                                        out,
                                        dout,
                                        in_n,
                                        in_h,
                                        adjustedSeqLen,
                                        bidirection,
                                        biased,
                                        hy_d,
                                        hy_n,
                                        hy_h,
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

        RunLSTMBackwardDataGEMMCPUVerify(din_host,
                                         wei,
                                         dhy,
                                         dhx_host,
                                         hx,
                                         dcy,
                                         dcx_host,
                                         cx,
                                         out,
                                         dout,
                                         in_n,
                                         in_h,
                                         adjustedSeqLen,
                                         bidirection,
                                         biased,
                                         hy_d,
                                         hy_n,
                                         hy_h,
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

        RunGRUBackwardDataGEMMCPUVerify(din_host,
                                        wei,
                                        dhy,
                                        dhx_host,
                                        hx,
                                        out,
                                        dout,
                                        in_n,
                                        in_h,
                                        adjustedSeqLen,
                                        bidirection,
                                        biased,
                                        hy_d,
                                        hy_n,
                                        hy_h,
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

/*
template <typename Tgpu, typename Tref>
std::string RNNDriver<Tgpu, Tref>::GetVerificationCacheFileName() const
{
    std::ostringstream ss;

    miopenRNNMode_t mode;
    int seqLength, layer, bidir;
    miopenGetRNNDescriptor(
        rnnDesc, &mode, &seqLength, &layer, &bidir);

    const auto inputDesc = GetTensorLengths(const_cast<miopenTensorDescriptor_t&>(inputTensor));
    const auto weiDesc   = GetTensorLengths(const_cast<miopenTensorDescriptor_t&>(weightTensor));
    const auto outDesc   = GetTensorLengths(const_cast<miopenTensorDescriptor_t&>(outputTensor));

    ss << inputDesc[1]        //_n_inputs
       << "x" << inputDesc[2] //_in_height
       << "x" << inputDesc[3] //_in_width
       << "x" << weiDesc[2]   //_kernel_size1
       << "x" << weiDesc[3]   //_kernel_size0
       << "x" << weiDesc[0]   //_n_outputs
       << "x" << outDesc[2]   //_out_height
       << "x" << outDesc[3]   //_out_width
       << "x" << inputDesc[0] //_batch_sz
       << "_" << weiDesc[1] << "x" << pad_h << "x" << pad_w << "x" << u << "x" << v << "x" << sx
       << "x" << sy << "x" << inflags.GetValueInt("pad_val");

    return ss.str();
}

template <typename Tgpu, typename Tref>
bool RNNDriver<Tgpu, Tref>::TryReadVerificationCache(const std::string& file_name,
                                             miopenTensorDescriptor_t& tensorDesc,
                                             Tgpu* data) const
{
    const auto verification_cache_path = inflags.GetValueStr("verification_cache");

    if(!verification_cache_path.empty())
    {
        const auto file_path =
            verification_cache_path + "/" + file_name + "_" + GetVerificationCacheFileName();
        if(std::ifstream(file_path).good())
        {
            if(readBufferFromFile(data, GetTensorSize(tensorDesc), file_path.c_str()))
            {
                return true;
            }
        }
    }

    return false;
}

template <typename Tgpu, typename Tref>
void RNNDriver<Tgpu, Tref>::TrySaveVerificationCache(const std::string& file_name,
                                             std::vector<Tgpu>& data) const
{
    const auto verification_cache_path = inflags.GetValueStr("verification_cache");
    if(!verification_cache_path.empty())
    {
        const auto file_path =
            verification_cache_path + "/" + file_name + "_" + GetVerificationCacheFileName();
        dumpBufferToFile(file_path.c_str(), data.data(), data.size());
    }
}
*/

template <typename Tgpu, typename Tref>
int RNNDriver<Tgpu, Tref>::VerifyForward()
{
    std::vector<int> in_n    = GetInputTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();
    std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    int in_h                 = in_n.back();
    int out_h                = out_len[0];
    int hy_d = hid_len[0], hy_n = in_n[0], hy_h = hid_len[1];

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

    if(CheckGuard(in_h, out_h, hy_d, hy_n, hy_h, dirMode, inputMode))
    {
        printf("Bad Parameters! Verification FAILED\n");
        return miopenStatusBadParm;
    }

    //   if(!TryReadVerificationCache("fwd_out", outputTensor, outhost.data()))
    {
        RunForwardCPU();
    }

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
int RNNDriver<Tgpu, Tref>::VerifyBackward()
{
    std::vector<int> in_n    = GetInputTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();
    std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    int in_h                 = in_n.back();
    int out_h                = out_len[0];
    int hy_d = hid_len[0], hy_n = in_n[0], hy_h = hid_len[1];

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

    if(CheckGuard(in_h, out_h, hy_d, hy_n, hy_h, dirMode, inputMode))
    {
        printf("Bad Parameters! Verification FAILED\n");
        return miopenStatusBadParm;
    }

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

#endif // GUARD_MIOPEN_RNN_DRIVER_HPP
