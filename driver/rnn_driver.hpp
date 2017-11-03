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
#include "rnn_verify.hpp"
#include "rnn_verify_gemm.hpp"
#include "lstm_verify.hpp"
#include "lstm_verify_gemm.hpp"
#include "gru_verify.hpp"
#include "gru_verify_gemm.hpp"
#include "driver.hpp"
#include "mloConvHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include <../test/verify.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/env.hpp>
#include <numeric>
#include <sstream>
#include <vector>
#include <array>

// MIOPEN_DECLARE_ENV_VAR(MIOPEN_DRIVER_PAD_BUFFERS_2M)

/*
template <typename T>
void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        printf("Wrote output to file %s\n", fileName);
    }
    else
    {
        printf("Could not open file %s for writing\n", fileName);
    }
}

template <typename T>
bool readBufferFromFile(T* data, size_t dataNumItems, const char* fileName)
{
    std::ifstream infile(fileName, std::ios::binary);
    if(infile)
    {
        infile.read(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        infile.close();
        printf("Read data from input file %s\n", fileName);
        return true;
    }
    else
    {
        printf("Could not open file %s for reading\n", fileName);
        return false;
    }
}
*/

template <typename T>
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
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetHiddenTensorLengthsFromCmdLine();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();
    std::vector<int> GetOutputTensorLengthsFromCmdLine();

    int SetRNNDescriptorFromCmdLineArgs();
    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();
    int RunBackwardGPU();
    int RunBackwardDataCPU();
    int RunBackwardWeightsCPU();
    int VerifyBackward();
    int VerifyForward();
    ~RNNDriver()
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

    std::vector<T> in;
    std::vector<T> din;
    std::vector<T> wei;
    std::vector<T> dwei;
    std::vector<T> out;
    std::vector<T> dout;
    std::vector<T> hx;
    std::vector<T> cx;
    std::vector<T> hy;
    std::vector<T> cy;
    std::vector<T> dhx;
    std::vector<T> dcx;
    std::vector<T> dhy;
    std::vector<T> dcy;
    std::vector<T> workspace;
    std::vector<T> reservespace;
    std::vector<T> outhost;
    std::vector<T> workspace_host;
    std::vector<T> reservespace_host;
    std::vector<T> din_host;
    std::vector<T> dwei_host;
    std::vector<T> hy_host;
    std::vector<T> cy_host;
    std::vector<T> dhx_host;
    std::vector<T> dcx_host;

    miopenRNNDescriptor_t rnnDesc;

    int batchsize;
    int adjustedSeqLen;
    std::vector<int> batchseq;

    //    std::string GetVerificationCacheFileName() const;
    //    bool TryReadVerificationCache(const std::string& file_name,
    //                                  miopenTensorDescriptor_t& tensorDesc,
    //                                  T* data) const;
    //    void TrySaveVerificationCache(const std::string& file_name, std::vector<T>& data) const;
};

template <typename T>
int RNNDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename T>
int RNNDriver<T>::GetandSetData()
{
    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> hid_len = GetHiddenTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();

    for(int i = 0; i < in_len.size() - 1; i++)
    {
        std::array<int, 2> in_lens = {in_len[i], in_len.back()};
        miopenCreateTensorDescriptor(&inputTensor);
        miopenSetTensorDescriptor(inputTensor, miopenFloat, 2, in_lens.data(), nullptr);
        inputTensors.push_back(inputTensor);

        std::array<int, 2> out_lens = {in_len[i], out_len[0]};
        miopenCreateTensorDescriptor(&outputTensor);
        miopenSetTensorDescriptor(outputTensor, miopenFloat, 2, out_lens.data(), nullptr);
        outputTensors.push_back(outputTensor);
    }

    std::array<int, 3> hid_lens = {hid_len[0], in_len[0], hid_len[1]};
    miopenSetTensorDescriptor(hiddenTensor, miopenFloat, 3, hid_lens.data(), nullptr);

    SetRNNDescriptorFromCmdLineArgs();
    miopenGetRNNParamsDescriptor(handle, rnnDesc, inputTensor, weightTensor, miopenFloat);

    return miopenStatusSuccess;
}

template <typename T>
int RNNDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward RNN (Default=0)", "int");
    inflags.AddInputFlag("num_layer", 'l', "1", "Number of hidden stacks (Default=1)", "int");
    inflags.AddInputFlag(
        "seq_len", 'k', "1", "Number of iterations to unroll over (Default=10)", "int");
    inflags.AddInputFlag(
        "bidirection", 'r', "0", "uni- or bi-direction, default uni- (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "4", "Mini-batch size (Default=4)", "vector");
    inflags.AddInputFlag("hid_h", 'H', "32", "Hidden State Length (Default=32)", "int");
    inflags.AddInputFlag("in_h", 'W', "32", "Input Length (Default=32)", "int");
    inflags.AddInputFlag("out_h", 'O', "32", "Output Length (Default=32)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("verification_cache",
                         'C',
                         "",
                         "Use specified directory to cache verification data. Off by default.",
                         "string");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");
    inflags.AddInputFlag("weights", 'e', "", "Input weights filename (Default=)", "string");
    inflags.AddInputFlag("bias", 'b', "", "Use Bias (Default=0)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "tanh", "RNN Mode (relu, tanh, lstm, gru) (Default=tanh)", "str");
    inflags.AddInputFlag(
        "inputmode", 'p', "0", "linear or skip, default linear (Default=0)", "int");
    inflags.AddInputFlag(
        "rnnalgo", 'a', "0", "default, persist static or persist dynamic (Default=0)", "int");
    inflags.AddInputFlag("datatype", 'f', "1", "16-bit or 32-bit fp (Default=1)", "int");

    return 0;
}

template <typename T>
std::vector<int> RNNDriver<T>::GetInputTensorLengthsFromCmdLine()
{
    int nseq = inflags.GetValueInt("seq_len");
    int in_h = inflags.GetValueInt("in_h");
    std::vector<int> in_n(nseq, 0);
    std::string batchstr = inflags.GetValueStr("batchsize");

    std::stringstream ss(batchstr);
    int cont             = 0;

    int element;
    while (ss >> element)
    {
        if(cont >= nseq)
        {
            printf("Length of data sequence is longer than required unrolled time sequence "
                   "provided.\n"
                   "The data sequence will be truncated to match unrolled time sequence.\n");
            break;
        }

        if (ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }
        
        if(in_n[cont] > in_n[cont - 1])
        {
            printf("Incorrect input batch size at time %d\n", cont);
            break;
        }
        else
        {
            in_n[cont] = in_n[cont] * 10 + element;
            cont++;
        }
    }
    adjustedSeqLen = nseq;
    in_n.push_back(in_h);
    
    /*
    int cont             = 0;
    for(int i = 0; i < batchstr.length(); i++)
    {
        if(cont >= nseq)
        {
            printf("Length of data sequence is longer than required unrolled time sequence "
                   "provided.\n"
                   "The data sequence will be truncated to match unrolled time sequence.\n");
            break;
        }

        if(batchstr[i] == ',')
        {
            if(cont >= 1)
            {
                if(in_n[cont] > in_n[cont - 1])
                {
                    printf("Incorrect input batch size at time %d\n", cont);
                    break;
                }
            }
            cont++;
        }
        else if(batchstr[i] >= '0' && batchstr[i] <= '9')
        {
            in_n[cont] = in_n[cont] * 10 + stoi(batchstr.substr(i, 1));
        }
        else
        {
            printf("illegal input of in_n batch size");
            break;
        }
    }
    adjustedSeqLen = nseq;
    in_n.push_back(in_h);
    */

    return in_n;
}

template <typename T>
std::vector<int> RNNDriver<T>::GetHiddenTensorLengthsFromCmdLine()
{
    int hid_h = inflags.GetValueInt("hid_h");
    int hid_l = inflags.GetValueInt("num_layer");
    if((inflags.GetValueInt("bidirection")) == 1)
        hid_l *= 2;

    return std::vector<int>({hid_l, hid_h});
}

template <typename T>
std::vector<int> RNNDriver<T>::GetWeightTensorLengthsFromCmdLine()
{
    int wei_ih = inflags.GetValueInt("in_h");
    int wei_hh = inflags.GetValueInt("hid_h");
    int wei_oh = inflags.GetValueInt("out_h");
    int wei_l  = inflags.GetValueInt("num_layer");
    int wei_bi = 1;
    if((inflags.GetValueInt("bidirection")) == 1)
        wei_bi = 2;

    int wei_sc = 1;
    if((inflags.GetValueStr("mode")) == "lstm")
        wei_sc = 4;
    else if((inflags.GetValueStr("mode")) == "gru")
        wei_sc = 3;

    return std::vector<int>({wei_bi, wei_l, wei_ih, wei_hh, wei_oh, wei_sc});
}

template <typename T>
int RNNDriver<T>::SetRNNDescriptorFromCmdLineArgs()
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
        exit(0);
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
        exit(0);
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
        exit(0);
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
        exit(0);
    }

    miopenRNNAlgo_t algo;
    if((inflags.GetValueInt("rnnalgo")) == 0)
    {
        algo = miopenRNNdefault;
    }
    else if((inflags.GetValueInt("rnnalgo")) == 1)
    {
        algo = miopenRNNpersistStatic;
    }
    else if((inflags.GetValueInt("rnnalgo")) == 2)
    {
        algo = miopenRNNpersistDynamic;
    }
    else
    {
        printf("Incorrect RNN algorithm\n");
        exit(0);
    }

    miopenDataType_t dataType;
    if((inflags.GetValueInt("datatype")) == 0)
    {
        dataType = miopenHalf;
    }
    else if((inflags.GetValueInt("datatype")) == 1)
    {
        dataType = miopenFloat;
    }
    else
    {
        printf("Incorrect Data Type\n");
        exit(0);
    }

    miopenSetRNNDescriptor(
        rnnDesc, wei_hh, layer, inMode, directionMode, mode, biasMode, algo, dataType);

    return miopenStatusSuccess;
}

template <typename T>
std::vector<int> RNNDriver<T>::GetOutputTensorLengthsFromCmdLine() // need removed
{
    int out_h = inflags.GetValueInt("out_h");
    return std::vector<int>({out_h});
}

template <typename T>
int RNNDriver<T>::AllocateBuffersAndCopy()
{
    // int seqLength    = inflags.GetValueInt("seq_len");
    size_t in_sz     = 0;
    size_t out_sz    = 0;
    size_t wei_sz    = 0;
    size_t hy_sz     = 0;
    size_t params_sz = 0;
    size_t workSpaceSize;
    size_t reserveSpaceSize;

    miopenGetRNNInputSuperTensorSize(GetHandle(),
                                     rnnDesc,
                                     adjustedSeqLen,
                                     inputTensors.data(),
                                     &in_sz); // use c_array to pass vector for all size function
    miopenGetRNNInputSuperTensorSize(
        GetHandle(), rnnDesc, adjustedSeqLen, outputTensors.data(), &out_sz);
    miopenGetRNNHiddenSuperTensorSize(
        GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), &hy_sz);
    miopenGetRNNWorkspaceSize(
        GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), &workSpaceSize);
    miopenGetRNNTrainingReserveSize(
        GetHandle(), rnnDesc, adjustedSeqLen, inputTensors.data(), &reserveSpaceSize);
    miopenGetRNNParamsSize(GetHandle(), rnnDesc, inputTensors[0], &params_sz, miopenFloat);
    miopenGetRNNWeightSuperTensorSize(
        GetHandle(), rnnDesc, &wei_sz, inputTensors[0], outputTensors[0]);

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    din_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    wei_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(float)));
    dwei_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(float)));
    dout_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));
    hx_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    cx_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    hy_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    cy_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    dhx_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    dcx_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    dhy_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    dcy_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, hy_sz, sizeof(float)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSize / sizeof(T), sizeof(T)));
    reservespace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, reserveSpaceSize / sizeof(T), sizeof(T)));

    in                = std::vector<T>(in_sz);
    din               = std::vector<T>(in_sz, 0);
    wei               = std::vector<T>(wei_sz);
    dwei              = std::vector<T>(wei_sz, 0);
    dout              = std::vector<T>(out_sz, 0);
    out               = std::vector<T>(out_sz, 0);
    hx                = std::vector<T>(hy_sz, 0);
    cx                = std::vector<T>(hy_sz, 0);
    hy                = std::vector<T>(hy_sz, 0);
    cy                = std::vector<T>(hy_sz, 0);
    dhx               = std::vector<T>(hy_sz, 0);
    dcx               = std::vector<T>(hy_sz, 0);
    dhy               = std::vector<T>(hy_sz, 0);
    dcy               = std::vector<T>(hy_sz, 0);
    workspace         = std::vector<T>(workSpaceSize / sizeof(T), 0);
    reservespace      = std::vector<T>(reserveSpaceSize / sizeof(T), 0);
    outhost           = std::vector<T>(out_sz, 0);
    workspace_host    = std::vector<T>(workSpaceSize / sizeof(T), 0);
    reservespace_host = std::vector<T>(reserveSpaceSize / sizeof(T), 0);
    dwei_host         = std::vector<T>(wei_sz, 0);
    din_host          = std::vector<T>(in_sz, 0);
    hy_host           = std::vector<T>(hy_sz, 0);
    cy_host           = std::vector<T>(hy_sz, 0);
    dhx_host          = std::vector<T>(hy_sz, 0);
    dcx_host          = std::vector<T>(hy_sz, 0);

    std::string inFileName  = inflags.GetValueStr("in_data");
    std::string weiFileName = inflags.GetValueStr("weights");

    /*
    Unless seed is persistent between runs validation using cache stored in file is impossible.
    */
    srand(0);

    bool dataRead = false;
    if(!inFileName.empty())
    {
        dataRead = readBufferFromFile(in.data(), in_sz, inFileName.c_str());
    }

    double scale = 0.01;

    if(!dataRead)
    {
        for(int i = 0; i < in_sz; i++)
        {
            in[i] = static_cast<T>((static_cast<double>(scale * rand()) * (1.0 / RAND_MAX)));
        }
    }

    for(int i = 0; i < out_sz; i++)
    {
        dout[i] = static_cast<T>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
    }

    for(int i = 0; i < hy_sz; i++)
    {
        hx[i] = static_cast<T>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
    }

    for(int i = 0; i < hy_sz; i++)
    {
        dhy[i] = static_cast<T>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
    }

    if((inflags.GetValueStr("mode")) == "lstm")
    {
        for(int i = 0; i < hy_sz; i++)
        {
            cx[i] = static_cast<T>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
        }

        for(int i = 0; i < hy_sz; i++)
        {
            dcy[i] = static_cast<T>((scale * static_cast<double>(rand()) * (1.0 / RAND_MAX)));
        }
    }

    bool weiRead = false;
    if(!weiFileName.empty())
    {
        weiRead = readBufferFromFile(wei.data(), wei_sz, weiFileName.c_str());
    }

    if(!weiRead)
    {
        for(int i = 0; i < wei_sz; i++)
        {
            wei[i] =
                static_cast<T>((scale * static_cast<double>((rand()) * (1.0 / RAND_MAX) - 0.5)));
        }
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
    status |= din_dev->ToGPU(q, din.data());
    status |= wei_dev->ToGPU(q, wei.data());
    status |= dwei_dev->ToGPU(q, dwei.data());
    status |= dout_dev->ToGPU(q, dout.data());
    status |= out_dev->ToGPU(q, out.data());
    status |= hx_dev->ToGPU(q, hx.data());
    status |= cx_dev->ToGPU(q, cx.data());
    status |= hy_dev->ToGPU(q, hy.data());
    status |= cy_dev->ToGPU(q, cy.data());
    status |= dhx_dev->ToGPU(q, dhx.data());
    status |= dcx_dev->ToGPU(q, dcx.data());
    status |= dhy_dev->ToGPU(q, dhy.data());
    status |= dcy_dev->ToGPU(q, dcy.data());
    status |= workspace_dev->ToGPU(q, workspace.data());
    status |= reservespace_dev->ToGPU(q, reservespace.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

#include <array>
#include <initializer_list>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

template <typename T>
int RNNDriver<T>::RunForwardGPU()
{

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        std::fill(out.begin(), out.end(), 0);
        out_dev->ToGPU(GetStream(), out.data());

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

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward RNN Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        // printf("MIOpen Forward RNN Algorithm: %d\n", perf_results[0].fwd_algo);
        printf("GPU Kernel Time Forward RNN Elapsed: %f ms\n", time);
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

template <typename T>
int RNNDriver<T>::RunForwardCPU()
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

    miopenGetRNNDescriptor(
        rnnDesc, &mode, &algoMode, &inputMode, &dirMode, &biasMode, &hiddenSize, &layer);

    bidirection = (dirMode == miopenRNNbidirection);
    biased      = (biasMode == miopenRNNwithBias);

    if(mode == miopenRNNRELU || mode == miopenRNNTANH)
    {
        printf("verify rnn fwd \n");

        RunRNNForwardGEMMCPUVerify(in,
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
                                   reservespace_host);
    }
    else if(mode == miopenLSTM)
    {
        printf("verify lstm fwd \n");

        RunLSTMForwardGEMMCPUVerify(in,
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
                                    reservespace_host);
    }
    else if(mode == miopenGRU)
    {
        printf("verify gru fwd \n");

        RunGRUForwardGEMMCPUVerify(in,
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
                                   reservespace_host);
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

template <typename T>
int RNNDriver<T>::RunBackwardGPU()
{
    int ret = 0;

    Timer t;
    START_TIME;

    workspace_dev->ToGPU(q, workspace.data());

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        ret = miopenRNNBackwardData(GetHandle(),
                                    rnnDesc,
                                    adjustedSeqLen,
                                    outputTensors.data(),
                                    out_dev->GetMem(), // why we need this
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
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward Data RNN Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        // printf("MIOpen Backward Data RNN Algorithm: %d\n", perf_results_data[0].bwd_data_algo);
        printf("GPU Kernel Time Backward Data RNN Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());
    dhx_dev->FromGPU(GetStream(), dhx.data());
    dcx_dev->FromGPU(GetStream(), dcx.data());
    workspace_dev->FromGPU(GetStream(), workspace.data());

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

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        // printf("MIOpen Backward Weights RNN Algorithm: %d\n",
        //        perf_results_weights[0].bwd_weights_algo);
        printf("GPU Kernel Time Backward Weights RNN Elapsed: %f ms\n", time);
    }

    dwei_dev->FromGPU(GetStream(), dwei.data());
    /*
    if(perf_results_weights[0].bwd_weights_algo == 0)
    { // miopenRNNBwdWeightsAlgoGEMM
        workspace_bwd_dev->FromGPU(GetStream(), workspace_bwd.data());
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_bwd_din_gpu.bin", din.data(), din.size());
        dumpBufferToFile("dump_bwd_dwei_gpu.bin", dwei.data(), dwei.size());
    }
        */

    return ret;
}

template <typename T>
int RNNDriver<T>::RunBackwardWeightsCPU()
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

    miopenGetRNNDescriptor(
        rnnDesc, &mode, &algoMode, &inputMode, &dirMode, &biasMode, &hiddenSize, &layer);

    bidirection = (dirMode == miopenRNNbidirection);
    biased      = (biasMode == miopenRNNwithBias);

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
                                          workspace_host);
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
                                           workspace_host);
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
                                          workspace_host);
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

template <typename T>
int RNNDriver<T>::RunBackwardDataCPU()
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

    miopenGetRNNDescriptor(
        rnnDesc, &mode, &algoMode, &inputMode, &dirMode, &biasMode, &hiddenSize, &layer);

    bidirection = (dirMode == miopenRNNbidirection);
    biased      = (biasMode == miopenRNNwithBias);

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
                                        workspace_host);
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
                                         workspace_host);
    }
    else if(mode == miopenGRU)
    {
        printf("verify gru bwddata \n");

        RunGRUBackwardDataGEMMCPUVerify(din_host,
                                        wei,
                                        dhy,
                                        dhx_host,
                                        hx,
                                        dcx_host,
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
                                        workspace_host);
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
template <typename T>
std::string RNNDriver<T>::GetVerificationCacheFileName() const
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

template <typename T>
bool RNNDriver<T>::TryReadVerificationCache(const std::string& file_name,
                                             miopenTensorDescriptor_t& tensorDesc,
                                             T* data) const
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

template <typename T>
void RNNDriver<T>::TrySaveVerificationCache(const std::string& file_name,
                                             std::vector<T>& data) const
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

template <typename T>
int RNNDriver<T>::VerifyForward()
{

    //   if(!TryReadVerificationCache("fwd_out", outputTensor, outhost.data()))
    {
        RunForwardCPU();
    }

    auto error = miopen::rms_range(outhost, out);

    const double tolerance = 1e-6;
    if(!(error < tolerance))
    {
        std::cout << std::string("Forward RNN Failed: ") << error << "\n";
    }
    else
    {
        printf("Forward RNN Verifies on CPU and GPU\n");
    }

    auto error2 = miopen::rms_range(hy_host, hy);

    if(!(error2 < tolerance))
    {
        std::cout << std::string("final hidden state Failed: ") << error2 << "\n";
    }
    else
    {
        printf("final hidden Verifies on CPU and GPU\n");
    }

    if((inflags.GetValueStr("mode")) == "lstm")
    {
        auto error3 = miopen::rms_range(cy_host, cy);

        if(!(error3 < tolerance))
        {
            std::cout << std::string("final cell state Failed: ") << error3 << "\n";
        }
        else
        {
            printf("final cell Verifies on CPU and GPU\n");
        }
    }

    auto error4 = miopen::rms_range(reservespace_host, reservespace);

    if(!(error4 < tolerance))
    {
        std::cout << std::string("reserve space Failed: ") << error4 << "\n";
    }
    else
    {
        printf("reserve space Verifies on CPU and GPU\n");
    }

    return miopenStatusSuccess;
}

template <typename T>
int RNNDriver<T>::VerifyBackward()
{
    const double tolerance = 1e-6;

    //   if(!TryReadVerificationCache("bwd_dat", inputTensor, din_host.data()))
    {
        RunBackwardDataCPU();
    }

    auto error_data = miopen::rms_range(din_host, din);

    if(!(error_data < tolerance))
    {
        std::cout << std::string("Backward RNN Data Failed: ") << error_data << std::string("\n");
    }
    else
    {
        printf("Backward RNN Data Verifies on CPU and GPU\n");
    }

    auto error_data2 = miopen::rms_range(dhx_host, dhx);

    if(!(error_data2 < tolerance))
    {
        std::cout << std::string("difference at inital hidden state Failed: ") << error_data2
                  << "\n";
    }
    else
    {
        printf("difference at inital hidden state Verifies on CPU and GPU\n");
    }

    if((inflags.GetValueStr("mode")) == "lstm")
    {
        auto error_data3 = miopen::rms_range(dcx_host, dcx);

        if(!(error_data3 < tolerance))
        {
            std::cout << std::string("difference at inital cell state Failed: ") << error_data3
                      << "\n";
        }
        else
        {
            printf("difference at inital cell state Verifies on CPU and GPU\n");
        }
    }

    auto error_data4 = miopen::rms_range(workspace_host, workspace);

    if(!(error_data4 < tolerance))
    {
        std::cout << std::string("work space Failed: ") << error_data4 << "\n";
    }
    else
    {
        printf("work space Verifies on CPU and GPU\n");
    }

    //    if(!TryReadVerificationCache("bwd_wei", weightTensor, dwei_host.data()))
    {
        RunBackwardWeightsCPU();
    }

    auto error_weights = miopen::rms_range(dwei_host, dwei);
    if(!(error_weights < tolerance))
    {
        std::cout << std::string("Backward RNN Weights Failed: ") << error_weights
                  << std::string("\n");
    }
    else
    {
        printf("Backward RNN Weights Verifies on CPU and GPU\n");
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_RNN_DRIVER_HPP
