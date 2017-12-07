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
#ifndef GUARD_MIOPEN_CONV_DRIVER_HPP
#define GUARD_MIOPEN_CONV_DRIVER_HPP

#ifdef MLO_NEURON_SOFTRELU
#undef MLO_NEURON_SOFTRELU
#endif

#ifdef MLO_NEURON_POWER
#undef MLO_NEURON_POWER
#endif

#include "InputFlags.hpp"
#include "conv_verify.hpp"
#include "driver.hpp"
#include "mloConvHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include <miopen/convolution.hpp>
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

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DRIVER_PAD_BUFFERS_2M)

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

template <typename T>
class ConvDriver : public Driver
{
    public:
    ConvDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasTensor);

        miopenCreateConvolutionDescriptor(&convDesc);

        workspace_bwd_dev = nullptr;
        workspace_fwd_dev = nullptr;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();

    int SetConvDescriptorFromCmdLineArgs();

    std::vector<int> GetOutputTensorLengths();

    int AllocateBuffersAndCopy();

    int FindForward(int& ret_algo_count,
                    int request_algo_count,
                    std::vector<miopenConvAlgoPerf_t>& perf_results);
    int RunForwardGPU();
    int RunForwardCPU();

    int FindBackwardData(int& ret_algo_count,
                         int request_algo_count,
                         std::vector<miopenConvAlgoPerf_t>& perf_results);
    int FindBackwardWeights(int& ret_algo_count,
                            int request_algo_count,
                            std::vector<miopenConvAlgoPerf_t>& perf_results);
    int RunBackwardGPU();
    int RunBackwardDataCPU();
    int RunBackwardWeightsCPU();
    int RunBackwardBiasCPU();

    int VerifyBackward();
    int VerifyForward();
    ~ConvDriver()
    {

        miopenDestroyTensorDescriptor(biasTensor);
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(weightTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyConvolutionDescriptor(convDesc);
    }

    private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t outputTensor;
    miopenTensorDescriptor_t biasTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> wei_dev;
    std::unique_ptr<GPUMem> dwei_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dout_dev;
    std::unique_ptr<GPUMem> workspace_bwd_dev;
    std::unique_ptr<GPUMem> workspace_fwd_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::unique_ptr<GPUMem> db_dev;

    std::vector<T> in;
    std::vector<T> din;
    std::vector<T> wei;
    std::vector<T> dwei;
    std::vector<T> out;
    std::vector<T> dout;
    std::vector<T> workspace_bwd;
    std::vector<T> workspace_fwd;
    std::vector<T> outhost;
    std::vector<T> workspace_bwd_host;
    std::vector<T> workspace_fwd_host;
    std::vector<T> din_host;
    std::vector<T> dwei_host;
    std::vector<T> b;
    std::vector<T> db;
    std::vector<T> db_host;

    miopenConvolutionDescriptor_t convDesc;

    std::string GetVerificationCacheFileName() const;
    bool TryReadVerificationCache(const std::string& file_name,
                                  miopenTensorDescriptor_t& tensorDesc,
                                  T* data) const;
    void TrySaveVerificationCache(const std::string& file_name, std::vector<T>& data) const;
};

template <typename T>
int ConvDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename T>
int ConvDriver<T>::GetandSetData()
{
    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len);
    SetTensor4d(weightTensor, wei_len);

    SetConvDescriptorFromCmdLineArgs();

    std::vector<int> out_len = GetOutputTensorLengths();

    SetTensor4d(outputTensor, out_len);

    if(inflags.GetValueInt("bias") != 0)
    {
        if((inflags.GetValueStr("mode")) == "conv")
        {
            std::vector<int> b_len{1, inflags.GetValueInt("out_channels"), 1, 1};
            SetTensor4d(biasTensor, b_len);
        }
        else if((inflags.GetValueStr("mode")) == "trans")
        {
            std::vector<int> b_len{1, inflags.GetValueInt("in_channels"), 1, 1};
            SetTensor4d(biasTensor, b_len);
        }
    }
    return (0);
}

template <typename T>
int ConvDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Convolution (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
    inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    inflags.AddInputFlag(
        "conv_stride_0", 'u', "1", "Convolution Stride Vertical (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_1", 'v', "1", "Convolution Stride Horizontal (Default=1)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
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
        "mode", 'm', "conv", "Convolution Mode (conv, trans) (Default=conv)", "str");

    inflags.AddInputFlag(
        "pad_mode", 'z', "conv", "Padding Mode (same, valid, default) (Default=default)", "str");

    inflags.AddInputFlag("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
    inflags.AddInputFlag("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");
    inflags.AddInputFlag("in_bias", 'a', "", "Input bias filename (Default=)", "string");

    return 0;
}

template <typename T>
std::vector<int> ConvDriver<T>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename T>
std::vector<int> ConvDriver<T>::GetWeightTensorLengthsFromCmdLine()
{
    int wei_n = inflags.GetValueInt("out_channels");
    int wei_c = inflags.GetValueInt("in_channels");
    int wei_h = inflags.GetValueInt("fil_h");
    int wei_w = inflags.GetValueInt("fil_w");

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0);
    }

    if(mode == miopenTranspose)
        return std::vector<int>({wei_c, wei_n, wei_h, wei_w});

    return std::vector<int>({wei_n, wei_c, wei_h, wei_w});
}

template <typename T>
int ConvDriver<T>::SetConvDescriptorFromCmdLineArgs()
{

    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopenPaddingDefault;
    int in_h                  = inflags.GetValueInt("in_h");
    int in_w                  = inflags.GetValueInt("in_w");
    int wei_h                 = inflags.GetValueInt("fil_h");
    int wei_w                 = inflags.GetValueInt("fil_w");
    int pad_h                 = inflags.GetValueInt("pad_h");
    int pad_w                 = inflags.GetValueInt("pad_w");
    int u                     = inflags.GetValueInt("conv_stride_0");
    int v                     = inflags.GetValueInt("conv_stride_1");
    int dilation_h            = inflags.GetValueInt("dilation_h");
    int dilation_w            = inflags.GetValueInt("dilation_w");
    if((inflags.GetValueStr("mode")) == "conv")
    {
        pmode = miopenPaddingDefault;
        mode  = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        pmode = miopenPaddingDefault;
        mode  = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0);
    }

    if((inflags.GetValueStr("pad_mode")) == "same")
    {
        mode  = miopenConvolution;
        pmode = miopenPaddingSame;
        pad_h = (in_h % u == 0) ? (std::max((wei_h - u), 0)) : (std::max((wei_h - (in_h % u)), 0));
        pad_w = (in_w % v == 0) ? (std::max((wei_w - v), 0)) : (std::max((wei_w - (in_w % v)), 0));
        pad_h /= 2;
        pad_w /= 2;
    }
    else if((inflags.GetValueStr("pad_mode")) == "valid")
    {
        pmode = miopenPaddingValid;
        mode  = miopenConvolution;
        pad_h = 0;
        pad_w = 0;
    }

    miopen::deref(convDesc) =
        miopen::ConvolutionDescriptor(mode, pmode, pad_h, pad_w, u, v, dilation_h, dilation_w);
    return miopenStatusSuccess;
}

template <typename T>
std::vector<int> ConvDriver<T>::GetOutputTensorLengths()
{
    int n, c, h, w;

    miopenGetConvolutionForwardOutputDim(convDesc, inputTensor, weightTensor, &n, &c, &h, &w);

    return std::vector<int>({n, c, h, w});
}

template <typename T>
int ConvDriver<T>::AllocateBuffersAndCopy()
{

    size_t in_sz  = GetTensorSize(inputTensor);
    size_t wei_sz = GetTensorSize(weightTensor);
    size_t out_sz = GetTensorSize(outputTensor);

    size_t workSpaceSize_bwd_wt = 0;
    size_t workSpaceSize_bwd_dt = 0;
    miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        GetHandle(), outputTensor, inputTensor, convDesc, weightTensor, &workSpaceSize_bwd_wt);
    miopenConvolutionBackwardDataGetWorkSpaceSize(
        GetHandle(), outputTensor, weightTensor, convDesc, inputTensor, &workSpaceSize_bwd_dt);
    size_t workSpaceSize_bwd =
        workSpaceSize_bwd_dt > workSpaceSize_bwd_wt ? workSpaceSize_bwd_dt : workSpaceSize_bwd_wt;

    size_t workSpaceSize_fwd = 0;
    miopenConvolutionForwardGetWorkSpaceSize(
        GetHandle(), weightTensor, inputTensor, convDesc, outputTensor, &workSpaceSize_fwd);

    // Workaround: Pad buffers allocations to be a multiple of 2M
    if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
    {
        // PadBufferSize(in_sz, 4);
        PadBufferSize(wei_sz, 4);
        PadBufferSize(out_sz, 4);
    }

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    wei_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(float)));
    dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(float)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));
    out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));
    if(workSpaceSize_bwd != 0)
    {
        workspace_bwd_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSize_bwd / sizeof(T), sizeof(T)));
        workspace_bwd      = std::vector<T>(workSpaceSize_bwd / sizeof(T), 0);
        workspace_bwd_host = std::vector<T>(workSpaceSize_bwd / sizeof(T), 0);
    }
    if(workSpaceSize_fwd != 0)
    {
        workspace_fwd_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSize_fwd / sizeof(T), sizeof(T)));
        workspace_fwd      = std::vector<T>(workSpaceSize_fwd / sizeof(T), 0);
        workspace_fwd_host = std::vector<T>(workSpaceSize_fwd / sizeof(T), 0);
    }

    in   = std::vector<T>(in_sz);
    din  = std::vector<T>(in_sz);
    wei  = std::vector<T>(wei_sz);
    dwei = std::vector<T>(wei_sz, 0);
    dout = std::vector<T>(out_sz, 0);
    out  = std::vector<T>(out_sz, 0);

    outhost = std::vector<T>(out_sz, 0);

    dwei_host = std::vector<T>(wei_sz, 0);
    din_host  = std::vector<T>(in_sz, 0);

    std::string inFileName   = inflags.GetValueStr("in_data");
    std::string weiFileName  = inflags.GetValueStr("weights");
    std::string biasFileName = inflags.GetValueStr("in_bias");

    /* Unless seed is persistent between runs validation using cache stored in file is impossible.
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

    if(inflags.GetValueInt("bias") != 0)
    {
        size_t b_sz = GetTensorSize(biasTensor);
        b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(float)));
        db_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(float)));
        b           = std::vector<T>(b_sz);
        db          = std::vector<T>(b_sz);
        db_host     = std::vector<T>(b_sz, 0);
        for(int i = 0; i < b_sz; i++)
        {
            b[i]  = i % 8;
            db[i] = i % 8;
            if((inflags.GetValueStr("mode")) == "trans")
            {
                db[i] = 0;
            }
        }

        if(!biasFileName.empty())
        {
            readBufferFromFile(b.data(), b_sz, biasFileName.c_str());
        }

        b_dev->ToGPU(q, b.data());
        db_dev->ToGPU(q, db.data());
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
        if(inflags.GetValueInt("bias") != 0)
            dumpBufferToFile("dump_bias.bin", b.data(), GetTensorSize(biasTensor));
    }
#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
#define CL_SUCCESS 0
    int status;
#endif
    status = in_dev->ToGPU(q, in.data());
    status |= din_dev->ToGPU(q, in.data());
    status |= wei_dev->ToGPU(q, wei.data());
    status |= dwei_dev->ToGPU(q, dwei.data());
    status |= dout_dev->ToGPU(q, dout.data());
    status |= out_dev->ToGPU(q, out.data());
    if(workSpaceSize_bwd != 0)
        status |= workspace_bwd_dev->ToGPU(q, workspace_bwd.data());
    if(workSpaceSize_fwd != 0)
        status |= workspace_fwd_dev->ToGPU(q, workspace_fwd.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename T>
int ConvDriver<T>::FindForward(int& ret_algo_count,
                               int request_algo_count,
                               std::vector<miopenConvAlgoPerf_t>& perf_results)
{

    return miopenFindConvolutionForwardAlgorithm(
        GetHandle(),
        inputTensor,
        in_dev->GetMem(),
        weightTensor,
        wei_dev->GetMem(),
        convDesc,
        outputTensor,
        out_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem() : nullptr,
        (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetSize() : 0,
        (inflags.GetValueInt("search") == 1) ? true : false);
}

template <typename T>
int ConvDriver<T>::RunForwardGPU()
{

    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results(request_algo_count);

    FindForward(ret_algo_count, request_algo_count, perf_results);

    float alpha = 1, beta = 0;

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenConvolutionForward(GetHandle(),
                                 &alpha,
                                 inputTensor,
                                 in_dev->GetMem(),
                                 weightTensor,
                                 wei_dev->GetMem(),
                                 convDesc,
                                 perf_results[0].fwd_algo, // use the fastest algo
                                 &beta,
                                 outputTensor,
                                 out_dev->GetMem(),
                                 (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem()
                                                                : nullptr,
                                 (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetSize() : 0);
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward Conv. Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        printf("MIOpen Forward Conv. Algorithm: %d\n", perf_results[0].fwd_algo);
        printf("GPU Kernel Time Forward Conv. Elapsed: %f ms\n", time);
    }

    if(inflags.GetValueInt("bias") != 0)
    {
        if((inflags.GetValueStr("mode")) == "conv")
        {
            miopenConvolutionForwardBias(GetHandle(),
                                         &alpha,
                                         biasTensor,
                                         b_dev->GetMem(),
                                         &beta,
                                         outputTensor,
                                         out_dev->GetMem());
        }
        else if((inflags.GetValueStr("mode")) == "trans")
        {
            miopenConvolutionBackwardBias(GetHandle(),
                                          &alpha,
                                          inputTensor,
                                          in_dev->GetMem(),
                                          &beta,
                                          biasTensor,
                                          b_dev->GetMem());
        }

        if(inflags.GetValueInt("time") == 1)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);

            printf("GPU Kernel Time Forward Conv. Bias Elapsed: %f ms\n", time);
        }
    }

    out_dev->FromGPU(GetStream(), out.data());

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_fwd_out_gpu.bin", out.data(), out.size());
    }

    return miopenStatusSuccess;
}

template <typename T>
int ConvDriver<T>::RunForwardCPU()
{

    int in_n, in_c, in_h, in_w;
    int in_nstride, in_cstride, in_hstride, in_wstride;
    miopenDataType_t dt;
    miopenGet4dTensorDescriptor(inputTensor,
                                &dt,
                                &in_n,
                                &in_c,
                                &in_h,
                                &in_w,
                                &in_nstride,
                                &in_cstride,
                                &in_hstride,
                                &in_wstride);

    int wei_n, wei_c, wei_h, wei_w;
    int wei_nstride, wei_cstride, wei_hstride, wei_wstride;

    miopenGet4dTensorDescriptor(weightTensor,
                                &dt,
                                &wei_n,
                                &wei_c,
                                &wei_h,
                                &wei_w,
                                &wei_nstride,
                                &wei_cstride,
                                &wei_hstride,
                                &wei_wstride);

    int out_n, out_c, out_h, out_w;
    int out_nstride, out_cstride, out_hstride, out_wstride;
    miopenGet4dTensorDescriptor(outputTensor,
                                &dt,
                                &out_n,
                                &out_c,
                                &out_h,
                                &out_w,
                                &out_nstride,
                                &out_cstride,
                                &out_hstride,
                                &out_wstride);

    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(convDesc).paddingMode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    if(pmode == miopenPaddingSame)
    {
        pad_h = (in_h % u == 0) ? (std::max((wei_h - u), 0)) : (std::max((wei_h - (in_h % u)), 0));
        pad_w = (in_w % v == 0) ? (std::max((wei_w - v), 0)) : (std::max((wei_w - (in_w % v)), 0));
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_h = 0;
        pad_w = 0;
    }

    if(out_h <= 0 || out_w <= 0)
        MIOPEN_THROW("Invalid Test Case: Check Output Dimension.");

    if(mode == miopenConvolution)
    {
        miopenGet4dTensorDescriptor(weightTensor,
                                    &dt,
                                    &wei_n,
                                    &wei_c,
                                    &wei_h,
                                    &wei_w,
                                    &wei_nstride,
                                    &wei_cstride,
                                    &wei_hstride,
                                    &wei_wstride);

        for(int o = 0; o < out_n; o++)
        { // mini-batch size
            for(int w = 0; w < out_c; w++)
            { // out_channels (num filters)
                for(int i = 0; i < out_h; i++)
                { // output_height (from getforwardoutputdim())
                    int in_off_h = i * u;
                    for(int j = 0; j < out_w; j++)
                    { // output_width (from getforwardoutputdim())
                        float acc    = 0;
                        int in_off_w = j * v;
                        for(int k = 0; k < in_c; k++)
                        { // in_channels (RGB)
                            for(int x = 0; x < wei_h; x++)
                            {
                                int in_x = in_off_h - pad_h + x * dilation_h;
                                if(in_x >= 0 && in_x < in_h)
                                {
                                    for(int y = 0; y < wei_w; y++)
                                    {
                                        int in_y = in_off_w - pad_w + y * dilation_w;
                                        if(in_y >= 0 && in_y < in_w)
                                        {
                                            acc += in[o * in_nstride + k * in_cstride +
                                                      in_x * in_w + in_y] *
                                                   wei[w * wei_nstride + k * wei_cstride +
                                                       x * wei_hstride + y];
                                        }
                                    }
                                }
                            }
                        }
                        acc = inflags.GetValueInt("bias") != 0 ? acc + b[w] : acc;
                        outhost[o * out_nstride + w * out_cstride + i * out_hstride + j] = acc;
                    }
                }
            }
        }
    }
    else if(mode == miopenTranspose)
    {
        miopenGet4dTensorDescriptor(weightTensor,
                                    &dt,
                                    &wei_c,
                                    &wei_n,
                                    &wei_h,
                                    &wei_w,
                                    &wei_cstride,
                                    &wei_nstride,
                                    &wei_hstride,
                                    &wei_wstride);

        for(int o = 0; o < in_n; o++)
        { // mini-batch size
            for(int k = 0; k < out_c; k++)
            { // out_channels (RGB)
                for(int w = 0; w < in_c; w++)
                { // in_channels (num filters)
                    for(int i = 0; i < in_h; i++)
                    { // input_height
                        int out_off_h = i * v;
                        for(int j = 0; j < in_w; j++)
                        { // input_width
                            int out_off_w = j * u;
                            for(int x = 0; x < wei_h; x++)
                            {
                                int out_x = out_off_h - pad_h + x * dilation_h;
                                if(out_x >= 0 && out_x < out_h)
                                {
                                    for(int y = 0; y < wei_w; y++)
                                    {
                                        int out_y = out_off_w - pad_w + y * dilation_w;
                                        if(out_y >= 0 && out_y < out_w)
                                        {
                                            outhost[o * out_nstride + k * out_cstride +
                                                    out_x * out_hstride + out_y] +=
                                                in[o * in_nstride + w * in_cstride +
                                                   i * in_hstride + j] *
                                                wei[w * wei_cstride + k * wei_nstride +
                                                    x * wei_hstride + y];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_fwd_out_cpu.bin", outhost.data(), outhost.size());
    }

    TrySaveVerificationCache("fwd_out", outhost);
    return 0;
}

template <typename T>
int ConvDriver<T>::FindBackwardData(int& ret_algo_count,
                                    int request_algo_count,
                                    std::vector<miopenConvAlgoPerf_t>& perf_results)
{

    return miopenFindConvolutionBackwardDataAlgorithm(GetHandle(),
                                                      outputTensor,
                                                      dout_dev->GetMem(),
                                                      weightTensor,
                                                      wei_dev->GetMem(),
                                                      convDesc,
                                                      inputTensor,
                                                      din_dev->GetMem(),
                                                      request_algo_count,
                                                      &ret_algo_count,
                                                      perf_results.data(),
                                                      workspace_bwd_dev->GetMem(),
                                                      workspace_bwd_dev->GetSize(),
                                                      (inflags.GetValueInt("search") == 1) ? true
                                                                                           : false);
}

template <typename T>
int ConvDriver<T>::FindBackwardWeights(int& ret_algo_count,
                                       int request_algo_count,
                                       std::vector<miopenConvAlgoPerf_t>& perf_results)
{

    miopenFindConvolutionBackwardWeightsAlgorithm(GetHandle(),
                                                  outputTensor,
                                                  dout_dev->GetMem(),
                                                  inputTensor,
                                                  in_dev->GetMem(),
                                                  convDesc,
                                                  weightTensor,
                                                  wei_dev->GetMem(),
                                                  request_algo_count,
                                                  &ret_algo_count,
                                                  perf_results.data(),
                                                  workspace_bwd_dev->GetMem(),
                                                  workspace_bwd_dev->GetSize(),
                                                  (inflags.GetValueInt("search") == 1) ? true
                                                                                       : false);

    return 0;
}

template <typename T>
int ConvDriver<T>::RunBackwardGPU()
{

    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results_data(request_algo_count);

    FindBackwardData(ret_algo_count, request_algo_count, perf_results_data);

    float alpha = 1, beta = 0;
    int ret = 0;

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        ret = miopenConvolutionBackwardData(GetHandle(),
                                            &alpha,
                                            outputTensor,
                                            dout_dev->GetMem(),
                                            weightTensor,
                                            wei_dev->GetMem(),
                                            convDesc,
                                            perf_results_data[0].bwd_data_algo,
                                            &beta,
                                            inputTensor,
                                            din_dev->GetMem(),
                                            workspace_bwd_dev->GetMem(),
                                            workspace_bwd_dev->GetSize());
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward Data Conv. Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        printf("MIOpen Backward Data Conv. Algorithm: %d\n", perf_results_data[0].bwd_data_algo);
        printf("GPU Kernel Time Backward Data Conv. Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());

    std::vector<miopenConvAlgoPerf_t> perf_results_weights(request_algo_count);

    FindBackwardWeights(ret_algo_count, request_algo_count, perf_results_weights);

    ret = miopenConvolutionBackwardWeights(GetHandle(),
                                           &alpha,
                                           outputTensor,
                                           dout_dev->GetMem(),
                                           inputTensor,
                                           in_dev->GetMem(),
                                           convDesc,
                                           perf_results_weights[0].bwd_weights_algo,
                                           &beta,
                                           weightTensor,
                                           dwei_dev->GetMem(),
                                           workspace_bwd_dev->GetMem(),
                                           workspace_bwd_dev->GetSize());

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        printf("MIOpen Backward Weights Conv. Algorithm: %d\n",
               perf_results_weights[0].bwd_weights_algo);
        printf("GPU Kernel Time Backward Weights Conv. Elapsed: %f ms\n", time);
    }
    dwei_dev->FromGPU(GetStream(), dwei.data());

    if(perf_results_weights[0].bwd_weights_algo == 0)
    { // miopenConvolutionBwdWeightsAlgoGEMM
        workspace_bwd_dev->FromGPU(GetStream(), workspace_bwd.data());
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_bwd_din_gpu.bin", din.data(), din.size());
        dumpBufferToFile("dump_bwd_dwei_gpu.bin", dwei.data(), dwei.size());
    }

    if(inflags.GetValueInt("bias") != 0)
    {

        if((inflags.GetValueStr("mode")) == "conv")
        {
            ret = miopenConvolutionBackwardBias(GetHandle(),
                                                &alpha,
                                                outputTensor,
                                                dout_dev->GetMem(),
                                                &beta,
                                                biasTensor,
                                                db_dev->GetMem());
        }
        //       else if((inflags.GetValueStr("mode")) == "trans")
        //       {
        //           ret = miopenConvolutionForwardBias(GetHandle(),
        //		                            &alpha,
        //		                            biasTensor,
        //		                            db_dev->GetMem(),
        //		                            &beta,
        //		                            inputTensor,
        //		                            din_dev->GetMem());
        //       }

        if(inflags.GetValueInt("time") == 1)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            printf("GPU Kernel Time Backward Bias Conv. Elapsed: %f ms\n", time);
        }

        db_dev->FromGPU(GetStream(), db.data());
        if(inflags.GetValueInt("dump_output"))
        {
            dumpBufferToFile("dump_bwd_db_gpu.bin", db.data(), db.size());
        }
    }
    return ret;
}

template <typename T>
int ConvDriver<T>::RunBackwardWeightsCPU()
{

    int in_n, in_c, in_h, in_w;
    int in_nstride, in_cstride, in_hstride, in_wstride;
    miopenDataType_t dt;
    miopenGet4dTensorDescriptor(inputTensor,
                                &dt,
                                &in_n,
                                &in_c,
                                &in_h,
                                &in_w,
                                &in_nstride,
                                &in_cstride,
                                &in_hstride,
                                &in_wstride);

    int wei_n, wei_c, wei_h, wei_w;
    int wei_nstride, wei_cstride, wei_hstride, wei_wstride;

    miopenGet4dTensorDescriptor(weightTensor,
                                &dt,
                                &wei_n,
                                &wei_c,
                                &wei_h,
                                &wei_w,
                                &wei_nstride,
                                &wei_cstride,
                                &wei_hstride,
                                &wei_wstride);

    int out_n, out_c, out_h, out_w;
    int out_nstride, out_cstride, out_hstride, out_wstride;
    miopenGet4dTensorDescriptor(outputTensor,
                                &dt,
                                &out_n,
                                &out_c,
                                &out_h,
                                &out_w,
                                &out_nstride,
                                &out_cstride,
                                &out_hstride,
                                &out_wstride);

    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(convDesc).paddingMode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    if(pmode == miopenPaddingSame)
    {
        pad_h = (in_h % u == 0) ? (std::max((wei_h - u), 0)) : (std::max((wei_h - (in_h % u)), 0));
        pad_w = (in_w % v == 0) ? (std::max((wei_w - v), 0)) : (std::max((wei_w - (in_w % v)), 0));
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_h = 0;
        pad_w = 0;
    }

    if(out_h <= 0 || out_w <= 0)
        MIOPEN_THROW("Invalid Test Case: Check Output Dimension.");

    if(mode == miopenConvolution)
    {
        miopenGet4dTensorDescriptor(weightTensor,
                                    &dt,
                                    &wei_n,
                                    &wei_c,
                                    &wei_h,
                                    &wei_w,
                                    &wei_nstride,
                                    &wei_cstride,
                                    &wei_hstride,
                                    &wei_wstride);

#ifdef MIOPEN_USE_MIOPENGEMM
#ifndef NDEBUG
        if(in_n == 1 && wei_h != 1 && wei_w != 1)
        {
            // workspace_bwd will be nonzero only if gemm was chosen as the algo
            bool zeros = std::all_of(
                workspace_bwd.begin(), workspace_bwd.end(), [](int i) { return i == 0; });

            if(!zeros)
            {
                Im2ColCPU(in,
                          0,
                          in_c,
                          in_h,
                          in_w,
                          wei_h,
                          wei_w,
                          out_h,
                          out_w,
                          pad_h,
                          pad_w,
                          u,
                          v,
                          workspace_bwd_host);

                for(int i = 0; i < workspace_bwd.size(); i++)
                {
                    if(std::abs(workspace_bwd[i] - workspace_bwd_host[i]) > 0.0)
                    {
                        printf("Im2col error: %d %f %f\n ",
                               i,
                               workspace_bwd[i],
                               workspace_bwd_host[i]);
                    }
                }
            }
        }
#endif
#endif

        RunBackwardWeightsCPUVerify(dwei_host,
                                    in,
                                    dout,
                                    in_n,
                                    in_c,
                                    in_h,
                                    in_w,
                                    in_nstride,
                                    in_cstride,
                                    in_hstride,
                                    in_wstride,
                                    wei_n,
                                    wei_c,
                                    wei_h,
                                    wei_w,
                                    wei_nstride,
                                    wei_cstride,
                                    wei_hstride,
                                    wei_wstride,
                                    out_n,
                                    out_c,
                                    out_h,
                                    out_w,
                                    out_nstride,
                                    out_cstride,
                                    out_hstride,
                                    out_wstride,
                                    u,
                                    v,
                                    pad_h,
                                    pad_w,
                                    dilation_h,
                                    dilation_w);
    }
    else if(mode == miopenTranspose)
    {
        miopenGet4dTensorDescriptor(weightTensor,
                                    &dt,
                                    &wei_c,
                                    &wei_n,
                                    &wei_h,
                                    &wei_w,
                                    &wei_cstride,
                                    &wei_nstride,
                                    &wei_hstride,
                                    &wei_wstride);

#ifdef MIOPEN_USE_MIOPENGEMM
#ifndef NDEBUG
        if(in_n == 1 && wei_h != 1 && wei_w != 1)
        {
            // workspace_bwd will be nonzero only if gemm was chosen as the algo
            bool zeros = std::all_of(
                workspace_bwd.begin(), workspace_bwd.end(), [](int i) { return i == 0; });

            if(!zeros)
            {
                Im2ColCPU(dout,
                          0,
                          out_c,
                          out_h,
                          out_w,
                          wei_h,
                          wei_w,
                          in_h,
                          in_w,
                          pad_h,
                          pad_w,
                          v,
                          u,
                          workspace_bwd_host);

                for(int i = 0; i < workspace_bwd.size(); i++)
                {
                    if(std::abs(workspace_bwd[i] - workspace_bwd_host[i]) > 0.0)
                    {
                        printf("Im2col error: %d %f %f\n ",
                               i,
                               workspace_bwd[i],
                               workspace_bwd_host[i]);
                    }
                }
            }
        }
#endif
#endif

        RunBackwardWeightsCPUVerify(dwei_host,
                                    dout,
                                    in,
                                    out_n,
                                    out_c,
                                    out_h,
                                    out_w,
                                    out_nstride,
                                    out_cstride,
                                    out_hstride,
                                    out_wstride,
                                    wei_c,
                                    wei_n,
                                    wei_h,
                                    wei_w,
                                    wei_cstride,
                                    wei_nstride,
                                    wei_hstride,
                                    wei_wstride,
                                    in_n,
                                    in_c,
                                    in_h,
                                    in_w,
                                    in_nstride,
                                    in_cstride,
                                    in_hstride,
                                    in_wstride,
                                    u,
                                    v,
                                    pad_h,
                                    pad_w,
                                    dilation_h,
                                    dilation_w);
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_bwd_dwei_cpu.bin", dwei_host.data(), dwei_host.size());
    }

    TrySaveVerificationCache("bwd_wei", dwei_host);
    return 0;
}

template <typename T>
int ConvDriver<T>::RunBackwardDataCPU()
{

    int in_n, in_c, in_h, in_w;
    int in_nstride, in_cstride, in_hstride, in_wstride;
    miopenDataType_t dt;
    miopenGet4dTensorDescriptor(inputTensor,
                                &dt,
                                &in_n,
                                &in_c,
                                &in_h,
                                &in_w,
                                &in_nstride,
                                &in_cstride,
                                &in_hstride,
                                &in_wstride);

    int wei_n, wei_c, wei_h, wei_w;
    int wei_nstride, wei_cstride, wei_hstride, wei_wstride;

    miopenGet4dTensorDescriptor(weightTensor,
                                &dt,
                                &wei_n,
                                &wei_c,
                                &wei_h,
                                &wei_w,
                                &wei_nstride,
                                &wei_cstride,
                                &wei_hstride,
                                &wei_wstride);

    int out_n, out_c, out_h, out_w;
    int out_nstride, out_cstride, out_hstride, out_wstride;
    miopenGet4dTensorDescriptor(outputTensor,
                                &dt,
                                &out_n,
                                &out_c,
                                &out_h,
                                &out_w,
                                &out_nstride,
                                &out_cstride,
                                &out_hstride,
                                &out_wstride);

    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(convDesc).paddingMode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    if(out_h <= 0 || out_w <= 0)
        MIOPEN_THROW("Invalid Test Case: Check Output Dimension.");

    if(pmode == miopenPaddingSame)
    {
        pad_h = (in_h % u == 0) ? (std::max((wei_h - u), 0)) : (std::max((wei_h - (in_h % u)), 0));
        pad_w = (in_w % v == 0) ? (std::max((wei_w - v), 0)) : (std::max((wei_w - (in_w % v)), 0));
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_h = 0;
        pad_w = 0;
    }

    if(mode == miopenConvolution)
    {
        miopenGet4dTensorDescriptor(weightTensor,
                                    &dt,
                                    &wei_n,
                                    &wei_c,
                                    &wei_h,
                                    &wei_w,
                                    &wei_nstride,
                                    &wei_cstride,
                                    &wei_hstride,
                                    &wei_wstride);

        for(int o = 0; o < out_n; o++)
        { // mini-batch size
            for(int k = 0; k < in_c; k++)
            { // in_channels (RGB)
                for(int w = 0; w < out_c; w++)
                { // out_channels (num filters)
                    for(int i = 0; i < out_h; i++)
                    { // output_height (from getforwardoutputdim())
                        int in_off_h = i * u;
                        for(int j = 0; j < out_w; j++)
                        { // output_width (from getforwardoutputdim())
                            int in_off_w = j * v;
                            for(int x = 0; x < wei_h; x++)
                            {
                                int in_x = in_off_h - pad_h + x * dilation_h;
                                if(in_x >= 0 && in_x < in_h)
                                {
                                    for(int y = 0; y < wei_w; y++)
                                    {
                                        int in_y = in_off_w - pad_w + y * dilation_w;
                                        if(in_y >= 0 && in_y < in_w)
                                        {
                                            din_host[o * in_nstride + k * in_cstride +
                                                     in_x * in_hstride + in_y] +=
                                                dout[o * out_nstride + w * out_cstride +
                                                     i * out_hstride + j] *
                                                wei[w * wei_nstride + k * wei_cstride +
                                                    x * wei_hstride + y];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else if(mode == miopenTranspose)
    {
        miopenGet4dTensorDescriptor(weightTensor,
                                    &dt,
                                    &wei_c,
                                    &wei_n,
                                    &wei_h,
                                    &wei_w,
                                    &wei_cstride,
                                    &wei_nstride,
                                    &wei_hstride,
                                    &wei_wstride);

        for(int o = 0; o < in_n; o++)
        { // mini-batch size
            for(int w = 0; w < in_c; w++)
            { // in_channels (num filters)
                for(int i = 0; i < in_h; i++)
                { // input_height (from getforwardoutputdim())
                    int out_off_h = i * v;
                    for(int j = 0; j < in_w; j++)
                    { // input_width (from getforwardoutputdim())
                        float acc     = 0;
                        int out_off_w = j * u;
                        for(int k = 0; k < out_c; k++)
                        { // out_channels (RGB)
                            for(int x = 0; x < wei_h; x++)
                            {
                                int out_x = out_off_h - pad_h + x * dilation_h;
                                if(out_x >= 0 && out_x < out_h)
                                {
                                    for(int y = 0; y < wei_w; y++)
                                    {
                                        int out_y = out_off_w - pad_w + y * dilation_w;
                                        if(out_y >= 0 && out_y < out_w)
                                        {
                                            acc += dout[o * out_nstride + k * out_cstride +
                                                        out_x * out_w + out_y] *
                                                   wei[w * wei_cstride + k * wei_nstride +
                                                       x * wei_hstride + y];
                                        }
                                    }
                                }
                            }
                        }
                        //                      acc = inflags.GetValueInt("bias") != 0 ? acc + db[w]
                        //                      : acc;  // db is zero in transpose case
                        din_host[o * in_nstride + w * in_cstride + i * in_hstride + j] = acc;
                    }
                }
            }
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_bwd_din_cpu.bin", din_host.data(), din_host.size());
    }

    TrySaveVerificationCache("bwd_dat", din_host);
    return 0;
}

template <typename T>
int ConvDriver<T>::RunBackwardBiasCPU()
{

    miopenDataType_t dt;
    int out_n, out_c, out_h, out_w;
    int out_nstride, out_cstride, out_hstride, out_wstride;

    miopenGet4dTensorDescriptor(outputTensor,
                                &dt,
                                &out_n,
                                &out_c,
                                &out_h,
                                &out_w,
                                &out_nstride,
                                &out_cstride,
                                &out_hstride,
                                &out_wstride);

    for(int c = 0; c < out_c; c++)
    {
        db_host[c] = 0.0f;
        for(int n = 0; n < out_n; n++)
        {
            for(int h = 0; h < out_h; h++)
            {
                for(int w = 0; w < out_w; w++)
                {
                    if((inflags.GetValueStr("mode")) == "conv")
                    {
                        db_host[c] += dout[n * out_nstride + c * out_cstride + h * out_hstride + w];
                    }
                    //                    else if((inflags.GetValueStr("mode")) == "trans")
                    //                    {
                    //                        db_host[c] += dout[n * out_nstride + c * out_cstride +
                    //                        h * out_hstride + w] * dYb[c];
                    //                    }
                }
            }
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile("dump_bwd_db_cpu.bin", db_host.data(), db_host.size());
    }

    TrySaveVerificationCache("bwd_bai", db_host);
    return 0;
}

template <typename T>
std::string ConvDriver<T>::GetVerificationCacheFileName() const
{
    std::ostringstream ss;

    miopenConvolutionMode_t mode;
    int pad_h, pad_w, u, v, sx, sy;
    miopenGetConvolutionDescriptor(convDesc, &mode, &pad_h, &pad_w, &u, &v, &sx, &sy);

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
bool ConvDriver<T>::TryReadVerificationCache(const std::string& file_name,
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
void ConvDriver<T>::TrySaveVerificationCache(const std::string& file_name,
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

template <typename T>
int ConvDriver<T>::VerifyForward()
{

    if(!TryReadVerificationCache("fwd_out", outputTensor, outhost.data()))
    {
        RunForwardCPU();
    }

    auto error             = miopen::rms_range(outhost, out);
    const double tolerance = 1e-6;
    if(!(error < tolerance))
    {
        std::cout << std::string("Forward Convolution Failed: ") << error << "\n";
    }
    else
    {
        printf("Forward Convolution Verifies on CPU and GPU\n");
    }

    return 0;
}

template <typename T>
int ConvDriver<T>::VerifyBackward()
{
    const double tolerance = 1e-6;

    if(!TryReadVerificationCache("bwd_dat", inputTensor, din_host.data()))
    {
        RunBackwardDataCPU();
    }

    auto error_data = miopen::rms_range(din_host, din);

    if(!(error_data < tolerance))
    {
        std::cout << std::string("Backward Convolution Data Failed: ") << error_data
                  << std::string("\n");
    }
    else
    {
        printf("Backward Convolution Data Verifies on CPU and GPU\n");
    }

    if(!TryReadVerificationCache("bwd_wei", weightTensor, dwei_host.data()))
    {
        RunBackwardWeightsCPU();
    }

    auto error_weights = miopen::rms_range(dwei_host, dwei);
    if(!(error_weights < tolerance))
    {
        std::cout << std::string("Backward Convolution Weights Failed: ") << error_weights
                  << std::string("\n");
    }
    else
    {
        printf("Backward Convolution Weights Verifies on CPU and GPU\n");
    }

    if(inflags.GetValueInt("bias") != 0)
    {
        if(!TryReadVerificationCache("bwd_bai", biasTensor, db_host.data()))
        {
            RunBackwardBiasCPU();
        }

        auto error_bias = miopen::rms_range(db_host, db);
        if(!(error_bias < tolerance))
        {
            std::cout << std::string("Backward Convolution Bias Failed: ") << error_bias
                      << std::string("\n");
        }
        else
        {
            printf("Backward Convolution Bias Verifies on CPU and GPU\n");
        }
    }

    return 0;
}

#endif // GUARD_MIOPEN_CONV_DRIVER_HPP
