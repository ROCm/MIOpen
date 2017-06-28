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
#ifndef GUARD_MIOPEN_LRN_DRIVER_HPP
#define GUARD_MIOPEN_LRN_DRIVER_HPP

#include "../test/verify.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloNormHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>

template <typename T>
class LRNDriver : public Driver
{
    public:
    LRNDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateLRNDescriptor(&lrnDesc);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetLRNDescriptorFromCmdLineArgs();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU();
    int RunBackwardCPU();

    int VerifyBackward();
    int VerifyForward();
    ~LRNDriver()
    {

        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyLRNDescriptor(lrnDesc);
    }

    private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> scale_dev;

    std::vector<T> in;
    std::vector<T> out;
    std::vector<T> outhost;
    std::vector<T> scale;
    std::vector<T> scalehost;

    miopenLRNDescriptor_t lrnDesc;

    miopenTensorDescriptor_t dInputTensor;
    miopenTensorDescriptor_t dOutputTensor;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> dout_dev;

    std::vector<T> din;
    std::vector<T> dout;
    std::vector<T> dinhost;
};

template <typename T>
int LRNDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
#if 0
	if(inflags.GetValueInt("back") == 0 && inflags.GetValueStr("mode") == "cross") {
		printf("Cross channel LRN needs do_backward=1\n");
		exit(0);
	}
#endif
    return 0;
}

template <typename T>
int LRNDriver<T>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len);

    SetLRNDescriptorFromCmdLineArgs();

    SetTensor4d(outputTensor, in_len);

    SetTensor4d(dInputTensor, in_len);
    SetTensor4d(dOutputTensor, in_len);
    return (0);
}

template <typename T>
int LRNDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward LRN Normalization (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("lrnN", 'N', "5", " lrnN (Default=5)", "int");
    inflags.AddInputFlag("alpha", 'A', "0.001", "lrn Alpha (Default=0.001)", "double");
    inflags.AddInputFlag("beta", 'B', "0.75", "lrn Beta (Default=0.75)", "double");
    inflags.AddInputFlag("lrnK", 'K', "1.0", "lrnK (Default=1.0)", "double");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    //	inflags.AddInputFlag("back", 'b', "1", "Optimization: Do Backward LRN (Default=1)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag("mode",
                         'm',
                         "within",
                         "LRN Mode (within_channel or cross_channel) (Default=within)",
                         "str");

    return 0;
}

template <typename T>
std::vector<int> LRNDriver<T>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename T>
int LRNDriver<T>::SetLRNDescriptorFromCmdLineArgs()
{

    miopenLRNMode_t mode;
    int lrnN        = inflags.GetValueInt("lrnN");
    double lrnAlpha = inflags.GetValueDouble("alpha");
    double lrnBeta  = inflags.GetValueDouble("beta");
    double lrnK     = inflags.GetValueDouble("lrnK");
    if((inflags.GetValueStr("mode")) == "within")
    {
        mode = miopenLRNWithinChannel;
    }
    else if((inflags.GetValueStr("mode")) == "cross")
    {
        mode = miopenLRNCrossChannel;
    }
    else
    {
        printf("Incorrect LRN Mode\n");
        exit(0);
    }

    miopenSetLRNDescriptor(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
    return (0);
}

template <typename T>
int LRNDriver<T>::AllocateBuffersAndCopy()
{

    size_t workspaceSize = 0;
    miopenLRNGetWorkSpaceSize(outputTensor, &workspaceSize);

    size_t in_sz  = GetTensorSize(inputTensor);
    size_t out_sz = GetTensorSize(outputTensor);
#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

    if(inflags.GetValueInt("forw") == 0)
    {
        scale_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workspaceSize / sizeof(float), sizeof(float)));
    }

    in      = std::vector<float>(in_sz);
    out     = std::vector<float>(out_sz, 0);
    outhost = std::vector<float>(out_sz, 0);
    if(inflags.GetValueInt("forw") == 0)
    {
        scale     = std::vector<float>(workspaceSize / sizeof(float), 0);
        scalehost = std::vector<float>(workspaceSize / sizeof(float), 0);
    }
    din     = std::vector<float>(in_sz);
    dout    = std::vector<float>(out_sz, 0);
    dinhost = std::vector<float>(in_sz, 0);

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = static_cast<T>((static_cast<double>(rand()) * (1.0 / RAND_MAX)));
    }

    for(int i = 0; i < out_sz; i++)
    {
        dout[i] = static_cast<T>((static_cast<double>((rand()) * (1.0 / RAND_MAX) - 0.5) * 0.001));
    }
#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
    int status;
#endif
    status = in_dev->ToGPU(q, in.data());
    if(inflags.GetValueInt("forw") == 0)
    {
        status |= scale_dev->ToGPU(q, scale.data());
    }
    status |= out_dev->ToGPU(q, out.data());

    status = din_dev->ToGPU(q, din.data());
    status |= dout_dev->ToGPU(q, dout.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename T>
int LRNDriver<T>::RunForwardGPU()
{

    int alpha = 1, beta = 1;

    miopenLRNForward(GetHandle(),
                     lrnDesc,
                     &alpha,
                     inputTensor,
                     in_dev->GetMem(),
                     &beta,
                     outputTensor,
                     out_dev->GetMem(),
                     (inflags.GetValueInt("forw") == 0) ? true : false,
                     (inflags.GetValueInt("forw") == 0) ? scale_dev->GetMem() : NULL);

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenLRNForward(GetHandle(),
                         lrnDesc,
                         &alpha,
                         inputTensor,
                         in_dev->GetMem(),
                         &beta,
                         outputTensor,
                         out_dev->GetMem(),
                         (inflags.GetValueInt("forw") == 0) ? true : false,
                         (inflags.GetValueInt("forw") == 0) ? scale_dev->GetMem() : NULL);
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward LRN Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Forward LRN Elapsed: %f ms\n", time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    if(inflags.GetValueInt("forw") == 0)
    {
        scale_dev->FromGPU(GetStream(), scale.data());
    }

    return miopenStatusSuccess;
}

template <typename T>
int LRNDriver<T>::RunForwardCPU()
{
    int nInStride, cInStride, hInStride, wInStride;
    miopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
    int nIn, cIn, hIn, wIn;
    miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
    int nOutStride, cOutStride, hOutStride, wOutStride;
    miopenGet4dTensorDescriptorStrides(
        outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
    int nOut, cOut, hOut, wOut;
    miopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);
    miopenLRNMode_t v_mode;
    unsigned int v_lrnN;
    double v_lrnAlpha;
    double v_lrnBeta;
    double v_lrnK;

    miopenGetLRNDescriptor(lrnDesc, &v_mode, &v_lrnN, &v_lrnAlpha, &v_lrnBeta, &v_lrnK);

    float alphaoverarea = static_cast<float>(
        (v_mode == miopenLRNCrossChannel) ? v_lrnAlpha / v_lrnN : v_lrnAlpha / (v_lrnN * v_lrnN));

    int pre_pad = (v_lrnN - 1) / 2;
    int pad     = v_lrnN - pre_pad - 1;

    int batch_sz           = nIn;
    int n_inputs           = cIn;
    int bot_height         = hIn;
    int bot_width          = wIn;
    int bot_stride         = hInStride;
    int bot_channel_stride = cInStride;
    int bot_batch_stride   = nInStride;

    int n_outputs  = cOut;
    int top_height = hOut;
    int top_width  = wOut;

    int top_v_stride           = hOutStride;
    int top_v_channel_stride   = cOutStride;
    int top_v_batch_stride     = nOutStride;
    int scale_v_stride         = top_v_stride;
    int scale_v_channel_stride = top_v_channel_stride;
    int scale_v_batch_stride   = top_v_batch_stride;

    mloLRNForwardRunHost<T>((inflags.GetValueInt("forw") == 0) ? true : false,
                            v_mode,
                            pad,
                            v_lrnN,
                            alphaoverarea,
                            static_cast<T>(v_lrnAlpha),
                            static_cast<T>(v_lrnBeta),
                            static_cast<T>(v_lrnK),
                            batch_sz,
                            n_outputs,
                            n_inputs,
                            bot_height,
                            bot_width,
                            bot_stride,
                            bot_channel_stride,
                            bot_batch_stride,
                            top_height,
                            top_width,
                            top_v_stride,
                            top_v_channel_stride,
                            top_v_batch_stride,
                            scale_v_stride,
                            scale_v_channel_stride,
                            scale_v_batch_stride,
                            in.data(),
                            scalehost.data(),
                            outhost.data());
    return (0);
}

template <typename T>
int LRNDriver<T>::RunBackwardGPU()
{
    float alpha = 1., beta = 1.;

    miopenLRNBackward(GetHandle(),
                      lrnDesc,
                      &alpha,
                      outputTensor,
                      out_dev->GetMem(),
                      dOutputTensor,
                      dout_dev->GetMem(),
                      inputTensor,
                      in_dev->GetMem(),
                      &beta,
                      dInputTensor,
                      din_dev->GetMem(),
                      scale_dev->GetMem());

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenLRNBackward(GetHandle(),
                          lrnDesc,
                          &alpha,
                          outputTensor,
                          out_dev->GetMem(),
                          dOutputTensor,
                          dout_dev->GetMem(),
                          inputTensor,
                          in_dev->GetMem(),
                          &beta,
                          dInputTensor,
                          din_dev->GetMem(),
                          scale_dev->GetMem());
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward LRN Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Backward LRN Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());
    return (0);
}

template <typename T>
int LRNDriver<T>::VerifyForward()
{

    RunForwardCPU();

    auto error             = miopen::rms_range(outhost, out);
    const double tolerance = 1e-6;
    if(error > tolerance)
    {
        std::cout << "Forward LRN Failed: " << error << "\n";
    }
    else
    {
        printf("Forward LRN Verifies on CPU and GPU\n");
    }

    return 0;
}

template <typename T>
int LRNDriver<T>::RunBackwardCPU()
{

    int nInStride, cInStride, hInStride, wInStride;
    miopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
    int nIn, cIn, hIn, wIn;
    miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
    int nOutStride, cOutStride, hOutStride, wOutStride;
    miopenGet4dTensorDescriptorStrides(
        outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
    int nOut, cOut, hOut, wOut;
    miopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);

    int ndInStride, cdInStride, hdInStride, wdInStride;
    miopenGet4dTensorDescriptorStrides(
        dInputTensor, &ndInStride, &cdInStride, &hdInStride, &wdInStride);
    int ndIn, cdIn, hdIn, wdIn;
    miopenGet4dTensorDescriptorLengths(dInputTensor, &ndIn, &cdIn, &hdIn, &wdIn);
    int ndOutStride, cdOutStride, hdOutStride, wdOutStride;
    miopenGet4dTensorDescriptorStrides(
        dOutputTensor, &ndOutStride, &cdOutStride, &hdOutStride, &wdOutStride);
    int ndOut, cdOut, hdOut, wdOut;
    miopenGet4dTensorDescriptorLengths(dOutputTensor, &ndOut, &cdOut, &hdOut, &wdOut);

    miopenLRNMode_t v_mode;
    unsigned int v_lrnN;
    double v_lrnAlpha;
    double v_lrnBeta;
    double v_lrnK;

    miopenGetLRNDescriptor(lrnDesc, &v_mode, &v_lrnN, &v_lrnAlpha, &v_lrnBeta, &v_lrnK);

    float alphaoverarea = static_cast<float>(
        (v_mode == miopenLRNCrossChannel) ? v_lrnAlpha / v_lrnN : v_lrnAlpha / (v_lrnN * v_lrnN));

    int pre_pad = (v_lrnN - 1) / 2;
    int pad     = v_lrnN - pre_pad - 1;

    int batch_sz           = nIn;
    int n_inputs           = cIn;
    int bot_height         = hIn;
    int bot_width          = wIn;
    int bot_stride         = hInStride;
    int bot_channel_stride = cInStride;
    int bot_batch_stride   = nInStride;

    int bot_df_v_stride         = hdInStride;
    int bot_df_v_channel_stride = cdInStride;
    int bot_df_v_batch_stride   = ndInStride;

    int n_outputs          = cOut;
    int top_height         = hOut;
    int top_width          = wOut;
    int top_stride         = hOutStride;
    int top_channel_stride = cOutStride;
    int top_batch_stride   = nOutStride;

    int top_df_stride         = hdOutStride;
    int top_df_channel_stride = cdOutStride;
    int top_df_batch_stride   = ndOutStride;

    int scale_stride         = top_df_stride;
    int scale_channel_stride = top_df_channel_stride;
    int scale_batch_stride   = top_df_batch_stride;

    mloLRNBackwardRunHost<float>(static_cast<int>(v_mode),
                                 pad,
                                 v_lrnN,
                                 alphaoverarea,
                                 static_cast<float>(v_lrnAlpha),
                                 static_cast<float>(v_lrnBeta),
                                 static_cast<float>(v_lrnK),
                                 batch_sz,
                                 n_outputs,
                                 n_inputs,
                                 bot_height,
                                 bot_width,
                                 bot_stride,
                                 bot_channel_stride,
                                 bot_batch_stride,
                                 bot_df_v_stride,
                                 bot_df_v_channel_stride,
                                 bot_df_v_batch_stride,
                                 top_height,
                                 top_width,
                                 top_stride,
                                 top_channel_stride,
                                 top_batch_stride,
                                 top_df_stride,
                                 top_df_channel_stride,
                                 top_df_batch_stride,
                                 scale_stride,
                                 scale_channel_stride,
                                 scale_batch_stride,
                                 out.data(),
                                 dout.data(),
                                 scale.data(),
                                 in.data(),
                                 dinhost.data());

    return 0;
}

template <typename T>
int LRNDriver<T>::VerifyBackward()
{

    RunBackwardCPU();

    auto error             = miopen::rms_range(dinhost, din);
    const double tolerance = 1e-6;
    if(error > tolerance)
    {
        std::cout << "Backward LRN Failed: " << error << "\n";
    }
    else
    {
        printf("Backward LRN Verifies on CPU and GPU\n");
    }

    return 0;
}

#endif // GUARD_MIOPEN_CONV_DRIVER_HPP
