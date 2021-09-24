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
#include "random.hpp"

template <typename Tgpu, typename Tref>
class LRNDriver : public Driver
{
    public:
    LRNDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);

        miopenCreateLRNDescriptor(&lrnDesc);
        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetLRNDescriptorFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~LRNDriver() override
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

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
    std::vector<Tgpu> scale;
    std::vector<Tref> scalehost;

    miopenLRNDescriptor_t lrnDesc;
    bool do_backward;

    miopenTensorDescriptor_t dInputTensor;
    miopenTensorDescriptor_t dOutputTensor;

    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> dout_dev;

    std::vector<Tgpu> din;
    std::vector<Tgpu> dout;
    std::vector<Tref> dinhost;
};

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    auto dir_val = inflags.GetValueInt("forw");

    do_backward = (dir_val == 0) || (dir_val == 2);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
#if 0
	if(inflags.GetValueInt("back") == 0 && inflags.GetValueStr("mode") == "cross") {
		printf("Cross channel LRN needs do_backward=1\n");
		exit(0); // NOLINT (concurrency-mt-unsafe)
	}
#endif
    return 0;
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len, data_type);
    SetTensor4d(dInputTensor, in_len, data_type);
    SetLRNDescriptorFromCmdLineArgs();

    SetTensor4d(outputTensor, in_len, data_type);
    SetTensor4d(dOutputTensor, in_len, data_type);
    return (0);
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::AddCmdLineArgs()
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

template <typename Tgpu, typename Tref>
std::vector<int> LRNDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::SetLRNDescriptorFromCmdLineArgs()
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
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    return (miopenSetLRNDescriptor(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK));
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{

    size_t in_sz         = GetTensorSize(inputTensor);
    size_t out_sz        = GetTensorSize(outputTensor);
    size_t workSpaceSize = 0;
    miopenLRNGetWorkSpaceSize(outputTensor, &workSpaceSize);
    size_t workSpaceNbVal = workSpaceSize / sizeof(Tgpu);
#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    if(do_backward)
    {
        scale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal, sizeof(Tgpu)));
    }

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    if(do_backward)
    {
        scale     = std::vector<Tgpu>(workSpaceNbVal, static_cast<Tgpu>(0));
        scalehost = std::vector<Tref>(workSpaceNbVal, static_cast<Tref>(0));
        if(inflags.GetValueInt("forw") == 2)
        {
            for(int i = 0; i < scale.size(); i++)
            {
                scale[i]     = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
                scalehost[i] = Tref(scale[i]);
            }
        }
    }
    din     = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dout    = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    dinhost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    Tgpu Data_scale = static_cast<Tgpu>(0.001);
    for(int i = 0; i < out_sz; i++)
    {
        dout[i] = Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(-0.5), static_cast<Tgpu>(0.5));
    }

#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
    int status;
#endif
    status = in_dev->ToGPU(q, in.data());
    if(do_backward)
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

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::RunForwardGPU()
{

    Tgpu alpha = static_cast<Tgpu>(1), beta = static_cast<Tgpu>(0);

    miopenLRNForward(GetHandle(),
                     lrnDesc,
                     &alpha,
                     inputTensor,
                     in_dev->GetMem(),
                     &beta,
                     outputTensor,
                     out_dev->GetMem(),
                     do_backward,
                     do_backward ? scale_dev->GetMem() : nullptr);

    Timer t;
    START_TIME

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
                         do_backward,
                         do_backward ? scale_dev->GetMem() : nullptr);
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward LRN Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Forward LRN Elapsed: %f ms\n", time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    if(do_backward)
    {
        scale_dev->FromGPU(GetStream(), scale.data());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::RunForwardCPU()
{
    return (0);
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::RunBackwardGPU()
{
    Tgpu alpha = static_cast<Tgpu>(1), beta = static_cast<Tgpu>(0);

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
    START_TIME

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

        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward LRN Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Backward LRN Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::VerifyForward()
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

    Tref alphaoverarea =
        (v_mode == miopenLRNCrossChannel) ? v_lrnAlpha / v_lrnN : v_lrnAlpha / (v_lrnN * v_lrnN);

    int pre_pad = (v_lrnN - 1) / 2;
    int pad     = v_lrnN - pre_pad - 1;

    mloLRNForwardRunHost<Tgpu, Tref>(do_backward,
                                     v_mode,
                                     pad,
                                     v_lrnN,
                                     alphaoverarea,
                                     v_lrnAlpha,
                                     v_lrnBeta,
                                     v_lrnK,
                                     nIn,        // batch_sz,
                                     cOut,       // n_outputs,
                                     cIn,        // n_inputs,
                                     hIn,        // bot_height,
                                     wIn,        // bot_width,
                                     hInStride,  // bot_stride,
                                     cInStride,  // bot_channel_stride,
                                     nInStride,  // bot_batch_stride,
                                     hOut,       // top_height,
                                     wOut,       // top_width,
                                     hOutStride, // top_v_stride,
                                     cOutStride, // top_v_channel_stride,
                                     nOutStride, // top_v_batch_stride,
                                     hOutStride, // scale_v_stride,
                                     cOutStride, // scale_v_channel_stride,
                                     nOutStride, // scale_v_batch_stride,
                                     in.data(),
                                     scalehost.data(),
                                     outhost.data());

    auto error           = miopen::rms_range(outhost, out);
    const Tref tolerance = 1.5e-4; // 1e-6;
    if(error > tolerance)
    {
        std::cout << "Forward LRN Failed: " << error << "\n";
    }
    else
    {
        printf("Forward LRN Verifies on CPU and GPU (err=%f)\n", error);
    }

    return 0;
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::RunBackwardCPU()
{

    return 0;
}

template <typename Tgpu, typename Tref>
int LRNDriver<Tgpu, Tref>::VerifyBackward()
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

    Tref alphaoverarea =
        (v_mode == miopenLRNCrossChannel) ? v_lrnAlpha / v_lrnN : v_lrnAlpha / (v_lrnN * v_lrnN);

    int pre_pad = (v_lrnN - 1) / 2;
    int pad     = v_lrnN - pre_pad - 1;

    mloLRNBackwardRunHost<Tgpu, Tref>(static_cast<int>(v_mode),
                                      pad,
                                      v_lrnN,
                                      alphaoverarea,
                                      v_lrnAlpha,
                                      v_lrnBeta,
                                      v_lrnK,
                                      nIn,         // batch_sz,
                                      cOut,        // n_outputs,
                                      cIn,         // n_inputs,
                                      hIn,         // bot_height,
                                      wIn,         // bot_width,
                                      hInStride,   // bot_stride,
                                      cInStride,   // bot_channel_stride,
                                      nInStride,   // bot_batch_stride,
                                      hdInStride,  // bot_df_v_stride,
                                      cdInStride,  // bot_df_v_channel_stride,
                                      ndInStride,  // bot_df_v_batch_stride,
                                      hOut,        // top_height,
                                      wOut,        // top_width,
                                      hOutStride,  // top_stride,
                                      cOutStride,  // top_channel_stride,
                                      nOutStride,  // top_batch_stride,
                                      hdOutStride, // top_df_stride,
                                      cdOutStride, // top_df_channel_stride,
                                      ndOutStride, // top_df_batch_stride,
                                      hdOutStride, // scale_stride,
                                      cdOutStride, // scale_channel_stride,
                                      ndOutStride, // scale_batch_stride,
                                      out.data(),
                                      dout.data(),
                                      scale.data(),
                                      in.data(),
                                      dinhost.data());

    auto error           = miopen::rms_range(dinhost, din);
    const Tref tolerance = 6.0e-5;
    if(error > tolerance)
    {
        std::cout << "Backward LRN Failed: " << error << "\n";
    }
    else
    {
        printf("Backward LRN Verifies on CPU and GPU (err=%f)\n", error);
    }

    return 0;
}

#endif // GUARD_MIOPEN_CONV_DRIVER_HPP
