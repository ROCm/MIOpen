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
#ifndef GUARD_MIOPEN_POOL_DRIVER_HPP
#define GUARD_MIOPEN_POOL_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloPoolingHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/pooling.hpp>
#include <numeric>
#include <vector>
#include "random.hpp"

template <typename Tgpu, typename Tref>
class PoolDriver : public Driver
{
    public:
    PoolDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);

        miopenCreatePoolingDescriptor(&poolDesc);
        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetPoolDescriptorFromCmdLineArgs();

    std::vector<int> GetOutputTensorLengths();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU();
    int RunBackwardCPU(); // Verify implements it

    int VerifyBackward();
    int VerifyForward();
    ~PoolDriver()
    {

        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyPoolingDescriptor(poolDesc);
    }

    private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> mask_dev;
    std::vector<uint8_t> mask;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<size_t> maskhost;
    std::vector<Tref> outhost;

    miopenPoolingDescriptor_t poolDesc;
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
int PoolDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    do_backward = !(inflags.GetValueInt("forw"));

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len, data_type);
    SetTensor4d(dInputTensor, in_len, data_type);
    SetPoolDescriptorFromCmdLineArgs();

    std::vector<int> out_len = GetOutputTensorLengths();
    SetTensor4d(outputTensor, out_len, data_type);
    SetTensor4d(dOutputTensor, out_len, data_type);
    return (0);
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Pooling (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("win_h", 'y', "3", "Window Height (Default=3)", "int");
    inflags.AddInputFlag("win_w", 'x', "3", "Window Width (Default=3)", "int");
    inflags.AddInputFlag("pool_stride_0", 'u', "1", "Pooling Stride Vertical (Default=1)", "int");
    inflags.AddInputFlag("pool_stride_1", 'v', "1", "Pooling Stride Horizontal (Default=1)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("print", 'P', "1", "Print Pooling Dimensions (Default=1)", "int");
    inflags.AddInputFlag("mode", 'm', "max", "Pooling Mode (max, avg) (Default=max)", "str");
    inflags.AddInputFlag(
        "pad_mode", 'z', "default", "Padding Mode (same, valid, default) (Default=default)", "str");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> PoolDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::SetPoolDescriptorFromCmdLineArgs()
{

    miopenPoolingMode_t mode;
    miopenPaddingMode_t pmode = miopenPaddingDefault;
    int pad_h                 = inflags.GetValueInt("pad_h");
    int pad_w                 = inflags.GetValueInt("pad_w");
    int u                     = inflags.GetValueInt("pool_stride_0");
    int v                     = inflags.GetValueInt("pool_stride_1");
    int win_h                 = inflags.GetValueInt("win_h");
    int win_w                 = inflags.GetValueInt("win_w");
    if((inflags.GetValueStr("mode")) == "max")
    {
        mode  = miopenPoolingMax;
        pmode = miopenPaddingDefault;
    }
    else if((inflags.GetValueStr("mode")) == "avg")
    {
        mode  = miopenPoolingAverage;
        pmode = miopenPaddingDefault;
    }
    else
    {
        printf("Incorrect Pooling Mode\n");
        exit(0);
    }

    if((inflags.GetValueStr("pad_mode")) == "same")
    {
        pmode = miopenPaddingSame;
    }
    else if((inflags.GetValueStr("pad_mode")) == "valid")
    {
        pmode = miopenPaddingValid;
    }
    else if((inflags.GetValueStr("pad_mode")) == "default")
    {
        pmode = miopenPaddingDefault;
    }
    else
    {
        printf("Incorrect Padding Mode\n");
        exit(0);
    }
    std::initializer_list<int> lens    = {win_h, win_w};
    std::initializer_list<int> pads    = {pad_h, pad_w};
    std::initializer_list<int> strides = {u, v};
    miopen::deref(poolDesc) =
        miopen::PoolingDescriptor(mode, pmode, lens.begin(), pads.begin(), strides.begin(), 2);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> PoolDriver<Tgpu, Tref>::GetOutputTensorLengths()
{
    int n, c, h, w;

    miopenGetPoolingForwardOutputDim(poolDesc, inputTensor, &n, &c, &h, &w);

    return std::vector<int>({n, c, h, w});
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{

    size_t in_sz         = GetTensorSize(inputTensor);
    size_t out_sz        = GetTensorSize(outputTensor);
    size_t workSpaceSize = 0;
    miopenPoolingGetWorkSpaceSize(outputTensor, &workSpaceSize);
    size_t workSpaceNbVal =
        workSpaceSize /
        sizeof(uint8_t); // work space is used by mask_dev and mask which are of type uint8_t
#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    mask_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal, sizeof(uint8_t)));
    mask     = std::vector<uint8_t>(workSpaceNbVal, uint8_t(0));

    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    maskhost = std::vector<size_t>(out_sz, static_cast<size_t>(0));
    outhost  = std::vector<Tref>(out_sz, static_cast<Tref>(0));

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
    status |= out_dev->ToGPU(q, out.data());

    status = din_dev->ToGPU(q, din.data());
    status |= dout_dev->ToGPU(q, dout.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::RunForwardGPU()
{

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    miopenPoolingForward(GetHandle(),
                         poolDesc,
                         &alpha,
                         inputTensor,
                         in_dev->GetMem(),
                         &beta,
                         outputTensor,
                         out_dev->GetMem(),
                         do_backward,
                         mask_dev->GetMem(),
                         0);

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenPoolingForward(GetHandle(),
                             poolDesc,
                             &alpha,
                             inputTensor,
                             in_dev->GetMem(),
                             &beta,
                             outputTensor,
                             out_dev->GetMem(),
                             do_backward,
                             mask_dev->GetMem(),
                             0);
    }
    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward Pooling Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        printf("GPU Kernel Time Forward Pooling Elapsed: %f ms\n", time);
    }

    out_dev->FromGPU(GetStream(), out.data());
    mask_dev->FromGPU(GetStream(), mask.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::RunForwardCPU()
{
    return (0);
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    miopenPoolingBackward(GetHandle(),
                          poolDesc,
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
                          mask_dev->GetMem());

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenPoolingBackward(GetHandle(),
                              poolDesc,
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
                              mask_dev->GetMem());
    }
    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward Pooling Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Backward Pooling Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::VerifyForward()
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

    miopenPoolingMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(poolDesc).pmode;
    int windowHeight;
    int windowWidth;
    int pad_h;
    int pad_w;
    int u;
    int v;
    miopenGet2dPoolingDescriptor(
        poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &u, &v);

    if(pmode == miopenPaddingSame)
    {
        pad_h = (hIn % u == 0) ? (std::max((windowHeight - u), 0))
                               : (std::max((windowHeight - (hIn % u)), 0));
        pad_w = (wIn % v == 0) ? (std::max((windowWidth - v), 0))
                               : (std::max((windowWidth - (wIn % v)), 0));

        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_h = 0;
        pad_w = 0;
    }

    if(hOut <= 0 || wOut <= 0)
        MIOPEN_THROW("Invalid Test Case: Check Output Dimension.");

    int pooling_method = (mode == miopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;

    const Tref tolerance = (sizeof(Tgpu) == 4 || sizeof(Tgpu) == 8) ? 1e-6 : 1e-3;
    bool match           = mloPoolingForwardRunHostAndVerify<Tgpu, Tref>(pooling_method,
                                                               pad_h,
                                                               u,
                                                               windowHeight,
                                                               pad_w,
                                                               v,
                                                               windowWidth,
                                                               nIn,
                                                               cOut,
                                                               hIn,
                                                               wIn,
                                                               hInStride,
                                                               cInStride,
                                                               nInStride,
                                                               hOut,
                                                               wOut,
                                                               hOutStride,
                                                               cOutStride,
                                                               nOutStride,
                                                               in.data(),
                                                               out.data(),
                                                               do_backward,
                                                               maskhost.data(),
                                                               mask.data(),
                                                               tolerance);

    printf(match ? "Forward Pooling Verifies on CPU and GPU\n"
                 : "Forward Pooling Verification Failed !!\n");

    return 0;
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::RunBackwardCPU()
{

    return 0;
}

template <typename Tgpu, typename Tref>
int PoolDriver<Tgpu, Tref>::VerifyBackward()
{

    int nIn, cIn, hIn, wIn;
    miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
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

    miopenPoolingMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(poolDesc).pmode;
    int windowHeight;
    int windowWidth;
    int pad_h;
    int pad_w;
    int u;
    int v;
    miopenGet2dPoolingDescriptor(
        poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &u, &v);

    if(hOut <= 0 || wOut <= 0)
        MIOPEN_THROW("Invalid Test Case: Check Output Dimension.");

    if(pmode == miopenPaddingSame)
    {
        pad_h = (hIn % u == 0) ? (std::max((windowHeight - u), 0))
                               : (std::max((windowHeight - (hIn % u)), 0));
        pad_w = (wIn % v == 0) ? (std::max((windowWidth - v), 0))
                               : (std::max((windowWidth - (wIn % v)), 0));
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_h = 0;
        pad_w = 0;
    }
    int pooling_method = (mode == miopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;

    mloPoolingBackwardRunHost<Tgpu, Tref>(pooling_method,
                                          windowHeight,
                                          pad_h,
                                          u,
                                          windowWidth,
                                          pad_w,
                                          v,
                                          // host output
                                          dinhost.data(),
                                          dout.data(),
                                          maskhost.data(),
                                          ndInStride,
                                          cdInStride,
                                          hdInStride,
                                          wIn,
                                          hIn,
                                          cOut,
                                          nOut,
                                          ndOutStride,
                                          cdOutStride,
                                          hdOutStride,
                                          wOut,
                                          hOut);

    bool match            = true;
    const Tref allowedEps = (1 << 2);
    Tref max_sqr          = 1. / 1000000; // 100000000;
    Tref max_abs_diff     = 1. / 1000000; // 100000000;
    bool get_error_pos    = true;

    match = mloVerify<Tgpu, Tref>(nIn,
                                  cIn,
                                  hIn,
                                  wIn,
                                  ndInStride,
                                  cdInStride,
                                  hdInStride,
                                  ndInStride,
                                  cdInStride,
                                  hdInStride,
                                  dinhost.data(),
                                  din.data(),
                                  allowedEps,
                                  max_abs_diff,
                                  max_sqr,
                                  get_error_pos);

    if(match)
        printf("Backward Pooling Verifies on CPU and GPU\n");

    return 0;
}
#endif // GUARD_MIOPEN_POOL_DRIVER_HPP
