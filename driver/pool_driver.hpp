/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

template <typename Tgpu, typename Tref, typename Index>
class PoolDriver_impl : public Driver
{
    public:
    PoolDriver_impl() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);

        miopenCreatePoolingDescriptor(&poolDesc);
        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int SetPoolDescriptorFromCmdLineArgs();

    std::vector<int> GetOutputTensorLengths();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    int VerifyBackward() override;
    int VerifyForward() override;
    ~PoolDriver_impl() override
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
    std::vector<Index> mask;

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

    int spatial_dim;
};

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    do_backward = !(inflags.GetValueInt("forw"));

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::GetandSetData()
{
    auto input  = inflags.GetValueTensor("input");
    spatial_dim = input.lengths.size() - 2;

    if(input.SetTensordDescriptor(inputTensor, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");

    auto dinput = inflags.GetValueTensor("dinput").FillMissing(input);
    if(dinput.SetTensordDescriptor(dInputTensor, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing dinput tensor: " + inflags.GetValueStr("dinput") + ".");

    SetPoolDescriptorFromCmdLineArgs();

    auto output = inflags.GetValueTensor("output");
    if(output.lengths.empty())
        output.lengths = GetOutputTensorLengths();
    if(output.layout.empty() && output.strides.empty())
        output.layout = input.layout;

    if(output.SetTensordDescriptor(outputTensor, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor: " + inflags.GetValueStr("output") + ".");

    auto doutput = inflags.GetValueTensor("doutput").FillMissing(output);
    if(doutput.SetTensordDescriptor(dOutputTensor, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing doutput tensor: " + inflags.GetValueStr("doutput") + ".");

    return (0);
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::AddCmdLineArgs()
{
    inflags.AddTensorFlag("input", 'W', "100x3x32x32,NCHW");
    inflags.AddTensorFlag("dinput", 'D', "", "input tensor descriptor");
    inflags.AddTensorFlag("output", 'O', "", "generated from input tensor descriptor");
    inflags.AddTensorFlag("doutput", 'H', "", "output tensor descriptor");

    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Pooling (Default=0)", "int");
    inflags.AddInputFlag("win_d", 'Z', "3", "Window Depth (Default=3)", "int");
    inflags.AddInputFlag("win_h", 'y', "3", "Window Height (Default=3)", "int");
    inflags.AddInputFlag("win_w", 'x', "3", "Window Width (Default=3)", "int");
    inflags.AddInputFlag("pool_stride_d", 's', "1", "Pooling Stride Depth (Default=1)", "int");
    inflags.AddInputFlag("pool_stride_h", 'v', "1", "Pooling Stride Height (Default=1)", "int");
    inflags.AddInputFlag("pool_stride_w", 'u', "1", "Pooling Stride Width (Default=1)", "int");
    inflags.AddInputFlag("pad_d", 'o', "0", "Zero Padding Depth (Default=0)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    inflags.AddInputFlag(
        "index_position", 'M', "0", "Image index 1, mask index 0 (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("print", 'P', "1", "Print Pooling Dimensions (Default=1)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "max", "Pooling Mode (max, avg, avg_in) (Default=max)", "str");
    inflags.AddInputFlag(
        "pad_mode", 'z', "default", "Padding Mode (same, valid, default) (Default=default)", "str");
    inflags.AddInputFlag("index_type",
                         'I',
                         "miopenIndexUint8",
                         "Index Data Type (miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, "
                         "miopenIndexUint64) (Default=miopenIndexUint8)",
                         "str");

    return 0;
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::SetPoolDescriptorFromCmdLineArgs()
{

    miopenPoolingMode_t mode;
    miopenPaddingMode_t pmode    = miopenPaddingDefault;
    miopenIndexType_t index_type = miopenIndexUint8;
    int pad_d                    = inflags.GetValueInt("pad_d");
    int pad_h                    = inflags.GetValueInt("pad_h");
    int pad_w                    = inflags.GetValueInt("pad_w");
    int stride_d                 = inflags.GetValueInt("pool_stride_d");
    int stride_h                 = inflags.GetValueInt("pool_stride_h");
    int stride_w                 = inflags.GetValueInt("pool_stride_w");
    int win_d                    = inflags.GetValueInt("win_d");
    int win_h                    = inflags.GetValueInt("win_h");
    int win_w                    = inflags.GetValueInt("win_w");
    if((inflags.GetValueStr("mode")) == "max")
    {
        mode = miopenPoolingMax;
    }
    else if((inflags.GetValueStr("mode")) == "avg")
    {
        mode = miopenPoolingAverage;
    }
    else if((inflags.GetValueStr("mode")) == "avg_in")
    {
        mode = miopenPoolingAverageInclusive;
    }
    else
    {
        printf("Incorrect Pooling Mode\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
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
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    if((inflags.GetValueStr("index_type")) == "miopenIndexUint8")
    {
        index_type = miopenIndexUint8;
    }
    else if((inflags.GetValueStr("index_type")) == "miopenIndexUint16")
    {
        index_type = miopenIndexUint16;
    }
    else if((inflags.GetValueStr("index_type")) == "miopenIndexUint32")
    {
        index_type = miopenIndexUint32;
    }
    else if((inflags.GetValueStr("index_type")) == "miopenIndexUint64")
    {
        index_type = miopenIndexUint64;
    }
    else
    {
        printf("Incorrect Index Data Type\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    std::initializer_list<int> lens    = {win_d, win_h, win_w};
    std::initializer_list<int> pads    = {pad_d, pad_h, pad_w};
    std::initializer_list<int> strides = {stride_d, stride_h, stride_w};
    miopen::deref(poolDesc)            = miopen::PoolingDescriptor(mode,
                                                        pmode,
                                                        lens.begin() + 3 - spatial_dim,
                                                        pads.begin() + 3 - spatial_dim,
                                                        strides.begin() + 3 - spatial_dim,
                                                        spatial_dim);

    miopen::deref(poolDesc).SetIndexType(index_type);

    miopenSetPoolingWorkSpaceIndexMode(
        poolDesc,
        miopenPoolingWorkspaceIndexMode_t(
            spatial_dim == 3 ? 1 : inflags.GetValueInt("index_position")));

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Index>
std::vector<int> PoolDriver_impl<Tgpu, Tref, Index>::GetOutputTensorLengths()
{
    std::vector<int> out_dim(spatial_dim + 2);
    miopenGetPoolingNdForwardOutputDim(poolDesc, inputTensor, spatial_dim + 2, out_dim.data());

    return out_dim;
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::AllocateBuffersAndCopy()
{

    size_t in_sz         = GetTensorSize(inputTensor);
    size_t out_sz        = GetTensorSize(outputTensor);
    size_t workSpaceSize = 0;
    miopenPoolingGetWorkSpaceSizeV2(poolDesc, outputTensor, &workSpaceSize);

    size_t workSpaceNbVal =
        workSpaceSize /
        sizeof(Index); // work space is used by mask_dev and mask which are of type Index

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    mask_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal, sizeof(Index)));
    mask     = std::vector<Index>(workSpaceNbVal, Index(0));

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

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::RunForwardGPU()
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
    START_TIME

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

        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward Pooling Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        printf("GPU Kernel Time Forward Pooling Elapsed: %f ms\n", time);
    }

    out_dev->FromGPU(GetStream(), out.data());
    mask_dev->FromGPU(GetStream(), mask.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::RunForwardCPU()
{
    return (0);
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::RunBackwardGPU()
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
    START_TIME

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

        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward Pooling Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Backward Pooling Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::VerifyForward()
{
    int nInStride, cInStride, dInStride, hInStride, wInStride;
    int nIn, cIn, dIn, hIn, wIn;
    int nOutStride, cOutStride, dOutStride, hOutStride, wOutStride;
    int nOut, cOut, dOut, hOut, wOut;
    miopenPoolingMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(poolDesc).pmode;
    int windowDepth, windowHeight, windowWidth;
    int pad_d, pad_h, pad_w;
    int stride_d, stride_h, stride_w;

    if(spatial_dim == 2)
    {
        miopenGet4dTensorDescriptorStrides(
            inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
        dInStride = hInStride;
        miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
        dIn = 1;
        miopenGet4dTensorDescriptorStrides(
            outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
        dOutStride = hOutStride;
        miopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);
        dOut = 1;

        miopenGet2dPoolingDescriptor(
            poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &stride_h, &stride_w);
        windowDepth = 1;
        pad_d       = 0;
        stride_d    = 1;
    }
    else if(spatial_dim == 3)
    {
        std::vector<int> winV(spatial_dim);
        std::vector<int> padV(spatial_dim);
        std::vector<int> strV(spatial_dim);

        miopenGet5dTensorDescriptorStrides(
            inputTensor, &nInStride, &cInStride, &dInStride, &hInStride, &wInStride);
        miopenGet5dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &dIn, &hIn, &wIn);
        miopenGet5dTensorDescriptorStrides(
            outputTensor, &nOutStride, &cOutStride, &dOutStride, &hOutStride, &wOutStride);
        miopenGet5dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &dOut, &hOut, &wOut);

        miopenGetNdPoolingDescriptor(
            poolDesc, spatial_dim, &mode, nullptr, winV.data(), padV.data(), strV.data());
        std::tie(windowDepth, windowHeight, windowWidth) = miopen::tien<3>(winV);
        std::tie(pad_d, pad_h, pad_w)                    = miopen::tien<3>(padV);
        std::tie(stride_d, stride_h, stride_w)           = miopen::tien<3>(strV);
    }
    else
    {
        MIOPEN_THROW("Unsupported spatial dimension");
    }

    if(pmode == miopenPaddingSame)
    {
        pad_d = (dIn % stride_d == 0) ? (std::max((windowDepth - stride_d), 0))
                                      : (std::max((windowDepth - (dIn % stride_d)), 0));
        pad_h = (hIn % stride_h == 0) ? (std::max((windowHeight - stride_h), 0))
                                      : (std::max((windowHeight - (hIn % stride_h)), 0));
        pad_w = (wIn % stride_w == 0) ? (std::max((windowWidth - stride_w), 0))
                                      : (std::max((windowWidth - (wIn % stride_w)), 0));

        pad_d /= 2;
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_d = 0;
        pad_h = 0;
        pad_w = 0;
    }

    if(dOut <= 0 || hOut <= 0 || wOut <= 0)
        throw std::runtime_error("Invalid Test Case: Check Output Dimension.");

    int pooling_method =
        (mode == miopenPoolingMax)
            ? MLO_POOLING_OP_MAX
            : ((mode == miopenPoolingAverage) ? MLO_POOLING_OP_AVE : MLO_POOLING_OP_AVE_INCLUSIVE);

    const Tref tolerance = (sizeof(Tgpu) == 4 || sizeof(Tgpu) == 8) ? 1e-6 : 5e-3;
    bool match           = mloPoolingForwardRunHostAndVerify<Tgpu, Tref, Index>(
        pooling_method,
        pad_d,
        stride_d,
        windowDepth,
        pad_h,
        stride_h,
        windowHeight,
        pad_w,
        stride_w,
        windowWidth,
        inputTensor,
        outputTensor,
        in.data(),
        out.data(),
        do_backward,
        maskhost.data(),
        mask.data(),
        tolerance,
        spatial_dim == 3 ? 1 : inflags.GetValueInt("index_position"));

    printf(match ? "Forward Pooling Verifies on CPU and GPU\n"
                 : "Forward Pooling Verification Failed !!\n");

    return 0;
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::RunBackwardCPU()
{

    return 0;
}

template <typename Tgpu, typename Tref, typename Index>
int PoolDriver_impl<Tgpu, Tref, Index>::VerifyBackward()
{
    int ndInStride, cdInStride, ddInStride, hdInStride, wdInStride;
    int nIn, cIn, dIn, hIn, wIn;
    int ndOutStride, cdOutStride, ddOutStride, hdOutStride, wdOutStride;
    int nOut, cOut, dOut, hOut, wOut;
    int ndIn, cdIn, ddIn, hdIn, wdIn;
    int ndOut, cdOut, ddOut, hdOut, wdOut;
    miopenPoolingMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(poolDesc).pmode;
    int windowDepth, windowHeight, windowWidth;
    int pad_d, pad_h, pad_w;
    int stride_d, stride_h, stride_w;

    if(spatial_dim == 2)
    {
        miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
        dIn = 1;
        miopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);
        dOut = 1;
        miopenGet4dTensorDescriptorStrides(
            dInputTensor, &ndInStride, &cdInStride, &hdInStride, &wdInStride);
        ddInStride = hdInStride;
        miopenGet4dTensorDescriptorLengths(dInputTensor, &ndIn, &cdIn, &hdIn, &wdIn);
        ddIn = 1;
        miopenGet4dTensorDescriptorStrides(
            dOutputTensor, &ndOutStride, &cdOutStride, &hdOutStride, &wdOutStride);
        ddOutStride = hdOutStride;
        miopenGet4dTensorDescriptorLengths(dOutputTensor, &ndOut, &cdOut, &hdOut, &wdOut);
        ddOut = 1;

        miopenGet2dPoolingDescriptor(
            poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &stride_h, &stride_w);
        windowDepth = 1;
        pad_d       = 0;
        stride_d    = 1;
    }
    else if(spatial_dim == 3)
    {
        std::vector<int> winV(spatial_dim);
        std::vector<int> padV(spatial_dim);
        std::vector<int> strV(spatial_dim);

        miopenGet5dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &dIn, &hIn, &wIn);
        miopenGet5dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &dOut, &hOut, &wOut);
        miopenGet5dTensorDescriptorStrides(
            dInputTensor, &ndInStride, &cdInStride, &ddInStride, &hdInStride, &wdInStride);
        miopenGet5dTensorDescriptorLengths(dInputTensor, &ndIn, &cdIn, &ddIn, &hdIn, &wdIn);
        miopenGet5dTensorDescriptorStrides(
            dOutputTensor, &ndOutStride, &cdOutStride, &ddOutStride, &hdOutStride, &wdOutStride);
        miopenGet5dTensorDescriptorLengths(dOutputTensor, &ndOut, &cdOut, &ddOut, &hdOut, &wdOut);

        miopenGetNdPoolingDescriptor(
            poolDesc, spatial_dim, &mode, nullptr, winV.data(), padV.data(), strV.data());
        std::tie(windowDepth, windowHeight, windowWidth) = miopen::tien<3>(winV);
        std::tie(pad_d, pad_h, pad_w)                    = miopen::tien<3>(padV);
        std::tie(stride_d, stride_h, stride_w)           = miopen::tien<3>(strV);
    }
    else
    {
        MIOPEN_THROW("Unsupported spatial dimension");
    }

    if(dOut <= 0 || hOut <= 0 || wOut <= 0)
        throw std::runtime_error("Invalid Test Case: Check Output Dimension.");

    if(pmode == miopenPaddingSame)
    {
        pad_d = (dIn % stride_d == 0) ? (std::max((windowDepth - stride_d), 0))
                                      : (std::max((windowDepth - (dIn % stride_d)), 0));
        pad_h = (hIn % stride_h == 0) ? (std::max((windowHeight - stride_h), 0))
                                      : (std::max((windowHeight - (hIn % stride_h)), 0));
        pad_w = (wIn % stride_w == 0) ? (std::max((windowWidth - stride_w), 0))
                                      : (std::max((windowWidth - (wIn % stride_w)), 0));
        pad_d /= 2;
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_d = 0;
        pad_h = 0;
        pad_w = 0;
    }
    int pooling_method =
        (mode == miopenPoolingMax)
            ? MLO_POOLING_OP_MAX
            : ((mode == miopenPoolingAverage) ? MLO_POOLING_OP_AVE : MLO_POOLING_OP_AVE_INCLUSIVE);

    mloPoolingBackwardRunHost<Tgpu, Tref>(pooling_method,
                                          windowDepth,
                                          pad_d,
                                          stride_d,
                                          windowHeight,
                                          pad_h,
                                          stride_h,
                                          windowWidth,
                                          pad_w,
                                          stride_w,
                                          // host output
                                          dinhost.data(),
                                          dout.data(),
                                          maskhost.data(),
                                          ndInStride,
                                          cdInStride,
                                          ddInStride,
                                          hdInStride,
                                          wIn,
                                          hIn,
                                          dIn,
                                          cOut,
                                          nOut,
                                          ndOutStride,
                                          cdOutStride,
                                          ddOutStride,
                                          hdOutStride,
                                          wOut,
                                          hOut,
                                          dOut);

    bool match            = true;
    const Tref allowedEps = (1 << 2);
    Tref max_sqr          = 1. / 1000000; // 100000000;
    Tref max_abs_diff     = 1. / 1000000; // 100000000;
    bool get_error_pos    = true;

    match = mloVerify<Tgpu, Tref>(spatial_dim,
                                  nIn,
                                  cIn,
                                  dIn,
                                  hIn,
                                  wIn,
                                  ndInStride,
                                  cdInStride,
                                  ddInStride,
                                  hdInStride,
                                  ndInStride,
                                  cdInStride,
                                  ddInStride,
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

template <typename Tgpu, typename Tref>
class PoolDriver : public Driver
{
    public:
    PoolDriver() : Driver(), pool_driver_impl(nullptr) {}

    int AddCmdLineArgs() override
    {
        pool_driver_impl_uint8.AddCmdLineArgs();
        pool_driver_impl_uint16.AddCmdLineArgs();
        pool_driver_impl_uint32.AddCmdLineArgs();
        pool_driver_impl_uint64.AddCmdLineArgs();

        return 0;
    }

    int ParseCmdLineArgs(int argc, char* argv[]) override
    {
        pool_driver_impl = &pool_driver_impl_uint8;

        std::vector<std::string> as(argv + 1, argv + argc);

        if(std::any_of(as.begin(), as.end(), [](auto v) { return v == "miopenIndexUint16"; }))
        {
            pool_driver_impl = &pool_driver_impl_uint16;
        }
        else if(std::any_of(as.begin(), as.end(), [](auto v) { return v == "miopenIndexUint32"; }))
        {
            pool_driver_impl = &pool_driver_impl_uint32;
        }
        else if(std::any_of(as.begin(), as.end(), [](auto v) { return v == "miopenIndexUint64"; }))
        {
            pool_driver_impl = &pool_driver_impl_uint64;
        }

        pool_driver_impl->ParseCmdLineArgs(argc, argv);

        return 0;
    }

    InputFlags& GetInputFlags() override { return pool_driver_impl->GetInputFlags(); }
    int GetandSetData() override { return pool_driver_impl->GetandSetData(); }
    int AllocateBuffersAndCopy() override { return pool_driver_impl->AllocateBuffersAndCopy(); }
    int RunForwardGPU() override { return pool_driver_impl->RunForwardGPU(); }
    int VerifyForward() override { return pool_driver_impl->VerifyForward(); }
    int RunBackwardGPU() override { return pool_driver_impl->RunBackwardGPU(); }
    int VerifyBackward() override { return pool_driver_impl->VerifyBackward(); }

    private:
    Driver* pool_driver_impl;

    PoolDriver_impl<Tgpu, Tref, uint8_t> pool_driver_impl_uint8;
    PoolDriver_impl<Tgpu, Tref, uint16_t> pool_driver_impl_uint16;
    PoolDriver_impl<Tgpu, Tref, uint32_t> pool_driver_impl_uint32;
    PoolDriver_impl<Tgpu, Tref, uint64_t> pool_driver_impl_uint64;
};
#endif // GUARD_MIOPEN_POOL_DRIVER_HPP
