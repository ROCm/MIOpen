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
#ifndef GUARD_MIOPEN_CONVINFER_DRIVER_HPP
#define GUARD_MIOPEN_CONVINFER_DRIVER_HPP

#include "../test/verify.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen_BatchNormHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include "random.hpp"

#define MIO_BN_DEBUG 0
#define MIO_BN_MAX_DEBUGLOOP 65536

#define EPSILON 1e-3

#define ERRTOL 1e-4
#define RMSTOL_FP32 1e-4
#define RMSTOL_FP16 0.5e-3

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

//#define BN_RUNFOR_PROFILER

template <typename Tgpu, typename Tref>
class ConvInferDriver : public Driver
{
    public:
    ConvInferDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasScaleTensor);
        miopenCreateTensorDescriptor(&dxOutputTensor);
        miopenCreateTensorDescriptor(&dyInputTensor);

        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetModeFromCmdLine();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU();
    int RunBackwardCPU();

    void runGPUBNFwdInference(Tref epsilon, float alpha, float beta);
    void runCPUBNFwdInference(Tref epsilon, int batch_sz, int channels, int height, int width);

    void runGPUActivFwdInference();
    // void runCPUActivFwdInference();

    int VerifyBackward();
    int VerifyForward();

    ~ConvInferDriver()
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(biasScaleTensor);
        miopenDestroyTensorDescriptor(dxOutputTensor);
        miopenDestroyTensorDescriptor(dyInputTensor);
    }

    private:
    miopenBatchNormMode_t bn_mode;
    bool saveMeanVar;
    bool bsaveMeanVar;
    bool keepRunningMeanVar;
    bool estimatedMeanVar;

    unsigned char back;

    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t biasScaleTensor;
    miopenTensorDescriptor_t outputTensor;

    miopenActivationDescriptor_t activDesc;

    // Backwards
    miopenTensorDescriptor_t dyInputTensor;
    miopenTensorDescriptor_t dxOutputTensor;

    std::unique_ptr<GPUMem> dyin_dev; // this is the output of fwd
    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> scale_dev;
    std::unique_ptr<GPUMem> bias_dev;

    std::unique_ptr<GPUMem> dxout_dev;
    std::unique_ptr<GPUMem> dscale_dev;
    std::unique_ptr<GPUMem> dbias_dev;

    std::unique_ptr<GPUMem> runningMean_dev;
    std::unique_ptr<GPUMem> runningVariance_dev;
    std::unique_ptr<GPUMem> saveMean_dev;
    std::unique_ptr<GPUMem> saveInvVariance_dev;

    std::vector<Tgpu> dyin; // output of forward
    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> out_host;
    std::vector<Tgpu> dxout;
    std::vector<Tref> dxout_host;

    std::vector<Tgpu> scale;
    std::vector<Tgpu> scale_host;
    std::vector<Tgpu> bias;
    std::vector<Tgpu> bias_host;

    std::vector<Tgpu> dscale;
    std::vector<Tref> dscale_host;
    std::vector<Tgpu> dbias;
    std::vector<Tref> dbias_host;

    std::vector<Tgpu> runningMean;
    std::vector<Tgpu> runningVariance;
    std::vector<Tref> runningMean_host;
    std::vector<Tref> runningVariance_host;

    std::vector<Tgpu> saveMean;
    std::vector<Tgpu> saveInvVariance;

    std::vector<Tref> saveMean_host;
    std::vector<Tref> saveInvVariance_host;

    int createSaveBuffers();
    int createRunningBuffers();
    Tref maxval;
};

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::GetandSetData()
{

    SetBNParametersFromCmdLineArgs();

    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    std::vector<int> sb_len;
    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        sb_len.push_back(1);
        sb_len.push_back(in_len[1]);
        sb_len.push_back(in_len[2]);
        sb_len.push_back(in_len[3]);
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        sb_len.push_back(1);
        sb_len.push_back(in_len[1]);
        sb_len.push_back(1);
        sb_len.push_back(1);
    }

    SetTensor4d(inputTensor, in_len, data_type);
    SetTensor4d(biasScaleTensor, sb_len, data_type);
    SetTensor4d(outputTensor, in_len, data_type);

    // backwards
    SetTensor4d(dyInputTensor, in_len, data_type);
    SetTensor4d(dxOutputTensor, in_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("batchsize", 'n', "32", "Mini-batch size (Default=32)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("alpha", 'A', "1.0", "Alpha (Default=1.0)", "float");
    inflags.AddInputFlag("beta", 'B', "0.", "Beta (Default=0.)", "float");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag("mode",
                         'm',
                         "0",
                         "Normalization Mode (per-activation (0) or spatial (1)) (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "save",
        's',
        "0",
        "Save off mean and inverse variance, or on backprop, use these values. (Default=0)",
        "int");
    inflags.AddInputFlag(
        "run",
        'r',
        "0",
        "Keep running mean and variance, or on inference, use these values. (Default=0)",
        "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvInferDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");
    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::SetBNParametersFromCmdLineArgs()
{

    //    	double bnAlpha = inflags.GetValueDouble("alpha");
    //    	double bnBeta = inflags.GetValueDouble("beta");

    // batch norm mode type
    if(inflags.GetValueInt("mode") == 0)
    {
        bn_mode = miopenBNPerActivation;
    }
    else if(inflags.GetValueInt("mode") == 1)
    {
        bn_mode = miopenBNSpatial;
    }
    else
    {
        printf("Incorrect Batch Normalization Mode\n");
        exit(EXIT_FAILURE);
    }

    // save off mean and variance?
    if(inflags.GetValueInt("save") == 0)
    {
        saveMeanVar = false;
    }
    else if(inflags.GetValueInt("save") == 1)
    {
        saveMeanVar = true;
    }
    else
    {
        printf("Incorrect Batch Normalization Save mode\n");
        exit(EXIT_FAILURE);
    }

    // keep running mean and variance
    if(inflags.GetValueInt("run") == 0)
    {
        keepRunningMeanVar = false;
    }
    else if(inflags.GetValueInt("run") == 1)
    {
        keepRunningMeanVar = true;
    }
    else
    {
        printf("Incorrect Batch Normalization Running mode\n");
        exit(EXIT_FAILURE);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::createSaveBuffers()
{

#if MIOPEN_BACKEND_OPENCL
    cl_int status = CL_SUCCESS;
    cl_context ctx;
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    int status   = 0;
    uint32_t ctx = 0;
#endif

    size_t sb_sz = GetTensorSize(biasScaleTensor);

    if(saveMeanVar)
    {
        // GPU allocation
        saveMean_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
        saveInvVariance_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));

        // GPU host allocation
        saveMean        = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
        saveInvVariance = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

        // CPU allocation
        saveMean_host        = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
        saveInvVariance_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));

        // GPU data transfer
        status |= saveMean_dev->ToGPU(q, saveMean.data());
        status |= saveInvVariance_dev->ToGPU(q, saveInvVariance.data());
    }

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::createRunningBuffers()
{

#if MIOPEN_BACKEND_OPENCL
    cl_int status = CL_SUCCESS;
    cl_context ctx;
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    int status   = 0;
    uint32_t ctx = 0;
#endif
    size_t sb_sz = GetTensorSize(biasScaleTensor);

    if(keepRunningMeanVar)
    {
        // GPU allocation
        runningMean_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
        runningVariance_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));

        // GPU host allocation
        runningMean     = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
        runningVariance = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

        // CPU allocation
        runningMean_host     = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
        runningVariance_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));

        // Populate
        for(int i = 0; i < sb_sz; i++)
        {
            runningMean_host[i] = runningMean[i] =
                RAN_GEN<Tref>(static_cast<Tref>(0.0), static_cast<Tref>(1.0));
            runningVariance_host[i] = runningVariance[i] =
                RAN_GEN<Tref>(static_cast<Tref>(0.0), static_cast<Tref>(1.0));
        }

        // GPU data transfer
        status |= runningMean_dev->ToGPU(q, runningMean.data());
        status |= runningVariance_dev->ToGPU(q, runningVariance.data());
    }
    else
    {
        runningMean_dev     = nullptr;
        runningVariance_dev = nullptr;
    }
    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{

#if MIOPEN_BACKEND_OPENCL
    cl_int status = CL_SUCCESS;
    cl_context ctx;
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    int status   = 0;
    uint32_t ctx = 0;
#endif

    size_t in_sz = GetTensorSize(inputTensor);
    size_t sb_sz = GetTensorSize(biasScaleTensor);

    size_t out_sz = GetTensorSize(outputTensor);

    // GPU allocation
    in_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    scale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
    bias_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
    out_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    // GPU host allocation
    in    = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out   = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    scale = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
    bias  = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

    // CPU allocation
    out_host   = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    scale_host = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
    bias_host  = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

    // Data initialization
    for(int i = 0; i < in_sz; i++)
    {
        in[i] = std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0)));
    }
    status |= in_dev->ToGPU(q, in.data());

    // Using random beta and gamma
    for(int i = 0; i < sb_sz; i++)
    {
        scale[i] = scale_host[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        bias[i] = bias_host[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= scale_dev->ToGPU(q, scale.data());
    status |= bias_dev->ToGPU(q, bias.data());
    status |= out_dev->ToGPU(q, out.data());

    { // inference
        status |= createRunningBuffers();
    }

    if(status != CL_SUCCESS)
        printf("Fatal: Error copying data to GPU\nExiting...\n\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void ConvInferDriver<Tgpu, Tref>::runGPUBNFwdInference(Tref epsilon, float alpha, float beta)
{

    if(keepRunningMeanVar)
    { // use precalculated mean and variance
        miopenBatchNormalizationForwardInference(GetHandle(),
                                                 bn_mode,
                                                 &alpha,
                                                 &beta,
                                                 inputTensor,
                                                 in_dev->GetMem(),
                                                 outputTensor,
                                                 out_dev->GetMem(),
                                                 biasScaleTensor,
                                                 scale_dev->GetMem(),
                                                 bias_dev->GetMem(),
                                                 runningMean_dev->GetMem(),
                                                 runningVariance_dev->GetMem(),
                                                 epsilon);
    }
    else
    { // recalculate mean and variance
        miopenBatchNormalizationForwardInference(GetHandle(),
                                                 bn_mode,
                                                 &alpha,
                                                 &beta,
                                                 inputTensor,
                                                 in_dev->GetMem(),
                                                 outputTensor,
                                                 out_dev->GetMem(),
                                                 biasScaleTensor,
                                                 scale_dev->GetMem(),
                                                 bias_dev->GetMem(),
                                                 nullptr,
                                                 nullptr,
                                                 epsilon);
    }

    return;
}

template <typename Tgpu, typename Tref>
void ConvInferDriver<Tgpu, Tref>::runGPUActivFwdInference()
{

    float alpha = 1, beta = 0;

    miopenActivationForward(GetHandle(),
                            activDesc,
                            &alpha,
                            inputTensor,
                            in_dev->GetMem(),
                            &beta,
                            outputTensor,
                            out_dev->GetMem());

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        printf("GPU Kernel Time Forward Activation Elapsed: %f ms\n", time);
    }

    out_dev->FromGPU(GetStream(), out.data());
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::RunForwardGPU()
{

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    Tref epsilon = static_cast<Tref>(EPSILON);

    Timer t;
    double fulltime = 0.;
    auto iters      = inflags.GetValueInt("iter");
    float lowtime   = 100000000.0;
    float avgtime   = 0.;

    for(int i = 0; i < iters; i++)
    {

        START_TIME;

        printf("Running for inference.\n");
        runGPUBNFwdInference(epsilon, alpha, beta);

        miopen::deref(GetHandle()).Finish();
        STOP_TIME;

        if(WALL_CLOCK)
        {
            if(iters > 1 && i > 0)
                fulltime += t.gettime_ms();
            else if(iters == 1)
                fulltime = t.gettime_ms();
            // else do nothing, drop the first iteration
        }

        if(inflags.GetValueStr("time") == "1")
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            lowtime = (time < lowtime) ? time : lowtime;
            if(iters > 1 && i > 0)
                avgtime += time;
        }
    }

    if(WALL_CLOCK)
    {
        printf("Wall-clock Time Forward GPU Batch Norm Elapsed: %f ms, for %d iterations.\n",
               (iters == 1) ? t.gettime_ms() : (fulltime / float(iters - 1)),
               (iters > 1) ? iters - 1 : 1);
    }

    if(inflags.GetValueStr("time") == "1")
    {
        printf("GPU Kernel Min Time Forward Batch Normalization Elapsed: %f ms\n", lowtime);
        if(iters > 1)
            printf("GPU Kernel Avg Time Forward Batch Normalization Elapsed: %f ms, for %d "
                   "iterations.\n",
                   avgtime / (iters - 1),
                   iters - 1);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void ConvInferDriver<Tgpu, Tref>::runCPUBNFwdInference(
    Tref epsilon, int batch_sz, int channels, int height, int width)
{

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        miopenBNFwdInferPerActivationRunHost(/* alpha, beta, */ batch_sz,
                                             channels,
                                             height,
                                             width,
                                             in.data(),
                                             out_host.data(),
                                             scale_host.data(),
                                             bias_host.data(),
                                             epsilon,
                                             keepRunningMeanVar,
                                             runningMean_host.data(),
                                             runningVariance_host.data());
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        miopenBNFwdInferSpatialRunHost(/* alpha, beta, */ batch_sz,
                                       channels,
                                       height,
                                       width,
                                       in.data(),
                                       out_host.data(),
                                       scale_host.data(),
                                       bias_host.data(),
                                       epsilon,
                                       keepRunningMeanVar,
                                       runningMean_host.data(),
                                       runningVariance_host.data());
    }
    else
    {
        printf("Something went wrong.\nBad batch normalization mode in host kernel "
               "selection.\nExiting...\n\n");
        exit(EXIT_FAILURE);
    }
    return;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::RunForwardCPU()
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

    int batch_sz = nIn;
    int channels = cIn;
    int height   = hIn;
    int width    = wIn;

    //	T alpha = 0., beta  = 0.;
    Tref epsilon = static_cast<Tref>(EPSILON);

    // inference only
    runCPUBNFwdInference(epsilon, /* alpha, beta,*/ batch_sz, channels, height, width);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::VerifyForward()
{

    const Tref maxrms = static_cast<Tref>((sizeof(Tgpu) == 4) ? RMSTOL_FP32 : RMSTOL_FP16);

#if(MIO_BN_DEBUG == 1)
    const Tref tolerance = static_cast<Tref>(ERRTOL);
    Tref diff            = static_cast<Tref>(0.);
#endif

    bool anError = false;

    RunForwardCPU();

    // Check output tensor error
    out_dev->FromGPU(GetStream(), out.data());
    maxval        = static_cast<Tref>(0.0);
    auto errorOut = miopen::rms_range(out_host, out);
    if(errorOut > maxrms || std::isnan(errorOut))
    {
        std::cout << "Forward batch norm verification failed on output: " << errorOut << "\n";
        anError = true;
#if(MIO_BN_DEBUG == 1)
        unsigned int count = 0;
        for(int i = 0; i < out.size() && i < out_host.size(); i++)
        {
            if(std::isnan(out[i]))
            {
                std::cout << "out[" << i << "] produced a nan: " << out[i] << std::endl;
            }
            if(std::isnan(out_host[i]))
            {
                std::cout << "out_host[" << i << "] produced a nan: " << out_host[i] << std::endl;
            }
            diff   = Tref(fabs(out[i]) - fabs(out_host[i]));
            maxval = maxval < diff ? diff : maxval;
            if(diff > tolerance)
            {
                std::cout << "out[" << i << "]: " << out[i];
                std::cout << ", out_host[" << i << "]: " << out_host[i];
                std::cout << ", diff[" << i << "]: " << Tref(out[i] - out_host[i]) << std::endl;
                count++;
            }
        }

        std::cout << "Number of elements: " << out.size() << std::endl;
        std::cout << "Number of bad elements: " << count << std::endl;
        std::cout << "max difference in output: " << maxval << std::endl;
#endif
    }
    else
    {
        std::cout << "Forward batch norm verification passed on output\n";
    }

    // Done! Results?
    if(!anError)
    {
        std::cout << "Forward Batch Norm Verifies on CPU and GPU." << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvInferDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BN_DRIVER_HPP
