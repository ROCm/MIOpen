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
#ifndef GUARD_MIOPEN_CONV_BN_ACTIV_INFER_DRIVER_HPP
#define GUARD_MIOPEN_CONV_BN_ACTIV_INFER_DRIVER_HPP

#include "../test/verify.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen_BatchNormActivHost.hpp"
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
#include "mloNeuronHost.hpp"

#include <miopen/batch_norm_activ.hpp>

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
class CBAInferFusionDriver : public Driver
{
    public:
    CBAInferFusionDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasScaleTensor);

        miopenCreateActivationDescriptor(&activDesc);

        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetModeFromCmdLine();

    int SetActivationDescriptorFromCmdLineArgs();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU();
    int RunBackwardCPU();

    void runGPUBNFwdInference(Tref epsilon, float alpha, float beta);
    void runCPUBNFwdInference(Tref epsilon, int batch_sz, int channels, int height, int width);

    void runGPUActivFwdInference(float alpha, float beta);
    // void runCPUActivFwdInference();

    int VerifyBackward();
    int VerifyForward();

    ~CBAInferFusionDriver()
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(biasScaleTensor);

        miopenDestroyActivationDescriptor(activDesc);
    }

    private:
    miopenBatchNormMode_t bn_mode;
    bool estimatedMeanVar;

    unsigned char back;

    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t biasScaleTensor;
    miopenTensorDescriptor_t outputTensor;

    miopenActivationDescriptor_t activDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> scale_dev;
    std::unique_ptr<GPUMem> bias_dev;

    std::unique_ptr<GPUMem> runningMean_dev;
    std::unique_ptr<GPUMem> runningVariance_dev;
    std::unique_ptr<GPUMem> saveMean_dev;
    std::unique_ptr<GPUMem> saveInvVariance_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tgpu> din_host;
    std::vector<Tref> out_host;

    std::vector<Tgpu> scale;
    std::vector<Tgpu> scale_host;
    std::vector<Tgpu> bias;
    std::vector<Tgpu> bias_host;

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
int CBAInferFusionDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::SetActivationDescriptorFromCmdLineArgs()
{

    miopenActivationMode_t mode;
    double Alpha = inflags.GetValueDouble("alpha");
    double Beta  = inflags.GetValueDouble("beta");
    double Gamma = inflags.GetValueDouble("gamma");
    mode         = static_cast<miopenActivationMode_t>(inflags.GetValueInt("activMode"));

    return (miopenSetActivationDescriptor(activDesc, mode, Alpha, Beta, Gamma));
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::GetandSetData()
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

    SetActivationDescriptorFromCmdLineArgs();

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("batchsize", 'n', "32", "Mini-batch size (Default=32)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("alpha", 'A', "1.0", "Alpha (Default=1.0)", "float");
    inflags.AddInputFlag("beta", 'B', "0.", "Beta (Default=0.)", "float");
    inflags.AddInputFlag("gamma", 'G', "1", "Activation gamma (Default=1)", "double");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag(
        "activMode", 'm', "3", "Activation Mode (relu,..., see spec) (Default=3(relu))", "int");
    inflags.AddInputFlag("bnMode",
                         'M',
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
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> CBAInferFusionDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::SetBNParametersFromCmdLineArgs()
{

    //    	double bnAlpha = inflags.GetValueDouble("alpha");
    //    	double bnBeta = inflags.GetValueDouble("beta");

    // batch norm mode type
    if(inflags.GetValueInt("bnMode") == 0)
    {
        bn_mode = miopenBNPerActivation;
    }
    else if(inflags.GetValueInt("bnMode") == 1)
    {
        bn_mode = miopenBNSpatial;
    }
    else
    {
        printf("Incorrect Batch Normalization Mode\n");
        exit(EXIT_FAILURE);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::createSaveBuffers()
{

#if MIOPEN_BACKEND_OPENCL
    cl_int status = CL_SUCCESS;
    cl_context ctx;
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    int status   = 0;
    uint32_t ctx = 0;
#endif

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::createRunningBuffers()
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

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
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
    din_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    scale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
    bias_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
    out_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    // GPU host allocation
    in       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    din_host = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    scale    = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
    bias     = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

    // CPU allocation
    out_host   = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    scale_host = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
    bias_host  = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

    // Data initialization
    for(int i = 0; i < in_sz; i++)
    {
        in[i] = std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0)));
    }

    // Using random beta and gamma
    for(int i = 0; i < sb_sz; i++)
    {
        scale[i] = scale_host[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        bias[i] = bias_host[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= scale_dev->ToGPU(q, scale.data());
    status |= bias_dev->ToGPU(q, bias.data());
    status |= out_dev->ToGPU(q, out.data());
    status |= createRunningBuffers();
    status |= in_dev->ToGPU(q, in.data());

    if(status != CL_SUCCESS)
        printf("Fatal: Error copying data to GPU\nExiting...\n\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runGPUBNFwdInference(Tref epsilon, float alpha, float beta)
{
    miopenBatchNormalizationForwardInference(GetHandle(),
                                             bn_mode,
                                             &alpha,
                                             &beta,
                                             inputTensor,
                                             in_dev->GetMem(),
                                             outputTensor,
                                             din_dev->GetMem(),
                                             biasScaleTensor,
                                             scale_dev->GetMem(),
                                             bias_dev->GetMem(),
                                             runningMean_dev->GetMem(),
                                             runningVariance_dev->GetMem(),
                                             epsilon);
    return;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runGPUActivFwdInference(float alpha, float beta)
{

    miopenActivationForward(GetHandle(),
                            activDesc,
                            &alpha,
                            inputTensor,
                            din_dev->GetMem(),
                            &beta,
                            outputTensor,
                            out_dev->GetMem());

    return;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::RunForwardGPU()
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

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        float time0 = 0.0, time1 = 0.0;
#if 1
        miopen::BatchNormActivForwardInference(miopen::deref(GetHandle()),
                                               bn_mode,
                                               &alpha,
                                               &beta,
                                               miopen::deref(inputTensor),
                                               in_dev->GetMem(),
                                               miopen::deref(outputTensor),
                                               out_dev->GetMem(),
                                               miopen::deref(biasScaleTensor),
                                               scale_dev->GetMem(),
                                               bias_dev->GetMem(),
                                               runningMean_dev->GetMem(),
                                               runningVariance_dev->GetMem(),
                                               epsilon,
                                               activ_mode,
                                               activ_alpha,
                                               activ_beta,
                                               activ_gamma);

        miopenGetKernelTime(GetHandle(), &time0);
#else
        runGPUBNFwdInference(epsilon, alpha, beta);
        miopenGetKernelTime(GetHandle(), &time0);
        runGPUActivFwdInference(alpha, beta);
        miopenGetKernelTime(GetHandle(), &time1);
#endif

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
            float time = time0 + time1;
            lowtime    = (time < lowtime) ? time : lowtime;
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

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runCPUBNFwdInference(
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
                                             true,
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
                                       true,
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
int CBAInferFusionDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::VerifyForward()
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

    double allowedEps = std::numeric_limits<Tgpu>::epsilon() * 80;
    miopenActivationMode_t v_mode;
    double v_Alpha;
    double v_Beta;
    double v_Gamma;

    miopenGetActivationDescriptor(activDesc, &v_mode, &v_Alpha, &v_Beta, &v_Gamma);

    int match = 1;

    miopenBNActiveVerify(bn_mode,
                         batch_sz,
                         channels,
                         height,
                         width,
                         in.data(),
                         din_host.data(),
                         scale_host.data(),
                         bias_host.data(),
                         epsilon,
                         runningMean_host.data(),
                         runningVariance_host.data(),
                         v_mode,
                         static_cast<Tref>(v_Gamma),
                         static_cast<Tref>(v_Beta),
                         static_cast<Tref>(v_Alpha),
                         in.size(),
                         out.data(),
                         static_cast<Tref>(allowedEps));

    if(match)
        printf("Forward Activation Verifies on CPU and GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BN_DRIVER_HPP
