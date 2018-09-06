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
#include "miopen_ConvBatchNormActivHost.hpp"
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
#include <cassert>
#include "random.hpp"
#include "mloNeuronHost.hpp"

#define MIO_BN_DEBUG 0
#define MIO_BN_MAX_DEBUGLOOP 65536

#undef EPSILON
#define EPSILON 1e-6

#define MIO_CONV_ALGO_COUNT 4

#define ERRTOL 1e-4
#define RMSTOL_FP32 1e-4
#define RMSTOL_FP16 0.5e-3

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

#define CBA_DEBUG_VALUES 0

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
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputTensor);
        miopenCreateOperatorArgs(&fusionArgs);

        workspace_fwd_dev = nullptr;

        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
        initTiming();
        iters = 0;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetOutputTensorLengths();
    std::vector<int> GetModeFromCmdLine();

    int SetActivationDescriptorFromCmdLineArgs();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU() { return 0; };
    int RunBackwardCPU() { return 0; };

    void runGPUBatchNormActivInference();

    void runGPUBNFwdInference();
    void runCPUBNFwdInference();

    void runGPUActivFwdInference();
    void runCPUActivFwdInference();

    int VerifyBackward() { return 0; };
    int VerifyForward();

    Timer t;
    double fulltime;
    float lowtime;
    float avgtime;
    float time;
    int iters;

    void initTiming()
    {
        fulltime = 0.;
        lowtime  = 100000000.0;
        avgtime  = 0.;
        time     = 0.0;
        return;
    }

    void startTiming()
    {
        START_TIME;
        return;
    }

    void finishTiming(int i)
    {
        if(inflags.GetValueStr("time") == "1")
        {
            time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            lowtime = (time < lowtime) ? time : lowtime;
            if(iters > 1 && i > 0)
                avgtime += time;
        }

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
        return;
    }

    ~CBAInferFusionDriver()
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(biasScaleTensor);
        miopenDestroyActivationDescriptor(activDesc);
        miopenDestroyFusionPlanDescriptor(fusePlanDesc);
        miopenDestroyOperatorArgs(fusionArgs);
    }

    private:
    miopenBatchNormMode_t bn_mode;
    int bias_mode   = 0;
    int fusion_mode = 0;
    bool estimatedMeanVar;

    unsigned char back;

    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t biasScaleTensor;
    miopenTensorDescriptor_t outputTensor;

    miopenActivationDescriptor_t activDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> bn_res_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> scale_dev;
    std::unique_ptr<GPUMem> workspace_fwd_dev;
    std::unique_ptr<GPUMem> runningMean_dev;
    std::unique_ptr<GPUMem> runningVariance_dev;
    std::unique_ptr<GPUMem> saveMean_dev;
    std::unique_ptr<GPUMem> saveInvVariance_dev;
    std::unique_ptr<GPUMem> bias_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::vector<Tgpu> b;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tgpu> bn_res;
    std::vector<Tref> in_host;
    std::vector<Tref> bn_res_host;
    std::vector<Tref> out_host;
    std::vector<Tgpu> scale;
    std::vector<Tgpu> bias;
    std::vector<Tgpu> runningMean;
    std::vector<Tgpu> runningVariance;

    int createSaveBuffers();
    int createRunningBuffers();

    miopenStatus_t miopenError;
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenFusionOpDescriptor_t bNormOp;
    miopenFusionOpDescriptor_t activOp;
    miopenOperatorArgs_t fusionArgs;
};

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    fusion_mode = inflags.GetValueInt("fusion_mode");

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
    SetActivationDescriptorFromCmdLineArgs();

    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len, data_type);
    std::vector<int> out_len{};

    out_len = in_len;

    SetTensor4d(outputTensor, out_len, data_type);

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
    inflags.AddInputFlag(
        "activMode", 'm', "3", "Activation Mode (relu,..., see spec) (Default=3(relu))", "int");
    inflags.AddInputFlag("bnMode",
                         'M',
                         "0",
                         "Normalization Mode (per-activation (0) or spatial (1)) (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    inflags.AddInputFlag("fusion_mode", 'F', "0", "Fusion mode (na = 0) (Default=na)", "int");

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
    int status             = 0;
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
    int status             = 0;
    uint32_t ctx           = 0;
#endif

    size_t sb_sz = GetTensorSize(biasScaleTensor);

    // GPU allocation
    runningMean_dev     = std::make_unique<GPUMem>(ctx, sb_sz, sizeof(Tgpu));
    runningVariance_dev = std::make_unique<GPUMem>(ctx, sb_sz, sizeof(Tgpu));

    // GPU host allocation
    runningMean     = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
    runningVariance = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

    // Populate
    for(int i = 0; i < sb_sz; i++)
    {
#if(CBA_DEBUG_VALUES == 1)
        runningMean[i]     = 0.;
        runningVariance[i] = 1.;
#else
        runningMean[i]     = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        runningVariance[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
#endif
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
    int status             = 0;
    uint32_t ctx           = 0;
#endif

    size_t in_sz = GetTensorSize(inputTensor);
    size_t sb_sz = 0;

    miopenDeriveBNTensorDescriptor(biasScaleTensor, inputTensor, bn_mode);
    sb_sz = GetTensorSize(biasScaleTensor);

    size_t out_sz = 0;

    out_sz = in_sz; // This is for N+A so the output is the same as the input size

    // GPU allocation
    in_dev  = std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu));
    out_dev = std::make_unique<GPUMem>(ctx, out_sz, sizeof(Tgpu));

    scale       = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
    bias        = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
    bn_res      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    bn_res_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    bn_res_dev = std::make_unique<GPUMem>(ctx, out_sz, sizeof(Tgpu));
    scale_dev  = std::make_unique<GPUMem>(ctx, sb_sz, sizeof(Tgpu));
    bias_dev   = std::make_unique<GPUMem>(ctx, sb_sz, sizeof(Tgpu));
    // Using random beta and gamma
    for(int i = 0; i < sb_sz; i++)
    {
#if(CBA_DEBUG_VALUES == 1)
        scale[i] = 1.; // std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0),
                       // static_cast<Tgpu>(1.0))); // 1.0;
        bias[i] = 10.;
#else
        scale[i]           = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        bias[i]            = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
#endif
    }
    status |= scale_dev->ToGPU(q, scale.data());
    status |= bias_dev->ToGPU(q, bias.data());

    // GPU host allocation
    in  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    // CPU allocation
    in_host  = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    out_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    // Data initialization
    for(int i = 0; i < in_sz; i++)
    {
#if(CBA_DEBUG_VALUES == 1)
        auto rval =
            1.; // std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0))); // 1.0;
        in_host[i] = static_cast<double>(rval);
        in[i]      = rval;
#else
        auto rval  = std::fabs(RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0)));
        in_host[i] = static_cast<double>(rval);
        in[i]      = rval;
#endif
    }

    status |= in_dev->ToGPU(q, in.data());
    status |= createRunningBuffers();

    if(status != CL_SUCCESS)
        printf("Fatal: Error copying data to GPU\nExiting...\n\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runGPUBatchNormActivInference()
{

    miopenError = miopenStatusSuccess;
    double activ_alpha, activ_beta, activ_gamma;
    miopenActivationMode_t activ_mode;
    miopenGetActivationDescriptor(activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

    double epsilon = static_cast<double>(EPSILON);
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bn_mode, biasScaleTensor);

    miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);
    miopenSetOpArgsBatchNormInference(fusionArgs,
                                      bNormOp,
                                      &alpha,
                                      &beta,
                                      scale_dev->GetMem(),
                                      bias_dev->GetMem(),
                                      runningMean_dev->GetMem(),
                                      runningVariance_dev->GetMem(),
                                      epsilon);

    miopenSetOpArgsActivForward(
        fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

    miopenError = miopenCompileFusionPlan(GetHandle(), fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        std::cerr << "BatchNormActivInference plan not supported." << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int it = 0; it < iters; it++)
    {
        startTiming();
        miopenExecuteFusionPlan(GetHandle(),
                                fusePlanDesc,
                                inputTensor,
                                in_dev->GetMem(),
                                outputTensor,
                                out_dev->GetMem(),
                                fusionArgs);
        finishTiming(it);
    }
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runGPUBNFwdInference()
{
    double epsilon = static_cast<double>(EPSILON);
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    miopenBatchNormalizationForwardInference(GetHandle(),
                                             bn_mode,
                                             &alpha,
                                             &beta,
                                             outputTensor,
                                             in_dev->GetMem(),
                                             outputTensor,
                                             bn_res_dev->GetMem(),
                                             biasScaleTensor,
                                             scale_dev->GetMem(),
                                             bias_dev->GetMem(),
                                             runningMean_dev->GetMem(),
                                             runningVariance_dev->GetMem(),
                                             epsilon);

    // bn_res_dev->FromGPU(GetStream(), bn_res.data());

    return;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runCPUActivFwdInference()
{
    double activ_alpha, activ_beta, activ_gamma;
    miopenActivationMode_t activ_mode;
    miopenGetActivationDescriptor(activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
    miopenActivationFwdHost<Tgpu, Tref>(activ_mode,
                                        activ_gamma,
                                        activ_beta,
                                        activ_alpha,
                                        out.size(),
                                        bn_res_host.data(),
                                        out_host.data());

    return;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runGPUActivFwdInference()
{
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    miopenActivationForward(GetHandle(),
                            activDesc,
                            &alpha,
                            outputTensor,
                            bn_res_dev->GetMem(), // DLOWELL this might be a bug if not using BN
                            &beta,
                            outputTensor,
                            out_dev->GetMem());

    return;
}
template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::RunForwardGPU()
{

    iters = inflags.GetValueInt("iter");

    initTiming();

    runGPUBatchNormActivInference();

    if(WALL_CLOCK)
    {
        printf("Wall-clock Time Elapsed: %f ms, for %d iterations.\n",
               (iters == 1) ? t.gettime_ms() : (fulltime / float(iters - 1)),
               (iters > 1) ? iters - 1 : 1);
    }

    if(inflags.GetValueStr("time") == "1")
    {
        printf("GPU Fused Kernel Min Time Elapsed: %f ms\n", lowtime);
        if(iters > 1)
            printf("GPU Fused Kernel Avg Time Elapsed: %f ms, for %d "
                   "iterations.\n",
                   avgtime / (iters - 1),
                   iters - 1);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runCPUBNFwdInference()
{
    double epsilon = static_cast<double>(EPSILON);

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        std::cout << "Running CPU per activation BN." << std::endl;
        miopenBNPerActivFwdInferHost(inputTensor, // DLOWELL use output for splice test
                                     in_host.data(),
                                     bn_res_host.data(),
                                     scale.data(),
                                     bias.data(),
                                     epsilon,
                                     runningMean.data(),
                                     runningVariance.data());
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        std::cout << "Running CPU spatial BN." << std::endl;
        miopenBNSpatialFwdInferHost(inputTensor, // DLOWELL use output for splice test
                                    in_host.data(),
                                    bn_res_host.data(),
                                    scale.data(),
                                    bias.data(),
                                    epsilon,
                                    runningMean.data(),
                                    runningVariance.data());
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

    runCPUBNFwdInference();
    runCPUActivFwdInference();

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    double allowedEps = std::numeric_limits<Tgpu>::epsilon() * 80;

    int match = 1;

    match &= miopenInferVerify(out.size(), out_host.data(), out.data(), allowedEps);

    if(match)
        printf("Verifies on CPU and GPU\n");

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BN_DRIVER_HPP
