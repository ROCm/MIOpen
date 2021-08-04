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
#ifndef GUARD_MIOPEN_BN_DRIVER_HPP
#define GUARD_MIOPEN_BN_DRIVER_HPP

#include "../test/verify.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen_BatchNormHost.hpp"
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

#define ERRTOL_FP32 1e-4
#define ERRTOL_FP16 0.5e-3
#define RMSTOL_FP32 1e-4
#define RMSTOL_FP16 0.5e-3

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

//#define BN_RUNFOR_PROFILER

template <typename Tgpu, typename Tref, typename Tmix = Tgpu>
class BatchNormDriver : public Driver
{
    public:
    BatchNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasScaleTensor);
        miopenCreateTensorDescriptor(&dxOutputTensor);
        miopenCreateTensorDescriptor(&dyInputTensor);

        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetModeFromCmdLine();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    void runGPUFwdInference(Tref epsilon, float alpha, float beta);
    void runGPUFwdTrain(Tref epsilon, Tref eAF, float alpha, float beta);
    void runGPUBwd(Tref epsilon, float alpha, float beta);

    void runCPUFwdInference(
        Tref epsilon, int batch_sz, int channels, int height, int width, int depth = 0);
    void runCPUFwdTrain(
        Tref epsilon, Tref eAF, int batch_sz, int channels, int height, int width, int depth = 0);

    int VerifyBackward() override;
    int VerifyForward() override;

    ~BatchNormDriver() override
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

    int forw;
    int back;

    InputFlags inflags;
    bool isDepthSpecified = false;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t biasScaleTensor;
    miopenTensorDescriptor_t outputTensor;

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

    std::vector<Tmix> scale;
    std::vector<Tref> scale_host;
    std::vector<Tmix> bias;
    std::vector<Tref> bias_host;

    std::vector<Tmix> dscale;
    std::vector<Tref> dscale_host;
    std::vector<Tmix> dbias;
    std::vector<Tref> dbias_host;

    std::vector<Tmix> runningMean;
    std::vector<Tmix> runningVariance;
    std::vector<Tref> runningMean_host;
    std::vector<Tref> runningVariance_host;

    std::vector<Tmix> saveMean;
    std::vector<Tmix> saveInvVariance;

    std::vector<Tref> saveMean_host;
    std::vector<Tref> saveInvVariance_host;

    int createSaveBuffers();
    int createRunningBuffers();
    Tref maxval;
};

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::GetandSetData()
{

    SetBNParametersFromCmdLineArgs();

    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    std::vector<int> sb_len;
    if(bn_mode == miopenBNPerActivation)
    {
        // 1xCxHxW | in_len.size = 4
        sb_len.push_back(1);
        sb_len.push_back(in_len[1]);
        sb_len.push_back(in_len[2]);
        sb_len.push_back(in_len[3]);

        // 1xCxDxHxW | in_len.size = 5
        if(in_len.size() == 5)
        {
            sb_len.push_back(in_len[4]);
        }
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        sb_len.push_back(1);
        sb_len.push_back(in_len[1]);
        sb_len.push_back(1);
        sb_len.push_back(1);

        // 1xCx1x1x1
        if(in_len.size() == 5)
        {
            sb_len.push_back(1);
        }
    }

    SetTensorNd(inputTensor, in_len, data_type);
    SetTensorNd(biasScaleTensor, sb_len, ((sizeof(Tmix) == 4) ? miopenFloat : miopenHalf));
    SetTensorNd(outputTensor, in_len, data_type);

    // backwards
    SetTensorNd(dyInputTensor, in_len, data_type);
    SetTensorNd(dxOutputTensor, in_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw",
        'F',
        "0",
        "Run Forward Train (off: 0, train: 1, inference: 2) Batch Normalization (Default=1)",
        "int");
    inflags.AddInputFlag("back",
                         'b',
                         "0",
                         "Backwards Propagation (off: 0, on: 1) Batch Normalization (Default=0)",
                         "int");
    inflags.AddInputFlag("batchsize", 'n', "32", "Mini-batch size (Default=32)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
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

template <typename Tgpu, typename Tref, typename Tmix>
std::vector<int> BatchNormDriver<Tgpu, Tref, Tmix>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");
    int in_d = inflags.GetValueInt("in_d");

    if(in_d)
    {
        isDepthSpecified = true;

        // NxCxDxHxW -> NxCx(D*H)xW
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else
    {
        isDepthSpecified = false;
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::SetBNParametersFromCmdLineArgs()
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
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
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
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
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
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    forw = inflags.GetValueInt("forw");
    if(forw > 2)
    {
        printf("Incorrect Batch Normalization forward mode\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    back = inflags.GetValueInt("back");
    if(back > 1)
    {
        printf("Incorrect Batch Normalization backwards propagation mode\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    if(back && forw)
    {
        printf(
            "Warning: Deactivate forward to run backward on Batch Norm.\nRunning forward only.\n");
        back = 0;
    }
    else if(!back && !forw)
    {
        back = 0;
        forw = 1;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::createSaveBuffers()
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
        saveMean_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));
        saveInvVariance_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));

        if(back == 1)
        {
            // GPU host allocation
            saveMean        = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));
            saveInvVariance = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));

            // CPU allocation
            saveMean_host        = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
            saveInvVariance_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));

            // Populate
            for(int i = 0; i < sb_sz; i++)
            {
                saveMean_host[i] = saveMean[i] =
                    RAN_GEN<Tref>(static_cast<Tref>(0.0), static_cast<Tref>(1.0));
                saveInvVariance_host[i] = saveInvVariance[i] =
                    RAN_GEN<Tref>(static_cast<Tref>(0.0), static_cast<Tref>(1.0));
            }
        }
        else
        {
            // GPU host allocation
            saveMean        = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));
            saveInvVariance = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));

            // CPU allocation
            saveMean_host        = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
            saveInvVariance_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
        }
        // GPU data transfer
        status |= saveMean_dev->ToGPU(q, saveMean.data());
        status |= saveInvVariance_dev->ToGPU(q, saveInvVariance.data());
    }
    else
    {
        saveMean_dev        = nullptr;
        saveInvVariance_dev = nullptr;
    }

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::createRunningBuffers()
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
        runningMean_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));
        runningVariance_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));

        if(forw == 2)
        {
            // GPU host allocation
            runningMean     = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));
            runningVariance = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));

            // CPU allocation
            runningMean_host     = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
            runningVariance_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));

            // Populate
            for(int i = 0; i < sb_sz; i++)
            {
                runningMean[i]      = RAN_GEN<Tmix>(static_cast<Tmix>(0.0), static_cast<Tmix>(1.0));
                runningMean_host[i] = static_cast<Tref>(runningMean[i]);
                runningVariance[i]  = RAN_GEN<Tmix>(static_cast<Tmix>(0.0), static_cast<Tmix>(1.0));
                runningVariance_host[i] = static_cast<Tref>(runningVariance[i]);
            }
        }
        else
        {
            // GPU host allocation
            runningMean     = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));
            runningVariance = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));

            // CPU allocation
            runningMean_host     = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
            runningVariance_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
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

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::AllocateBuffersAndCopy()
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

    if(forw)
    {

        size_t out_sz = GetTensorSize(outputTensor);

        // GPU allocation
        in_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        scale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));
        bias_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));
        out_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        // GPU host allocation
        in    = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        out   = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        scale = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));
        bias  = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));

        // CPU allocation
        out_host   = std::vector<Tref>(out_sz, static_cast<Tref>(0));
        scale_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
        bias_host  = std::vector<Tref>(sb_sz, static_cast<Tref>(0));

        // Data initialization
        for(int i = 0; i < in_sz; i++)
        {
            in[i] = std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0)));
        }
        status |= in_dev->ToGPU(q, in.data());

        // Using random beta and gamma
        for(int i = 0; i < sb_sz; i++)
        {
            scale[i]      = RAN_GEN<Tmix>(static_cast<Tmix>(0.0), static_cast<Tmix>(1.0));
            scale_host[i] = static_cast<Tref>(scale[i]);
            bias[i]       = RAN_GEN<Tmix>(static_cast<Tmix>(0.0), static_cast<Tmix>(1.0));
            bias_host[i]  = static_cast<Tref>(bias[i]);
        }
        status |= scale_dev->ToGPU(q, scale.data());
        status |= bias_dev->ToGPU(q, bias.data());
        status |= out_dev->ToGPU(q, out.data());

        if(forw == 1)
        { // training
            status |= createRunningBuffers();
            status |= createSaveBuffers();
        }
        else if(forw == 2)
        { // inference
            status |= createRunningBuffers();
        }
    } // end forward

    if(back == 1)
    {

        size_t out_sz = GetTensorSize(dxOutputTensor);

        // GPU allocation
        in_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        dyin_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        dxout_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        dscale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));
        dbias_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));
        scale_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tmix)));

        // GPU host allocation
        in     = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        dyin   = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        dxout  = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        dscale = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));
        dbias  = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));
        scale  = std::vector<Tmix>(sb_sz, static_cast<Tmix>(0));

        // CPU allocation
        dxout_host  = std::vector<Tref>(out_sz, static_cast<Tref>(0));
        dscale_host = std::vector<Tref>(sb_sz, static_cast<Tref>(0));
        dbias_host  = std::vector<Tref>(sb_sz, static_cast<Tref>(0));

        // Populate
        for(int i = 0; i < sb_sz; i++)
        {
            scale[i] = RAN_GEN<Tmix>(static_cast<Tmix>(0.0), static_cast<Tmix>(1.0));
        }
        status |= scale_dev->ToGPU(q, scale.data());
        status |= dscale_dev->ToGPU(q, dscale.data());
        status |= dbias_dev->ToGPU(q, dbias.data());

        for(int i = 0; i < in_sz; i++)
        {
            dyin[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            in[i]   = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        status |= dyin_dev->ToGPU(q, dyin.data());
        status |= in_dev->ToGPU(q, in.data());
        status |= dxout_dev->ToGPU(q, dxout.data());

        status |= createSaveBuffers();
    }

    if(status != CL_SUCCESS)
        printf("Fatal: Error copying data to GPU\nExiting...\n\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
void BatchNormDriver<Tgpu, Tref, Tmix>::runGPUFwdInference(Tref epsilon, float alpha, float beta)
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

template <typename Tgpu, typename Tref, typename Tmix>
void BatchNormDriver<Tgpu, Tref, Tmix>::runGPUFwdTrain(Tref epsilon,
                                                       Tref eAF,
                                                       float alpha,
                                                       float beta)
{
    if(saveMeanVar && keepRunningMeanVar)
    {
        miopenBatchNormalizationForwardTraining(GetHandle(),
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
                                                eAF,
                                                runningMean_dev->GetMem(),
                                                runningVariance_dev->GetMem(),
                                                epsilon,
                                                saveMean_dev->GetMem(),
                                                saveInvVariance_dev->GetMem());
    }
    else if(saveMeanVar)
    {
        miopenBatchNormalizationForwardTraining(GetHandle(),
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
                                                eAF,
                                                nullptr,
                                                nullptr,
                                                epsilon,
                                                saveMean_dev->GetMem(),
                                                saveInvVariance_dev->GetMem());
    }
    else if(keepRunningMeanVar)
    {
        miopenBatchNormalizationForwardTraining(GetHandle(),
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
                                                eAF,
                                                runningMean_dev->GetMem(),
                                                runningVariance_dev->GetMem(),
                                                epsilon,
                                                nullptr,
                                                nullptr);
    }
    else
    {
        miopenBatchNormalizationForwardTraining(GetHandle(),
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
                                                eAF,
                                                nullptr,
                                                nullptr,
                                                epsilon,
                                                nullptr,
                                                nullptr);
    }

#ifdef BN_RUNFOR_PROFILER
    miopenBatchNormalizationForwardTraining(GetHandle(),
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
                                            eAF,
                                            nullptr,
                                            nullptr,
                                            epsilon,
                                            nullptr,
                                            nullptr);
#endif
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::RunForwardGPU()
{

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    Tref epsilon = static_cast<Tref>(EPSILON);
    Tref eAF     = static_cast<Tref>(1.0);

    Timer t;
    double fulltime = 0.;
    auto iters      = inflags.GetValueInt("iter");
    float lowtime   = 100000000.0;
    float avgtime   = 0.;

    for(int i = 0; i < iters; i++)
    {

        START_TIME

        // if run fwd train
        if(forw == 1)
        { // training only
            eAF = static_cast<Tref>(1.0) / (static_cast<Tref>(i) + static_cast<Tref>(1.0));
            runGPUFwdTrain(epsilon, eAF, alpha, beta);
        }
        else if(forw == 2)
        { // inference only
            // printf("Running for inference.\n");
            runGPUFwdInference(epsilon, alpha, beta);
        }
        else if(forw == 0)
        {
            return miopenStatusSuccess;
        }
        else
        {
            printf("Batch normalization mode forward GPU selection out of range, skipping.\n");
            return miopenStatusNotImplemented;
        }

        miopen::deref(GetHandle()).Finish();
        STOP_TIME
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
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
        size_t M                         = in_n * in_c * in_h * in_w;
        size_t dataSz = (M + 2 * in_c) * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
        float rdCnt   = -1.0;
        float wrCnt   = 1.0;
        if(forw == 1)
        {
            rdCnt = 2;
        }
        else if(forw == 2)
        {
            rdCnt = 1;
        }
        // layer, flopCnt, reads, writes, GFLOPS, GB/s, timeMs
        printf("stats: bnormf, 0, %zu, %zu, 0, %f, %f\n",
               dataSz,
               dataSz,
               (rdCnt * dataSz + wrCnt * dataSz) / lowtime / 1e6,
               lowtime);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
void BatchNormDriver<Tgpu, Tref, Tmix>::runCPUFwdInference(
    Tref epsilon, int batch_sz, int channels, int height, int width, int depth)
{

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        miopenBNFwdInferPerActivationRunHost<Tgpu, Tref>(/* alpha, beta, */ batch_sz,
                                                         channels,
                                                         (isDepthSpecified ? depth : 1),
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
        miopenBNFwdInferSpatialRunHost<Tgpu, Tref>(/* alpha, beta, */ batch_sz,
                                                   channels,
                                                   (isDepthSpecified ? depth : 1),
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
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }
    return;
}

template <typename Tgpu, typename Tref, typename Tmix>
void BatchNormDriver<Tgpu, Tref, Tmix>::runCPUFwdTrain(
    Tref epsilon, Tref eAF, int batch_sz, int channels, int height, int width, int depth)
{

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        miopenBNFwdTrainPerActivationRunHost<Tgpu, Tref>(/* alpha, beta, */ batch_sz,
                                                         channels,
                                                         (isDepthSpecified ? depth : 1),
                                                         height,
                                                         width,
                                                         in.data(),
                                                         out_host.data(),
                                                         scale_host.data(),
                                                         bias_host.data(),
                                                         epsilon,
                                                         saveMeanVar,
                                                         keepRunningMeanVar,
                                                         saveMean_host.data(),
                                                         saveInvVariance_host.data(),
                                                         runningMean_host.data(),
                                                         runningVariance_host.data(),
                                                         eAF);
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        miopenBNFwdTrainSpatialRunHost<Tgpu, Tref>(/* alpha, beta, */ batch_sz,
                                                   channels,
                                                   (isDepthSpecified ? depth : 1),
                                                   height,
                                                   width,
                                                   in.data(),
                                                   out_host.data(),
                                                   scale_host.data(),
                                                   bias_host.data(),
                                                   epsilon,
                                                   saveMeanVar,
                                                   keepRunningMeanVar,
                                                   saveMean_host.data(),
                                                   saveInvVariance_host.data(),
                                                   runningMean_host.data(),
                                                   runningVariance_host.data(),
                                                   eAF);
    }
    else
    {
        printf("Something went wrong.\nBad batch normalization mode in host kernel "
               "selection.\nExiting...\n\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::RunForwardCPU()
{
    int nIn = 0, cIn = 0, dIn = 0, hIn = 0, wIn = 0;

    if(isDepthSpecified)
        miopenGet5dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &dIn, &hIn, &wIn);
    else
        miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);

    int batch_sz = nIn;
    int channels = cIn;
    int height   = hIn;
    int width    = wIn;
    int depth    = dIn;

    //	T alpha = 0., beta  = 0.;
    Tref epsilon = static_cast<Tref>(EPSILON);
    Tref eAF     = static_cast<Tref>(1.0);

    if(forw == 1)
    { // training only
        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            eAF = static_cast<Tref>(1.0) / (static_cast<Tref>(i) + static_cast<Tref>(1.0));
            runCPUFwdTrain(
                epsilon, eAF, /* alpha, beta,*/ batch_sz, channels, height, width, depth);
        }
    }
    else if(forw == 2)
    { // inference only
        runCPUFwdInference(epsilon, /* alpha, beta,*/ batch_sz, channels, height, width, depth);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::RunBackwardGPU()
{

    if(!back)
        return miopenStatusSuccess;

    float alphaDataDiff = static_cast<float>(1), betaDataDiff = static_cast<float>(0);
    float alphaParamDiff = static_cast<float>(1), betaParamDiff = static_cast<float>(0);
    Tref epsilon = static_cast<Tref>(EPSILON);

    Timer t;
    double fulltime = 0.;
    auto iters      = inflags.GetValueInt("iter");
    float lowtime   = 100000000.0;
    float avgtime   = 0.;

    for(int i = 0; i < iters; i++)
    {
        START_TIME

        if(saveMeanVar)
        {
            miopenBatchNormalizationBackward(GetHandle(),
                                             bn_mode,
                                             &alphaDataDiff,
                                             &betaDataDiff,
                                             &alphaParamDiff,
                                             &betaParamDiff,
                                             inputTensor,
                                             in_dev->GetMem(),
                                             dyInputTensor,
                                             dyin_dev->GetMem(),
                                             dxOutputTensor,
                                             dxout_dev->GetMem(),
                                             biasScaleTensor,
                                             scale_dev->GetMem(),
                                             dscale_dev->GetMem(),
                                             dbias_dev->GetMem(),
                                             epsilon,
                                             saveMean_dev->GetMem(),
                                             saveInvVariance_dev->GetMem());
        }
        else
        {
            miopenBatchNormalizationBackward(GetHandle(),
                                             bn_mode,
                                             &alphaDataDiff,
                                             &betaDataDiff,
                                             &alphaParamDiff,
                                             &betaParamDiff,
                                             inputTensor,
                                             in_dev->GetMem(),
                                             dyInputTensor,
                                             dyin_dev->GetMem(),
                                             dxOutputTensor,
                                             dxout_dev->GetMem(),
                                             biasScaleTensor,
                                             scale_dev->GetMem(),
                                             dscale_dev->GetMem(),
                                             dbias_dev->GetMem(),
                                             epsilon,
                                             nullptr,
                                             nullptr);
        }

        miopen::deref(GetHandle()).Finish();
        STOP_TIME
        if(WALL_CLOCK)
        {
            if(iters > 1 && i > 0)
                fulltime += t.gettime_ms();
            else if(iters == 1)
                fulltime = t.gettime_ms();
        }

        if(inflags.GetValueStr("time") == "1")
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            lowtime = (time < lowtime) ? time : lowtime;
            if(iters > 1 && i > 0)
                avgtime += time;

            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) =
                miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
            size_t M = in_n * in_c * in_h * in_w;
            size_t dataSz =
                (M + 2 * in_c) * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
            float rdCnt = 2.0;
            float wrCnt = 1.0;
            // layer, flopCnt, reads, writes, GFLOPS, GB/s, timeMs
            printf("stats: bnormb, 0, %zu, %zu, 0, %f, %f\n",
                   dataSz,
                   dataSz,
                   (rdCnt * dataSz + wrCnt * dataSz) / lowtime / 1e6,
                   lowtime);
        }
    }

    if(WALL_CLOCK)
    {
        printf("Wall-clock Time Backward GPU Batch Norm Elapsed: %f ms\n",
               (iters == 1) ? t.gettime_ms() : (fulltime / float(iters - 1)));
    }
    if(inflags.GetValueStr("time") == "1")
    {
        printf("GPU Kernel Min Time Backwards Batch Normalization Elapsed: %f ms\n", lowtime);
        if(iters > 1)
            printf("GPU Kernel Avg Time Backward Batch Normalization Elapsed: %f ms\n",
                   avgtime / (iters - 1));
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::VerifyForward()
{

    // jump out since we are forcing forward off when doing backwards.
    if(!forw)
        return miopenStatusSuccess;

    const Tref maxrms = static_cast<Tref>((sizeof(Tgpu) == 4) ? RMSTOL_FP32 : RMSTOL_FP16);

#if(MIO_BN_DEBUG == 1)
    const Tref tolerance = static_cast<Tref>((sizeof(Tgpu) == 4) ? ERRTOL_FP32 : ERRTOL_FP16);
    Tref diff            = static_cast<Tref>(0.);
#endif

    bool anError = false;

    RunForwardCPU();

    if(forw == 1)
    {

        if(keepRunningMeanVar)
        { // copy back for verification
            runningMean_dev->FromGPU(GetStream(), runningMean.data());
            runningVariance_dev->FromGPU(GetStream(), runningVariance.data());

            auto errorRunMean = miopen::rms_range(runningMean_host, runningMean);
            if(errorRunMean > maxrms || std::isnan(errorRunMean))
            {
                std::cout << "Forward train batch norm verification failed on running mean: "
                          << errorRunMean << "\n";
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < runningMean.size() && i < runningMean_host.size() &&
                               i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(Tmix(fabs(runningMean[i]) - fabs(runningMean_host[i])));
                    if(diff > tolerance)
                    {
                        std::cout << "rm[" << i << "]: " << runningMean[i];
                        std::cout << ", rm_host[" << i << "]: " << runningMean_host[i];
                        std::cout << ", diff[" << i
                                  << "]: " << Tmix(fabs(runningMean[i]) - fabs(runningMean_host[i]))
                                  << std::endl;
                    }
                }
#endif
            }
            else
            {
                std::cout << "Forward train batch norm verification passed on running mean.\n";
            }

            auto errorRunVar = miopen::rms_range(runningVariance_host, runningVariance);
            if(errorRunVar > maxrms || std::isnan(errorRunVar))
            {
                std::cout << "Forward train batch norm verification failed on running variance: "
                          << errorRunVar << "\n";
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < runningVariance.size() && i < runningVariance_host.size() &&
                               i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(Tmix(fabs(runningVariance[i]) - fabs(runningVariance_host[i])));
                    if(diff > tolerance)
                    {
                        std::cout << "rv[" << i << "]: " << runningVariance[i];
                        std::cout << ", rv_host[" << i << "]: " << runningVariance_host[i];
                        std::cout << ", diff[" << i << "]: "
                                  << Tmix(fabs(runningVariance[i]) - fabs(runningVariance_host[i]))
                                  << std::endl;
                    }
                }
#endif
            }
            else
            {
                std::cout << "Forward train batch norm verification passed on running variance\n";
            }
        } // end if(keepRunningMeanVar)

        if(saveMeanVar)
        { // copy back for verification
            saveMean_dev->FromGPU(GetStream(), saveMean.data());
            saveInvVariance_dev->FromGPU(GetStream(), saveInvVariance.data());
            maxval             = static_cast<Tref>(0.0);
            auto errorSaveMean = miopen::rms_range(saveMean_host, saveMean);
            if(errorSaveMean > maxrms || std::isnan(errorSaveMean))
            {
                std::cout << "Forward train batch norm verification failed on saved mean: "
                          << errorSaveMean << "\n";
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0;
                    i < saveMean.size() && i < saveMean_host.size() && i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff   = fabs(Tmix(fabs(saveMean[i]) - fabs(saveMean_host[i])));
                    maxval = maxval < diff ? diff : maxval;
                    if(diff > tolerance)
                    {
                        std::cout << "sm[" << i << "]: " << saveMean[i];
                        std::cout << ", sm_host[" << i << "]: " << saveMean_host[i];
                        std::cout << ", diff[" << i
                                  << "]: " << Tmix(fabs(saveMean[i]) - fabs(saveMean_host[i]))
                                  << std::endl;
                    }
                }
#endif
                std::cout << "max difference in saved mean: " << maxval << std::endl;
            }
            else
            {
                std::cout << "Forward train batch norm verification passed on saved mean\n";
            }

            auto errorSaveVar = miopen::rms_range(saveInvVariance_host, saveInvVariance);
            if(errorSaveVar > maxrms || std::isnan(errorSaveVar))
            {
                std::cout
                    << "Forward train batch norm verification failed on saved inverse variance: "
                    << errorSaveVar << "\n";
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < saveInvVariance.size() && i < saveInvVariance_host.size() &&
                               i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(Tmix(fabs(saveInvVariance[i]) - fabs(saveInvVariance_host[i])));
                    if(diff > tolerance)
                    {
                        std::cout << "sv[" << i << "]: " << saveInvVariance[i];
                        std::cout << ", sv_host[" << i << "]: " << saveInvVariance_host[i];
                        std::cout << ", diff[" << i << "]: "
                                  << Tmix(fabs(saveInvVariance[i]) - fabs(saveInvVariance_host[i]))
                                  << std::endl;
                    }
                }
#endif
            }
            else
            {
                std::cout
                    << "Forward train batch norm verification passed on saved inverse variance.\n";
            }
        } // end if(saveMeanVar)
    }

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

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::RunBackwardCPU()
{

    if(!back)
        return miopenStatusSuccess;

    int nIn = 0, cIn = 0, dIn = 0, hIn = 0, wIn = 0;
    if(isDepthSpecified)
        miopenGet5dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &dIn, &hIn, &wIn);
    else
        miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);

    int batch_sz = nIn;
    int channels = cIn;
    int height   = hIn;
    int width    = wIn;
    int depth    = dIn;

    //	T alphaDiff = 1, betaDiff = 0;
    //	T alphaParam = 1, betaParam = 0;
    Tref epsilon = static_cast<Tref>(EPSILON);

    if(bn_mode == miopenBNPerActivation)
    {                                                     // 1xCxHxW
        miopenBNBwdPerActivationRunHost<Tgpu, Tref, Tmix>(/* alphaDiff, betaDiff, alphaParam,
                                                             betaParam, */
                                                          batch_sz,
                                                          channels,
                                                          (isDepthSpecified ? depth : 1),
                                                          height,
                                                          width,
                                                          in.data(),
                                                          dyin.data(),
                                                          dxout_host.data(),
                                                          scale.data(),
                                                          dscale_host.data(),
                                                          dbias_host.data(),
                                                          epsilon,
                                                          saveMeanVar,
                                                          saveMean_host.data(),
                                                          saveInvVariance_host.data());
    }
    else if(bn_mode == miopenBNSpatial)
    {                                               // 1xCx1x1
        miopenBNBwdSpatialRunHost<Tgpu, Tref, Tmix>(/* alphaDiff, betaDiff, alphaParam, betaParam,
                                                     */
                                                    batch_sz,
                                                    channels,
                                                    (isDepthSpecified ? depth : 1),
                                                    height,
                                                    width,
                                                    in.data(),
                                                    dyin.data(),
                                                    dxout_host.data(),
                                                    scale.data(),
                                                    dscale_host.data(),
                                                    dbias_host.data(),
                                                    epsilon,
                                                    saveMeanVar,
                                                    saveMean_host.data(),
                                                    saveInvVariance_host.data());
    }
    else
    {
        printf("Something went wrong.\nBad batch normalization mode in host kernel "
               "selection.\nExiting...\n\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::VerifyBackward()
{

    if(!back)
        return miopenStatusSuccess;

    const Tref maxrms = static_cast<Tref>(((sizeof(Tgpu) == 4) ? RMSTOL_FP32 : RMSTOL_FP16) * 1000);
    bool anError      = false;

    RunBackwardCPU();

    dxout_dev->FromGPU(GetStream(), dxout.data());
    dscale_dev->FromGPU(GetStream(), dscale.data());
    dbias_dev->FromGPU(GetStream(), dbias.data());
#if(MIO_BN_DEBUG == 1)
    const Tref tolerance =
        static_cast<Tref>(1000 * (sizeof(Tgpu) == 4) ? ERRTOL_FP32 : ERRTOL_FP16);
    Tref diff = static_cast<Tref>(0.0);
#endif
    maxval          = static_cast<Tref>(0.0);
    auto errordxout = miopen::rms_range(dxout_host, dxout);
    if(errordxout > maxrms || std::isnan(errordxout))
    {
        std::cout << "Backwards prop batch norm verification failed on dx: " << errordxout << "\n";
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < dxout.size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff   = fabs(Tgpu(fabs(dxout[i]) - fabs(dxout_host[i])));
            maxval = maxval < diff ? diff : maxval;
            if(diff > tolerance)
            {
                std::cout << "dxout[" << i << "]: " << dxout[i];
                std::cout << "\tdxout_host[" << i << "]: " << dxout_host[i];
                std::cout << "\tdiff[" << i << "]: " << Tgpu(fabs(dxout[i]) - fabs(dxout_host[i]));
                std::cout << "\tratioH: "
                          << fabs(fabs(dxout[i]) - fabs(dxout_host[i])) / fabs(dxout_host[i])
                          << std::endl;
            }
        }
#endif
        std::cout << "max difference in dxout: " << maxval << std::endl;
    }
    else
    {
        std::cout << "Backwards prop batch norm verification passed on dx.\n";
    }

    maxval           = static_cast<Tref>(0.0);
    auto errordscale = miopen::rms_range(dscale_host, dscale);
    if(errordscale > maxrms || std::isnan(errordscale))
    {
        std::cout << "Backwards prop batch norm verification failed on dscale: " << errordscale
                  << "\n";
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < dscale.size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff   = fabs(Tmix(fabs(dscale[i]) - fabs(dscale_host[i])));
            maxval = maxval < diff ? diff : maxval;
            if(diff > tolerance)
            {
                std::cout << "dscale[" << i << "]: " << dscale[i];
                std::cout << "\tdscale_host[" << i << "]: " << dscale_host[i];
                std::cout << "\tdiff[" << i
                          << "]: " << Tmix(fabs(dscale[i]) - fabs(dscale_host[i]));
                std::cout << "\tratioH: "
                          << fabs(fabs(dscale[i]) - fabs(dscale_host[i])) / fabs(dscale_host[i])
                          << std::endl;
            }
        }
#endif
        std::cout << "max difference in dscale: " << maxval << std::endl;
    }
    else
    {
        std::cout << "Backwards prop batch norm verification passed on dscale.\n";
    }

    auto errordbias = miopen::rms_range(dbias_host, dbias);
    if(errordbias > maxrms || std::isnan(errordbias))
    {
        std::cout << "Backwards prop batch norm verification failed on dbias: " << errordbias
                  << "\n";
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < dbias.size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff = fabs(Tmix(fabs(dbias[i]) - fabs(dbias_host[i])));
            if(diff > tolerance)
            {
                std::cout << "dbias[" << i << "]: " << dbias[i];
                std::cout << "\tdbias_host[" << i << "]: " << dbias_host[i];
                std::cout << "\tdiff[" << i << "]: " << Tmix(fabs(dbias[i]) - fabs(dbias_host[i]));
                std::cout << "\tratioH: "
                          << fabs(fabs(dbias[i]) - fabs(dbias_host[i])) / fabs(dbias_host[i])
                          << std::endl;
            }
        }
#endif
    }
    else
    {
        std::cout << "Backwards prop batch norm verification passed on dbias.\n";
    }

    if(!anError)
        std::cout << "Backwards Prop Batch Norm Verifies on CPU and GPU." << std::endl;

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BN_DRIVER_HPP
