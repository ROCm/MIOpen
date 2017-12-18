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
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>

#define MIO_BN_DEBUG 0
#define MIO_BN_MAX_DEBUGLOOP 65536

#define EPSILON 1e-4

#define ERRTOL 1e-5
#define RMSTOL 1e-5

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

//#define BN_RUNFOR_PROFILER

template <typename T>
class BatchNormDriver : public Driver
{
    public:
    BatchNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&biasScaleTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&dxOutputTensor);
        miopenCreateTensorDescriptor(&dyInputTensor);
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

    void runGPUFwdInference(double epsilon, T alpha, T beta);
    void runGPUFwdTrain(double epsilon, double eAF, T alpha, T beta);
    void runGPUBwd(double epsilon, T alpha, T beta);

    void runCPUFwdInference(double epsilon, int batch_sz, int channels, int height, int width);
    void
    runCPUFwdTrain(double epsilon, double eAF, int batch_sz, int channels, int height, int width);

    int VerifyBackward();
    int VerifyForward();

    ~BatchNormDriver()
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

    unsigned char forw;
    unsigned char back;

    InputFlags inflags;

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

    std::vector<T> dyin; // output of forward
    std::vector<T> in;
    std::vector<T> out;
    std::vector<double> out_host;
    std::vector<T> dxout;
    std::vector<double> dxout_host;

    std::vector<T> scale;
    std::vector<T> scale_host;
    std::vector<T> bias;
    std::vector<T> bias_host;

    std::vector<T> dscale;
    std::vector<double> dscale_host;
    std::vector<T> dbias;
    std::vector<double> dbias_host;

    std::vector<T> runningMean;
    std::vector<T> runningVariance;
    std::vector<double> runningMean_host;
    std::vector<double> runningVariance_host;

    std::vector<T> saveMean;
    std::vector<T> saveInvVariance;

    std::vector<double> saveMean_host;
    std::vector<double> saveInvVariance_host;

    int createSaveBuffers();
    int createRunningBuffers();
    double maxval;
};

template <typename T>
int BatchNormDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename T>
int BatchNormDriver<T>::GetandSetData()
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

    SetTensor4d(inputTensor, in_len);
    SetTensor4d(biasScaleTensor, sb_len);
    SetTensor4d(outputTensor, in_len);

    // backwards
    SetTensor4d(dyInputTensor, in_len);
    SetTensor4d(dxOutputTensor, in_len);

    return miopenStatusSuccess;
}

template <typename T>
int BatchNormDriver<T>::AddCmdLineArgs()
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
    inflags.AddInputFlag("alpha", 'A', "1.0", "Alpha (Default=1.0)", "double");
    inflags.AddInputFlag("beta", 'B', "0.", "Beta (Default=0.)", "double");
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

template <typename T>
std::vector<int> BatchNormDriver<T>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");
    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename T>
int BatchNormDriver<T>::SetBNParametersFromCmdLineArgs()
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

    forw = inflags.GetValueInt("forw");
    if(forw > 2)
    {
        printf("Incorrect Batch Normalization forward mode\n");
        exit(EXIT_FAILURE);
    }

    back = inflags.GetValueInt("back");
    if(back > 1)
    {
        printf("Incorrect Batch Normalization backwards propagation mode\n");
        exit(EXIT_FAILURE);
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

template <typename T>
int BatchNormDriver<T>::createSaveBuffers()
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
        saveMean_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));
        saveInvVariance_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));

        if(back == 1)
        {
            // GPU host allocation
            saveMean        = std::vector<float>(sb_sz);
            saveInvVariance = std::vector<float>(sb_sz);

            // CPU allocation
            saveMean_host        = std::vector<double>(sb_sz);
            saveInvVariance_host = std::vector<double>(sb_sz);

            // Populate
            for(int i = 0; i < sb_sz; i++)
            {
                saveMean_host[i] = saveMean[i] = double(rand() * (1.0 / RAND_MAX));
                saveInvVariance_host[i] = saveInvVariance[i] = double(rand() * (1.0 / RAND_MAX));
            }
        }
        else
        {
            // GPU host allocation
            saveMean        = std::vector<float>(sb_sz, 0);
            saveInvVariance = std::vector<float>(sb_sz, 0);

            // CPU allocation
            saveMean_host        = std::vector<double>(sb_sz, 0);
            saveInvVariance_host = std::vector<double>(sb_sz, 0);
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

template <typename T>
int BatchNormDriver<T>::createRunningBuffers()
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
        runningMean_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));
        runningVariance_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));

        if(forw == 2)
        {
            // GPU host allocation
            runningMean     = std::vector<float>(sb_sz);
            runningVariance = std::vector<float>(sb_sz);

            // CPU allocation
            runningMean_host     = std::vector<double>(sb_sz);
            runningVariance_host = std::vector<double>(sb_sz);

            // Populate
            for(int i = 0; i < sb_sz; i++)
            {
                runningMean_host[i] = runningMean[i] = double(rand() * (1.0 / RAND_MAX));
                runningVariance_host[i] = runningVariance[i] = double(rand() * (1.0 / RAND_MAX));
            }
        }
        else
        {
            // GPU host allocation
            runningMean     = std::vector<float>(sb_sz, 0);
            runningVariance = std::vector<float>(sb_sz, 0);

            // CPU allocation
            runningMean_host     = std::vector<double>(sb_sz, 0);
            runningVariance_host = std::vector<double>(sb_sz, 0);
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

template <typename T>
int BatchNormDriver<T>::AllocateBuffersAndCopy()
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
        in_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
        scale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));
        bias_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));
        out_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

        // GPU host allocation
        in    = std::vector<float>(in_sz);
        out   = std::vector<float>(out_sz, 0);
        scale = std::vector<float>(sb_sz);
        bias  = std::vector<float>(sb_sz);

        // CPU allocation
        out_host   = std::vector<double>(out_sz, 0);
        scale_host = std::vector<float>(sb_sz);
        bias_host  = std::vector<float>(sb_sz);

        // Data initialization
        for(int i = 0; i < in_sz; i++)
        {
            in[i] = std::fabs(double(rand() * (1.0 / RAND_MAX)));
        }
        status |= in_dev->ToGPU(q, in.data());

        // Using random beta and gamma
        for(int i = 0; i < sb_sz; i++)
        {
            scale[i] = scale_host[i] = double(rand() * (1.0 / RAND_MAX));
            bias[i] = bias_host[i] = double(rand() * (1.0 / RAND_MAX));
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
        in_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
        dyin_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
        dxout_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));
        dscale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));
        dbias_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));
        scale_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(float)));

        // GPU host allocation
        in     = std::vector<float>(in_sz);
        dyin   = std::vector<float>(in_sz);
        dxout  = std::vector<float>(out_sz, 0);
        dscale = std::vector<float>(sb_sz, 0);
        dbias  = std::vector<float>(sb_sz, 0);
        scale  = std::vector<float>(sb_sz);

        // CPU allocation
        dxout_host  = std::vector<double>(out_sz, 0);
        dscale_host = std::vector<double>(sb_sz, 0);
        dbias_host  = std::vector<double>(sb_sz, 0);

        // Populate
        for(int i = 0; i < sb_sz; i++)
        {
            scale[i] = double(rand() * (1.0 / RAND_MAX));
        }
        status |= scale_dev->ToGPU(q, scale.data());
        status |= dscale_dev->ToGPU(q, dscale.data());
        status |= dbias_dev->ToGPU(q, dbias.data());

        for(int i = 0; i < in_sz; i++)
        {
            dyin[i] = double(rand() * (1.0 / RAND_MAX));
            in[i]   = double(rand() * (1.0 / RAND_MAX));
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

template <typename T>
void BatchNormDriver<T>::runGPUFwdInference(double epsilon, T alpha, T beta)
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

template <typename T>
void BatchNormDriver<T>::runGPUFwdTrain(double epsilon, double eAF, T alpha, T beta)
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

template <typename T>
int BatchNormDriver<T>::RunForwardGPU()
{

    T alpha = 1, beta = 0;
    double epsilon = EPSILON;
    double eAF     = 1.0;

    Timer t;
    double fulltime = 0.;
    auto iters      = inflags.GetValueInt("iter");
    float lowtime   = 100000000.0;
    float avgtime   = 0.;

    for(int i = 0; i < iters; i++)
    {

        START_TIME;

        // if run fwd train
        if(forw == 1)
        { // training only
            eAF = 1.0 / (double(i) + 1.0);
            runGPUFwdTrain(epsilon, eAF, alpha, beta);
        }
        else if(forw == 2)
        { // inference only
            runGPUFwdInference(epsilon, alpha, beta);
        }
        else
        {
            // printf("Batch normalization mode forward GPU selection out of range, skipping.\n");
            return miopenStatusSuccess;
        }

        STOP_TIME;
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
        }
    }

    if(WALL_CLOCK)
    {
        printf("Wall-clock Time Forward GPU Batch Norm Elapsed: %f ms\n",
               (iters > 1) ? t.gettime_ms() : (fulltime / float(iters)));
    }

    if(inflags.GetValueStr("time") == "1")
    {
        printf("GPU Kernel Min Time Forward Batch Normalization Elapsed: %f ms\n", lowtime);
        if(iters > 1)
            printf("GPU Kernel Avg Time Forward Batch Normalization Elapsed: %f ms\n",
                   avgtime / (iters - 1));
    }
    return miopenStatusSuccess;
}

template <typename T>
void BatchNormDriver<T>::runCPUFwdInference(
    double epsilon, int batch_sz, int channels, int height, int width)
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

template <typename T>
void BatchNormDriver<T>::runCPUFwdTrain(
    double epsilon, double eAF, int batch_sz, int channels, int height, int width)
{

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        miopenBNFwdTrainPerActivationRunHost(/* alpha, beta, */ batch_sz,
                                             channels,
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
        miopenBNFwdTrainSpatialRunHost(/* alpha, beta, */ batch_sz,
                                       channels,
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
        exit(EXIT_FAILURE);
    }
}

template <typename T>
int BatchNormDriver<T>::RunForwardCPU()
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
    double epsilon = EPSILON;
    double eAF     = 1.0;

    if(forw == 1)
    { // training only
        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            eAF = 1.0 / (double(i) + 1.0);
            runCPUFwdTrain(epsilon, eAF, /* alpha, beta,*/ batch_sz, channels, height, width);
        }
    }
    else if(forw == 2)
    { // inference only
        runCPUFwdInference(epsilon, /* alpha, beta,*/ batch_sz, channels, height, width);
    }

    return miopenStatusSuccess;
}

template <typename T>
int BatchNormDriver<T>::RunBackwardGPU()
{

    if(!back)
        return miopenStatusSuccess;

    T alphaDataDiff = 1, betaDataDiff = 0;
    T alphaParamDiff = 1, betaParamDiff = 0;
    double epsilon = EPSILON;

    Timer t;
    double fulltime = 0.;
    auto iters      = inflags.GetValueInt("iter");
    float lowtime   = 100000000.0;
    float avgtime   = 0.;

    for(int i = 0; i < iters; i++)
    {
        START_TIME;

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
        STOP_TIME;
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
        }
    }

    if(WALL_CLOCK)
    {
        printf("Wall-clock Time Backward GPU Batch Norm Elapsed: %f ms\n",
               (iters > 1) ? t.gettime_ms() : (fulltime / float(iters)));
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

template <typename T>
int BatchNormDriver<T>::VerifyForward()
{

    // jump out since we are forcing forward off when doing backwards.
    if(!forw)
        return miopenStatusSuccess;

    const double tolerance = ERRTOL;
    const double maxrms    = RMSTOL;
    double diff            = 0.;
    bool anError           = false;

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
                    diff = fabs(float(fabs(runningMean[i]) - fabs(runningMean_host[i])));
                    if(diff > tolerance)
                    {
                        std::cout << "rm[" << i << "]: " << runningMean[i];
                        std::cout << ", rm_host[" << i << "]: " << runningMean_host[i];
                        std::cout << ", diff[" << i << "]: "
                                  << float(fabs(runningMean[i]) - fabs(runningMean_host[i]))
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
                    diff = fabs(float(fabs(runningVariance[i]) - fabs(runningVariance_host[i])));
                    if(diff > tolerance)
                    {
                        std::cout << "rv[" << i << "]: " << runningVariance[i];
                        std::cout << ", rv_host[" << i << "]: " << runningVariance_host[i];
                        std::cout << ", diff[" << i << "]: "
                                  << float(fabs(runningVariance[i]) - fabs(runningVariance_host[i]))
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
            maxval             = 0.;
            auto errorSaveMean = miopen::rms_range(saveMean_host, saveMean);
            if(errorSaveMean > maxrms || std::isnan(errorSaveMean))
            {
                std::cout << "Forward train batch norm verification failed on saved mean: "
                          << errorSaveMean << "\n";
                anError = true;
                //#if(MIO_BN_DEBUG == 1)
                for(int i = 0;
                    i < saveMean.size() && i < saveMean_host.size() && i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff   = fabs(float(fabs(saveMean[i]) - fabs(saveMean_host[i])));
                    maxval = maxval < diff ? diff : maxval;
                    if(diff > tolerance)
                    {
                        std::cout << "sm[" << i << "]: " << saveMean[i];
                        std::cout << ", sm_host[" << i << "]: " << saveMean_host[i];
                        std::cout << ", diff[" << i
                                  << "]: " << float(fabs(saveMean[i]) - fabs(saveMean_host[i]))
                                  << std::endl;
                    }
                }
                //#endif
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
                    diff = fabs(float(fabs(saveInvVariance[i]) - fabs(saveInvVariance_host[i])));
                    if(diff > tolerance)
                    {
                        std::cout << "sv[" << i << "]: " << saveInvVariance[i];
                        std::cout << ", sv_host[" << i << "]: " << saveInvVariance_host[i];
                        std::cout << ", diff[" << i << "]: "
                                  << float(fabs(saveInvVariance[i]) - fabs(saveInvVariance_host[i]))
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
    maxval        = 0.;
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
            diff   = double(fabs(out[i]) - fabs(out_host[i]));
            maxval = maxval < diff ? diff : maxval;
            if(diff > tolerance)
            {
                std::cout << "out[" << i << "]: " << out[i];
                std::cout << ", out_host[" << i << "]: " << out_host[i];
                std::cout << ", diff[" << i << "]: " << double(out[i] - out_host[i]) << std::endl;
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
        std::cout << "Forward batch norm verified on CPU and GPU." << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename T>
int BatchNormDriver<T>::RunBackwardCPU()
{

    if(!back)
        return miopenStatusSuccess;

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

    //	T alphaDiff = 1, betaDiff = 0;
    //	T alphaParam = 1, betaParam = 0;
    double epsilon = EPSILON;

    if(bn_mode == miopenBNPerActivation)
    {                                   // 1xCxHxW
        miopenBNBwdPerActivationRunHost(/* alphaDiff, betaDiff, alphaParam, betaParam, */
                                        batch_sz,
                                        channels,
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
    {                             // 1xCx1x1
        miopenBNBwdSpatialRunHost(/* alphaDiff, betaDiff, alphaParam, betaParam, */
                                  batch_sz,
                                  channels,
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
        exit(EXIT_FAILURE);
    }

    return miopenStatusSuccess;
}

template <typename T>
int BatchNormDriver<T>::VerifyBackward()
{

    if(!back)
        return miopenStatusSuccess;

    const double maxrms = RMSTOL * 1000;
    bool anError        = false;

    RunBackwardCPU();

    dxout_dev->FromGPU(GetStream(), dxout.data());
    dscale_dev->FromGPU(GetStream(), dscale.data());
    dbias_dev->FromGPU(GetStream(), dbias.data());

    maxval          = 0.;
    auto errordxout = miopen::rms_range(dxout_host, dxout);
    if(errordxout > maxrms || std::isnan(errordxout))
    {
        std::cout << "Backwards prop batch norm verification failed on dx: " << errordxout << "\n";
        anError = true;
#if(MIO_BN_DEBUG == 1)
        const double tolerance = ERRTOL * 1000;
        double diff            = 0.;
        for(int i = 0; i < dxout.size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff   = fabs(float(fabs(dxout[i]) - fabs(dxout_host[i])));
            maxval = maxval < diff ? diff : maxval;
            if(diff > tolerance)
            {
                std::cout << "dxout[" << i << "]: " << dxout[i];
                std::cout << "\tdxout_host[" << i << "]: " << dxout_host[i];
                std::cout << "\tdiff[" << i << "]: " << float(fabs(dxout[i]) - fabs(dxout_host[i]));
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

    maxval           = 0.;
    auto errordscale = miopen::rms_range(dscale_host, dscale);
    if(errordscale > maxrms || std::isnan(errordscale))
    {
        std::cout << "Backwards prop batch norm verification failed on dscale: " << errordscale
                  << "\n";
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < dscale.size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff   = fabs(float(fabs(dscale[i]) - fabs(dscale_host[i])));
            maxval = maxval < diff ? diff : maxval;
            if(diff > tolerance)
            {
                std::cout << "dscale[" << i << "]: " << dscale[i];
                std::cout << "\tdscale_host[" << i << "]: " << dscale_host[i];
                std::cout << "\tdiff[" << i
                          << "]: " << float(fabs(dscale[i]) - fabs(dscale_host[i]));
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
            diff = fabs(float(fabs(dbias[i]) - fabs(dbias_host[i])));
            if(diff > tolerance)
            {
                std::cout << "dbias[" << i << "]: " << dbias[i];
                std::cout << "\tdbias_host[" << i << "]: " << dbias_host[i];
                std::cout << "\tdiff[" << i << "]: " << float(fabs(dbias[i]) - fabs(dbias_host[i]));
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
        std::cout << "Backwards prop batch norm verified on CPU and GPU." << std::endl;

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BN_DRIVER_HPP
