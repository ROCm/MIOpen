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
#include <miopen/md_graph.hpp>
#include <numeric>
#include <vector>
#include <cassert>
#include "random.hpp"
#include "mloNeuronHost.hpp"

#define MIO_BN_DEBUG 0
#define MIO_BN_MAX_DEBUGLOOP 65536

#undef EPSILON
#define EPSILON 1e-6

//#define MIO_CONV_ALGO_COUNT 4

#define ERRTOL 1e-4
#define RMSTOL_FP32 1e-4
#define RMSTOL_FP16 0.5e-3

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

#define CBA_DEBUG_VALUES 0

//"Fusion mode (cbna = 0, cna = 1, na = 2, cn = 3, cba = 4, ca = 5, cb = 6) (Default=cbna)",
typedef enum
{
    miopen_fusion_cbna = 0,
    miopen_fusion_cna  = 1,
    miopen_fusion_na   = 2,
    miopen_fusion_cn   = 3,
    miopen_fusion_cba  = 4,
    miopen_fusion_ca   = 5,
    miopen_fusion_cb   = 6,
} fusionMode_t;

template <typename Tgpu, typename Tref>
class CBAInferFusionDriver : public Driver
{
public:
    CBAInferFusionDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasTensor);
        miopenCreateTensorDescriptor(&biasScaleTensor);
        miopenCreateConvolutionDescriptor(&convDesc);
        miopenCreateActivationDescriptor(&activDesc);
        miopenCreateOperatorArgs(&fusionArgs);

        workspace_fwd_dev = nullptr;

        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
        initTiming();
        iters = 0;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetOutputTensorLengths();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();
    std::vector<int> GetModeFromCmdLine();

    int SetActivationDescriptorFromCmdLineArgs();
    int SetConvDescriptorFromCmdLineArgs();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override { return 0; };
    int RunBackwardCPU() { return 0; };

    void runGPUConvBatchNormActivInference();
    void runGPUConvActivInference();
    void runGPUBatchNormActivInference();

    void runGPUBNFwdInference();
    void runCPUBNFwdInference();

    void runGPUActivFwdInference();
    void runCPUActivFwdInference();

    void runCPUConvFwdInference();

    void runGPUConvBiasInference();
    void runGPUFusedConvBiasInference();
    void runCPUConvBiasInference();

    int VerifyBackward() override { return 0; };
    int VerifyForward() override;

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
        START_TIME
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
        STOP_TIME

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

    ~CBAInferFusionDriver() override
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(weightTensor);
        miopenDestroyTensorDescriptor(biasTensor);
        miopenDestroyTensorDescriptor(biasScaleTensor);
        miopenDestroyActivationDescriptor(activDesc);
        miopenDestroyConvolutionDescriptor(convDesc);

        miopenDestroyFusionPlan(fusePlanDesc);
        miopenDestroyOperatorArgs(fusionArgs);
    }

private:
    miopenBatchNormMode_t bn_mode;
    int bias_mode   = 0;
    int fusion_mode = 0;
    bool estimatedMeanVar;
    bool useBatchNorm = false;
    unsigned char back;

    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t biasScaleTensor;
    miopenTensorDescriptor_t biasTensor;
    miopenTensorDescriptor_t outputTensor;

    miopenActivationDescriptor_t activDesc;
    miopenConvolutionDescriptor_t convDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> conv_res_dev;
    std::unique_ptr<GPUMem> bn_res_dev;
    std::unique_ptr<GPUMem> wei_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> scale_dev;
    std::unique_ptr<GPUMem> bias_dev;
    std::unique_ptr<GPUMem> workspace_fwd_dev;
    std::unique_ptr<GPUMem> runningMean_dev;
    std::unique_ptr<GPUMem> runningVariance_dev;
    std::unique_ptr<GPUMem> saveMean_dev;
    std::unique_ptr<GPUMem> saveInvVariance_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::vector<Tgpu> b;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tgpu> wei;
    std::vector<Tgpu> conv_res;
    std::vector<Tgpu> bn_res;
    std::vector<Tref> in_host;
    std::vector<Tref> conv_res_host;
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
    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t biasOp;
    miopenFusionOpDescriptor_t activOp;
    miopenOperatorArgs_t fusionArgs;
};

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueStr("dot_graph") != "")
    {
        std::string op = inflags.GetValueStr("dot_graph");
        miopen::FusionMDGraph mdg;
        if(op == "ConvForward")
        {
            miopen::FusionMDGraph::InitConv(mdg);
        }
        else if(op == "BatchNormInference")
        {
            miopen::FusionMDGraph::InitBN(mdg);
        }
        else
        {
            std::cerr
                << "Invalid Graph specified, valid values are ConvForward or BatchNormInference"
                << std::endl;
        }
        mdg.WriteToFile("/tmp/mdgraph.dot");
        std::cerr << "Graph written to /tmp/mdgraph.dot" << std::endl;
        exit(EXIT_SUCCESS); // NOLINT (concurrency-mt-unsafe)
    }

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    fusion_mode = inflags.GetValueInt("fusion_mode");
    if(fusion_mode > 6 || fusion_mode < 0)
    {
        std::cout << "Fusion mode out of range.\n Exiting..." << std::endl;
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }
    if(fusion_mode != miopen_fusion_cba && fusion_mode != miopen_fusion_ca &&
       fusion_mode != miopen_fusion_cb)
        useBatchNorm = true;
    else
        useBatchNorm = false;

    if(fusion_mode == miopen_fusion_cbna || fusion_mode == miopen_fusion_cba ||
       fusion_mode == miopen_fusion_cb)
        bias_mode = 1;
    else
        bias_mode = 0;

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
std::vector<int> CBAInferFusionDriver<Tgpu, Tref>::GetWeightTensorLengthsFromCmdLine()
{
    int wei_n = inflags.GetValueInt("out_channels");
    int wei_c = inflags.GetValueInt("in_channels");
    int wei_h = inflags.GetValueInt("fil_h");
    int wei_w = inflags.GetValueInt("fil_w");

    return std::vector<int>({wei_n, wei_c, wei_h, wei_w});
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::GetandSetData()
{

    SetBNParametersFromCmdLineArgs();
    SetConvDescriptorFromCmdLineArgs();
    SetActivationDescriptorFromCmdLineArgs();

    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len, data_type);

    miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputTensor);

    SetTensor4d(weightTensor, wei_len, data_type);

    std::vector<int> out_len{};
    if(fusion_mode != miopen_fusion_na)
    {
        out_len = GetOutputTensorLengths();
    }
    else
    {
        out_len = in_len;
    }
    SetTensor4d(outputTensor, out_len, data_type);

    if(bias_mode)
    {
        std::vector<int> b_len{1, out_len[1], 1, 1};
        SetTensor4d(biasTensor, b_len, data_type);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("batchsize", 'n', "32", "Mini-batch size (Default=32)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
    inflags.AddInputFlag(
        "group_count", 'g', "1", "Number of groups in convolution (Default=1)", "int");
    inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    inflags.AddInputFlag(
        "conv_stride_h", 'u', "1", "Convolution Stride Vertical (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_w", 'v', "1", "Convolution Stride Horizontal (Default=1)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    inflags.AddInputFlag("alpha", 'A', "1.0", "Alpha (Default=1.0)", "float");
    inflags.AddInputFlag("beta", 'B', "0.", "Beta (Default=0.)", "float");
    inflags.AddInputFlag("gamma", 'G', "1", "Activation gamma (Default=1)", "double");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");

    /*inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)",
     * "int");*/

    inflags.AddInputFlag(
        "activMode", 'm', "3", "Activation Mode (relu,..., see spec) (Default=3(relu))", "int");
    inflags.AddInputFlag("bnMode",
                         'M',
                         "0",
                         "Normalization Mode (per-activation (0) or spatial (1)) (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
    inflags.AddInputFlag("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");

    /*inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");*/

    inflags.AddInputFlag(
        "pad_mode", 'z', "conv", "Padding Mode (same, valid, default) (Default=default)", "str");

    inflags.AddInputFlag(
        "fusion_mode",
        'F',
        "0",
        "Fusion mode (cbna = 0, cna = 1, na = 2, cn = 3, cba = 4, ca = 5, cb = 6) (Default=cbna)",
        "int");
    inflags.AddInputFlag("dot_graph",
                         'D',
                         "",
                         "Write out the fusion metadata graph for the specified operator",
                         "str");

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
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::SetConvDescriptorFromCmdLineArgs()
{

    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopenPaddingDefault;
    int in_h                  = inflags.GetValueInt("in_h");
    int in_w                  = inflags.GetValueInt("in_w");
    int wei_h                 = inflags.GetValueInt("fil_h");
    int wei_w                 = inflags.GetValueInt("fil_w");
    int pad_h                 = inflags.GetValueInt("pad_h");
    int pad_w                 = inflags.GetValueInt("pad_w");
    int stride_h              = inflags.GetValueInt("conv_stride_h");
    int stride_w              = inflags.GetValueInt("conv_stride_w");
    int dilation_h            = inflags.GetValueInt("dilation_h");
    int dilation_w            = inflags.GetValueInt("dilation_w");
    int group_count           = inflags.GetValueInt("group_count");

    pmode = miopenPaddingDefault;
    mode  = miopenConvolution;

    if((inflags.GetValueStr("pad_mode")) == "same")
    {
        mode  = miopenConvolution;
        pmode = miopenPaddingSame;
        pad_h = (in_h % stride_h == 0) ? (std::max((wei_h - stride_h), 0))
                                       : (std::max((wei_h - (in_h % stride_h)), 0));
        pad_w = (in_w % stride_w == 0) ? (std::max((wei_w - stride_w), 0))
                                       : (std::max((wei_w - (in_w % stride_w)), 0));
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

    miopen::deref(convDesc) = miopen::ConvolutionDescriptor(2,
                                                            mode,
                                                            pmode,
                                                            {pad_h, pad_w},
                                                            {stride_h, stride_w},
                                                            {dilation_h, dilation_w},
                                                            {0, 0},
                                                            group_count);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> CBAInferFusionDriver<Tgpu, Tref>::GetOutputTensorLengths()
{
    int n, c, h, w;
    miopenGetConvolutionForwardOutputDim(convDesc, inputTensor, weightTensor, &n, &c, &h, &w);
    return std::vector<int>({n, c, h, w});
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

    if(useBatchNorm)
    {
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
        {
            printf("Error copying data to GPU\n");
            exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
        }
    }
    else
    {
        runningMean_dev     = nullptr;
        runningVariance_dev = nullptr;
    }
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

    size_t in_sz  = GetTensorSize(inputTensor);
    size_t wei_sz = GetTensorSize(weightTensor);
    size_t sb_sz  = 0;
    if(useBatchNorm)
    {
        miopenDeriveBNTensorDescriptor(biasScaleTensor, inputTensor, bn_mode);
        sb_sz = GetTensorSize(biasScaleTensor);
    }

    size_t out_sz = 0;
    if(fusion_mode != miopen_fusion_na)
        out_sz = GetTensorSize(outputTensor);
    else
        out_sz = in_sz; // This is for N+A so the output is the same as the input size

    if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
    {
        PadBufferSize(wei_sz, sizeof(Tgpu));
    }

    if(bias_mode)
    {
        size_t b_sz = GetTensorSize(biasTensor);
        b_dev       = std::make_unique<GPUMem>(ctx, b_sz, sizeof(Tgpu));
        b           = std::vector<Tgpu>(b_sz, static_cast<Tgpu>(0));
        for(int i = 0; i < b_sz; i++)
        {
            b[i] = std::fabs(RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0)));
        }
        status |= b_dev->ToGPU(q, b.data());
    }

    // GPU allocation
    in_dev       = std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu));
    conv_res_dev = std::make_unique<GPUMem>(ctx, out_sz, sizeof(Tgpu));
    wei_dev      = std::make_unique<GPUMem>(ctx, wei_sz, sizeof(Tgpu));
    out_dev      = std::make_unique<GPUMem>(ctx, out_sz, sizeof(Tgpu));

    if(useBatchNorm)
    {
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
            scale[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            bias[i]  = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
#endif
        }
        status |= scale_dev->ToGPU(q, scale.data());
        status |= bias_dev->ToGPU(q, bias.data());
    }

    // GPU host allocation
    in       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    conv_res = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    // CPU allocation
    in_host       = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    conv_res_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    out_host      = std::vector<Tref>(out_sz, static_cast<Tref>(0));

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

    if(fusion_mode != miopen_fusion_na)
    {
        wei = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
        for(int i = 0; i < wei_sz; i++)
        {
#if(CBA_DEBUG_VALUES == 1)
            wei[i] = 1.; // std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0),
                         // static_cast<Tgpu>(1.0))); // 1.;
#else
            wei[i] = std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0)));
#endif
        }
        status |= wei_dev->ToGPU(q, wei.data());
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
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
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
void CBAInferFusionDriver<Tgpu, Tref>::runGPUConvBatchNormActivInference()
{
    miopenError = miopenStatusSuccess;
    double activ_alpha, activ_beta, activ_gamma;
    miopenActivationMode_t activ_mode;
    miopenGetActivationDescriptor(activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

    double epsilon = static_cast<double>(EPSILON);
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    int stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w;
    std::string plan_error_str;

    miopenConvolutionMode_t mode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w);

    miopenCreateOpConvForward(fusePlanDesc, &convoOp, convDesc, weightTensor);
    plan_error_str += "Convolution";

    if(bias_mode)
    {
        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor);
        plan_error_str += "+Bias";
    }

    if(fusion_mode != miopen_fusion_cba)
    {
        miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bn_mode, biasScaleTensor);
        plan_error_str += "+BatchNorm";
    }

    if(fusion_mode != miopen_fusion_cn)
    {
        miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);
        plan_error_str += "+Activation";
    }

    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev->GetMem());

    if(bias_mode)
    {
        miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev->GetMem());
    }

    if(fusion_mode != miopen_fusion_cn)
    {
        miopenSetOpArgsActivForward(
            fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    }

    if(fusion_mode != miopen_fusion_cba)
        miopenSetOpArgsBatchNormInference(fusionArgs,
                                          bNormOp,
                                          &alpha,
                                          &beta,
                                          scale_dev->GetMem(),
                                          bias_dev->GetMem(),
                                          runningMean_dev->GetMem(),
                                          runningVariance_dev->GetMem(),
                                          epsilon);
    miopenError = miopenCompileFusionPlan(GetHandle(), fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        std::cerr << plan_error_str << " plan not supported." << std::endl;
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
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
void CBAInferFusionDriver<Tgpu, Tref>::runGPUConvActivInference()
{
    miopenError = miopenStatusSuccess;
    double activ_alpha, activ_beta, activ_gamma;
    miopenActivationMode_t activ_mode;
    miopenGetActivationDescriptor(activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    int stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w);

    miopenCreateOpConvForward(fusePlanDesc, &convoOp, convDesc, weightTensor);

    if(bias_mode)
    {
        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor);
    }

    miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);

    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev->GetMem());

    miopenSetOpArgsActivForward(
        fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

    if(bias_mode)
    {
        miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev->GetMem());
    }

    miopenError = miopenCompileFusionPlan(GetHandle(), fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        if(bias_mode)
            std::cerr << "ConvBiasActivInference plan not supported." << std::endl;
        else
            std::cerr << "ConvActivInference plan not supported." << std::endl;
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
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
                                             conv_res_dev->GetMem(),
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
                                        (!useBatchNorm) ? conv_res_host.data() : bn_res_host.data(),
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
void CBAInferFusionDriver<Tgpu, Tref>::runGPUConvBiasInference()
{

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    if(bias_mode)
    {
        miopenConvolutionForwardBias(GetHandle(),
                                     &alpha,
                                     biasTensor,
                                     b_dev->GetMem(),
                                     &beta,
                                     outputTensor,
                                     conv_res_dev->GetMem());
    }
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runGPUFusedConvBiasInference()
{

    miopenError = miopenStatusSuccess;
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    int stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w);
    miopenCreateOpConvForward(fusePlanDesc, &convoOp, convDesc, weightTensor);
    miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor);
    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev->GetMem());
    miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev->GetMem());
    miopenError = miopenCompileFusionPlan(GetHandle(), fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        std::cerr << "ConvBiasInference plan not supported." << std::endl;
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
int CBAInferFusionDriver<Tgpu, Tref>::RunForwardGPU()
{
    //"Fusion mode (cbna = 0, cna = 1, na = 2, cn = 3, cba = 4, ca = 5, cb = 6) (Default=cbna)"
    assert(fusion_mode < 7 && fusion_mode >= 0);
    iters = inflags.GetValueInt("iter");
    std::cout << "Running fusion: ";
    switch(fusion_mode)
    {
    case 0: std::cout << "Convolution+Bias+BatchNorm+Activation" << std::endl; break;
    case 1: std::cout << "Convolution+BatchNorm+Activation" << std::endl; break;
    case 2: std::cout << "BatchNorm+Activation" << std::endl; break;
    case 3: std::cout << "Convolution+BatchNorm" << std::endl; break;
    case 4: std::cout << "Convolution+Bias+Activation" << std::endl; break;
    case 5: std::cout << "Convolution+Activation" << std::endl; break;
    case 6: std::cout << "Convolution+Bias" << std::endl; break;
    }
    initTiming();
    switch(fusion_mode)
    {
    case 0:
    case 1:
    case 3:
    case 4: runGPUConvBatchNormActivInference(); break;
    case 5: runGPUConvActivInference(); break;
    case 2: runGPUBatchNormActivInference(); break;
    case 6: runGPUFusedConvBiasInference(); break;
    }

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
void CBAInferFusionDriver<Tgpu, Tref>::runCPUConvFwdInference()
{
    ConvForwardCPU<Tgpu, Tref>(in_host,
                               fusion_mode != miopen_fusion_cb ? conv_res_host
                                                               : out_host, // dlowell 6 or 5???
                               wei,
                               b,
                               bias_mode,
                               convDesc,
                               inputTensor,
                               weightTensor,
                               outputTensor);

    return;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runCPUBNFwdInference()
{
    double epsilon = static_cast<double>(EPSILON);

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        std::cout << "Running CPU per activation BN." << std::endl;
        miopenBNPerActivFwdInferHost(
            fusion_mode != miopen_fusion_na ? outputTensor
                                            : inputTensor, // DLOWELL use output for splice test
            fusion_mode != miopen_fusion_na
                ? conv_res_host.data()
                : in_host.data(), // conv_res_host.data(), //DLOWELL use conv for splice test
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
        miopenBNSpatialFwdInferHost(
            fusion_mode != miopen_fusion_na ? outputTensor
                                            : inputTensor, // DLOWELL use output for splice test
            fusion_mode != miopen_fusion_na
                ? conv_res_host.data()
                : in_host.data(), // conv_res_host.data(), //DLOWELL use conv for splice test
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
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }
    // C+N mode so we are done
    if(fusion_mode == miopen_fusion_cn)
        out_host = bn_res_host; // DLOWELL if we add C+B+N the is to be modified

    return;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::RunForwardCPU()
{
    //"Fusion mode (cbna = 0, cna = 1, na = 2, cn = 3, cba = 4, ca = 5, cb = 6) (Default=cbna)"
    MIOPEN_LOG_I("Fusion mode: " << fusion_mode);
    if(fusion_mode != miopen_fusion_na)
    {
        std::cout << "Running CPU fwd convolution." << std::endl;
        runCPUConvFwdInference();
    }

    if(useBatchNorm)
    {
        // std::cout << "Running CPU fwd batch normalization." << std::endl;
        runCPUBNFwdInference();
    }

    if(fusion_mode != miopen_fusion_cb && fusion_mode != miopen_fusion_cn)
    {
        std::cout << "Running CPU fwd activation." << std::endl;
        runCPUActivFwdInference();
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    double allowedEps = std::numeric_limits<Tgpu>::epsilon() * 80;

    int match = miopenInferVerify(out.size(), out_host.data(), out.data(), allowedEps);
    if(match == 0)
    {
        std::cout << "Forward Activation FAILED" << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "Forward Activation Verifies on CPU and GPU" << std::endl;
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BN_DRIVER_HPP
