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

#include <miopen/conv_batch_norm_activ.hpp>
#include <miopen/batch_norm_activ.hpp>
#include <miopen/direct_conv_ocl.hpp>

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
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasTensor);
        miopenCreateTensorDescriptor(&biasScaleTensor);
        miopenCreateConvolutionDescriptor(&convDesc);
        miopenCreateActivationDescriptor(&activDesc);

        // DLOWELL starting point for fusion plans
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputTensor);
        miopenCreateOperatorArgs(&fusionArgs);

        workspace_fwd_dev = nullptr;

        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetOutputTensorLengths();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();
    std::vector<int> GetModeFromCmdLine();

    int SetActivationDescriptorFromCmdLineArgs();
    int SetConvDescriptorFromCmdLineArgs();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy();

    int FindConvForward(int& ret_algo_count,
                        int request_algo_count,
                        std::vector<miopenConvAlgoPerf_t>& perf_results);

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU();
    int RunBackwardCPU();

    void runGPUConvBatchNormActivInference();
    void runGPUConvActivInference();
    void runGPUBatchNormActivInference();

    void runGPUBNFwdInference();
    void runCPUBNFwdInference();

    void runGPUActivFwdInference();
    void runCPUActivFwdInference();

    void runGPUConvFwdInference();
    void runCPUConvFwdInference();

    void runGPUConvBiasInference();
    void runGPUFusedConvBiasInference();
    void runCPUConvBiasInference();

    int VerifyBackward();
    int VerifyForward();

    ~CBAInferFusionDriver()
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(weightTensor);
        miopenDestroyTensorDescriptor(biasTensor);
        miopenDestroyTensorDescriptor(biasScaleTensor);

        miopenDestroyActivationDescriptor(activDesc);
        miopenDestroyConvolutionDescriptor(convDesc);
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

    // Tref maxval;
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
    SetTensor4d(weightTensor, wei_len, data_type);

    std::vector<int> out_len = GetOutputTensorLengths();
    SetTensor4d(outputTensor, out_len, data_type);

    std::vector<int> sb_len;

    if(bias_mode != 0)
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
    inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    inflags.AddInputFlag(
        "conv_stride_0", 'u', "1", "Convolution Stride Vertical (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_1", 'v', "1", "Convolution Stride Horizontal (Default=1)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
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
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
    inflags.AddInputFlag("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");
    inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");

    inflags.AddInputFlag(
        "pad_mode", 'z', "conv", "Padding Mode (same, valid, default) (Default=default)", "str");

    inflags.AddInputFlag(
        "fusion_mode",
        'F',
        "0",
        "Fusion mode (cbbna = 0, cbna = 1, bna = 2, cba = 3, ca = 4, cb = 5) (Default=cbbna)",
        "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> CBAInferFusionDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    // std::cerr << "inTensor = " << in_n << "," << in_c << "," << in_h << "," << in_w << std::endl;

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
    int u                     = inflags.GetValueInt("conv_stride_0");
    int v                     = inflags.GetValueInt("conv_stride_1");
    int dilation_h            = inflags.GetValueInt("dilation_h");
    int dilation_w            = inflags.GetValueInt("dilation_w");

    pmode = miopenPaddingDefault;
    mode  = miopenConvolution;

    if((inflags.GetValueStr("pad_mode")) == "same")
    {
        mode  = miopenConvolution;
        pmode = miopenPaddingSame;
        pad_h = (in_h % u == 0) ? (std::max((wei_h - u), 0)) : (std::max((wei_h - (in_h % u)), 0));
        pad_w = (in_w % v == 0) ? (std::max((wei_w - v), 0)) : (std::max((wei_w - (in_w % v)), 0));
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

    //(cbbna = 0, cbna = 1, bna = 2, cba = 3, ca = 4, cb = 5)
    if(fusion_mode < 2 || fusion_mode == 3 || fusion_mode == 5)
        bias_mode = 1;
    else
        bias_mode = 0;

    if(wei_h != wei_w)
    {
        std::cerr << "Direct is not supported" << std::endl;
        exit(0);
    }

    miopen::deref(convDesc) =
        miopen::ConvolutionDescriptor(mode, pmode, pad_h, pad_w, u, v, dilation_h, dilation_w);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> CBAInferFusionDriver<Tgpu, Tref>::GetOutputTensorLengths()
{
    int n, c, h, w;

    miopenGetConvolutionForwardOutputDim(convDesc, inputTensor, weightTensor, &n, &c, &h, &w);

    // std::cerr << "outTensor = " << n << "," << c << "," << h << "," << w << std::endl;

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
    size_t sb_sz = GetTensorSize(biasScaleTensor);
    if(fusion_mode < 3)
    {
        // GPU allocation
        runningMean_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
        runningVariance_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));

        // GPU host allocation
        runningMean     = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
        runningVariance = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));

        // Populate
        for(int i = 0; i < sb_sz; i++)
        {
            runningMean[i]     = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            runningVariance[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
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
    if(fusion_mode < 3)
    {
        miopenDeriveBNTensorDescriptor(biasScaleTensor, inputTensor, bn_mode);
        sb_sz = GetTensorSize(biasScaleTensor);
    }

    size_t out_sz = GetTensorSize(outputTensor);

    size_t workSpaceSize_fwd = 0;
    miopenConvolutionForwardGetWorkSpaceSize(
        GetHandle(), weightTensor, inputTensor, convDesc, outputTensor, &workSpaceSize_fwd);

    // Workaround: Pad buffers allocations to be a multiple of 2M
    if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
    {
        // PadBufferSize(in_sz, sizeof(Tgpu));
        PadBufferSize(wei_sz, sizeof(Tgpu));
        PadBufferSize(out_sz, sizeof(Tgpu));
    }

// size_t workSpaceNbVal_fwd = workSpaceSize_fwd / sizeof(Tgpu);
// if(workSpaceSize_fwd != 0)
//{
// workspace_fwd_dev =
// std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal_fwd, sizeof(Tgpu)));
//}

// std::cerr << "workSpaceNbVal_fwd = " << workSpaceNbVal_fwd << std::endl;

#if 1
    if(bias_mode != 0)
    {
        size_t b_sz = GetTensorSize(biasTensor);
        b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(Tgpu)));
        b           = std::vector<Tgpu>(b_sz, static_cast<Tgpu>(0));
        for(int i = 0; i < b_sz; i++)
        {
            b[i] = static_cast<Tgpu>(i % 8);
        }
        status |= b_dev->ToGPU(q, b.data());
    }
#endif

    // GPU allocation
    in_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    conv_res_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    wei_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
    out_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    if(fusion_mode < 3)
    {
        scale       = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
        bias        = std::vector<Tgpu>(sb_sz, static_cast<Tgpu>(0));
        bn_res      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        bn_res_host = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        bn_res_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        scale_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
        bias_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, sb_sz, sizeof(Tgpu)));
        // Using random beta and gamma
        for(int i = 0; i < sb_sz; i++)
        {
            scale[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            bias[i]  = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
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
        auto rval  = std::fabs(RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0)));
        in_host[i] = static_cast<double>(rval);
        in[i]      = rval;
    }

    // DLOWELL @todo set a gaurd here only for convolutions
    if(fusion_mode != 2)
    {
        wei = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
        for(int i = 0; i < wei_sz; i++)
        {
            wei[i] = std::fabs(RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0)));
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

    miopenCreateOpActivationForward(fusePlanDesc, &activOp, activDesc);
    miopenSetOpArgsBatchNormInference(fusionArgs,
                                      bNormOp,
                                      &alpha,
                                      &beta,
                                      scale_dev->GetMem(),
                                      bias_dev->GetMem(),
                                      runningMean_dev->GetMem(),
                                      runningVariance_dev->GetMem(),
                                      epsilon);
    miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta);
    miopenError = miopenIsFusionPlanValid(fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        std::cerr << "BatchNormActivInference plan not supported." << std::endl;
    }

    miopenExecuteFusionPlan(GetHandle(),
                            fusePlanDesc,
                            inputTensor,
                            in_dev->GetMem(),
                            outputTensor,
                            out_dev->GetMem(),
                            fusionArgs);

    /*miopen::BatchNormActivInference(miopen::deref(GetHandle()),
                                    bn_mode,
                                    &alpha,
                                    &beta,
                                    miopen::deref(outputTensor),
                                    conv_res_dev->GetMem(),
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
                                    activ_gamma);*/
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

    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  convDesc,
                                  miopenConvolutionFwdAlgoDirect, // DLOWELL Hardcoded. This assumes
                                                                  // immediate mode. Needs GetAlgo.
                                  weightTensor);

    if(bias_mode)
    {
        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor);
    }
    miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bn_mode, biasScaleTensor);
    miopenCreateOpActivationForward(fusePlanDesc, &activOp, activDesc);
    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev->GetMem());

    if(bias_mode)
    {
        miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev->GetMem());
    }

    miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta);
    miopenSetOpArgsBatchNormInference(fusionArgs,
                                      bNormOp,
                                      &alpha,
                                      &beta,
                                      scale_dev->GetMem(),
                                      bias_dev->GetMem(),
                                      runningMean_dev->GetMem(),
                                      runningVariance_dev->GetMem(),
                                      epsilon);
    miopenError = miopenIsFusionPlanValid(fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        if(bias_mode)
            std::cerr << "ConvBiasBatchNormActivInference plan not supported." << std::endl;
        else
            std::cerr << "ConvBatchNormActivInference plan not supported." << std::endl;
    }

    miopenExecuteFusionPlan(GetHandle(),
                            fusePlanDesc,
                            inputTensor,
                            in_dev->GetMem(),
                            outputTensor,
                            out_dev->GetMem(),
                            fusionArgs);

    /*    miopen::DirectConvBNActivInference(miopen::deref(GetHandle()),
                                           &alpha,
                                           miopen::deref(inputTensor),
                                           in_dev->GetMem(),
                                           miopen::deref(weightTensor),
                                           wei_dev->GetMem(),
                                           &beta,
                                           miopen::deref(outputTensor),
                                           out_dev->GetMem(),
                                           pad_h,
                                           pad_w,
                                           u,
                                           v,
                                           dilation_h,
                                           dilation_w,
                                           bias_mode,
                                           bias_mode != 0 ? b_dev->GetMem() : nullptr,
                                           bn_mode,
                                           fusion_mode < 3 ? scale_dev->GetMem() : nullptr,
                                           fusion_mode < 3 ? bias_dev->GetMem() : nullptr,
                                           fusion_mode < 3 ? runningMean_dev->GetMem() : nullptr,
                                           fusion_mode < 3 ? runningVariance_dev->GetMem() :
       nullptr,
                                           epsilon,
                                           activ_mode,
                                           activ_alpha,
                                           activ_beta,
                                           activ_gamma);*/
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runGPUConvActivInference()
{
    miopenError = miopenStatusSuccess;
    double activ_alpha, activ_beta, activ_gamma;
    miopenActivationMode_t activ_mode;
    miopenGetActivationDescriptor(activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  convDesc,
                                  // DLOWELL Hardcoded. This assumes immediate mode. Needs GetAlgo.
                                  miopenConvolutionFwdAlgoDirect,
                                  weightTensor);

    if(bias_mode)
    {
        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor);
    }

    miopenCreateOpActivationForward(fusePlanDesc, &activOp, activDesc);
    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev->GetMem());

    miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta);

    if(bias_mode)
    {
        miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev->GetMem());
    }

    miopenError = miopenIsFusionPlanValid(fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        if(bias_mode)
            std::cerr << "ConvBiasActivInference plan not supported." << std::endl;
        else
            std::cerr << "ConvActivInference plan not supported." << std::endl;
    }

    miopenExecuteFusionPlan(GetHandle(),
                            fusePlanDesc,
                            inputTensor,
                            in_dev->GetMem(),
                            outputTensor,
                            out_dev->GetMem(),
                            fusionArgs);

    /*    miopen::DirectConvActivInference(miopen::deref(GetHandle()),
                                         &alpha,
                                         miopen::deref(inputTensor),
                                         in_dev->GetMem(),
                                         miopen::deref(weightTensor),
                                         wei_dev->GetMem(),
                                         &beta,
                                         miopen::deref(outputTensor),
                                         out_dev->GetMem(),
                                         pad_h,
                                         pad_w,
                                         u,
                                         v,
                                         dilation_h,
                                         dilation_w,
                                         bias_mode,
                                         bias_mode != 0 ? b_dev->GetMem() : nullptr,
                                         activ_mode,
                                         activ_alpha,
                                         activ_beta,
                                         activ_gamma);*/
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

    miopenBNActiveNeuronFwdInferHost<Tgpu, Tref>(activ_mode,
                                                 activ_gamma,
                                                 activ_beta,
                                                 activ_alpha,
                                                 out.size(),
                                                 (fusion_mode > 2) ? conv_res_host.data()
                                                                   : bn_res_host.data(),
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
int CBAInferFusionDriver<Tgpu, Tref>::FindConvForward(
    int& ret_algo_count, int request_algo_count, std::vector<miopenConvAlgoPerf_t>& perf_results)
{
    return miopenFindConvolutionForwardAlgorithm(
        GetHandle(),
        inputTensor,
        in_dev->GetMem(),
        weightTensor,
        wei_dev->GetMem(),
        convDesc,
        outputTensor,
        conv_res_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem() : nullptr,
        (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetSize() : 0,
        (inflags.GetValueInt("search") == 1) ? true : false);
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
    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  convDesc,
                                  // DLOWELL Hardcoded. This assumes immediate mode. Needs GetAlgo.
                                  miopenConvolutionFwdAlgoDirect,
                                  weightTensor);

    miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasTensor);

    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev->GetMem());

    miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev->GetMem());

    miopenError = miopenIsFusionPlanValid(fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        std::cerr << "ConvBiasInference plan not supported." << std::endl;
    }
    miopenExecuteFusionPlan(GetHandle(),
                            fusePlanDesc,
                            inputTensor,
                            in_dev->GetMem(),
                            outputTensor,
                            out_dev->GetMem(),
                            fusionArgs);
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::RunForwardGPU()
{
    Timer t;
    double fulltime = 0.;
    auto iters      = inflags.GetValueInt("iter");
    float lowtime   = 100000000.0;
    float avgtime   = 0.;

    //(cbbna = 0, cbna = 1, bna = 2, cba = 3, ca = 4, cb = 5)
    assert(fusion_mode < 6 && fusion_mode >= 0);

    float time = 0.0;
    for(int i = 0; i < iters; i++)
    {
        START_TIME;

        switch(fusion_mode)
        {
        case 0:
        case 1: runGPUConvBatchNormActivInference(); break;
        case 3:
        case 4: runGPUConvActivInference(); break;
        case 2: runGPUBatchNormActivInference(); break;
        case 5: runGPUFusedConvBiasInference(); break;
        }
        // miopenGetKernelTime(GetHandle(), &time);
        // kl_time += time;

        if(inflags.GetValueStr("time") == "1")
        {
            time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            lowtime = (time < lowtime) ? time : lowtime;
            if(iters > 1 && i > 0)
                avgtime += time;
        }

        /*        runGPUConvBatchNormActivInference();
                miopenGetKernelTime(GetHandle(), &time);
                kl_time += time;
        */

        /*        runGPUConvBatchNormActivInference();
                miopenGetKernelTime(GetHandle(), &time);
                kl_time += time;*/

        /*        runGPUConvFwdInference();
                miopenGetKernelTime(GetHandle(), &time);
                kl_time += time;*/

        /*        runGPUConvBiasInference();
                miopenGetKernelTime(GetHandle(), &time);
                kl_time += time;*/

        /*        runGPUBatchNormActivInference();
                miopenGetKernelTime(GetHandle(), &time);
                kl_time += time;*/

        /*        runGPUBNFwdInference();
                miopenGetKernelTime(GetHandle(), &time);
                kl_time += time;*/

        /*        runGPUActivFwdInference();
                miopenGetKernelTime(GetHandle(), &time);
                kl_time += time;*/

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
    ConvForwardCPU<Tgpu, Tref>(
        in, conv_res_host, wei, b, bias_mode, convDesc, inputTensor, weightTensor, outputTensor);

    return;
}

template <typename Tgpu, typename Tref>
void CBAInferFusionDriver<Tgpu, Tref>::runCPUBNFwdInference()
{
    double epsilon = static_cast<double>(EPSILON);

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        miopenBNActiveBNPerActivFwdInferHost(
            inputTensor,    // outputTensor, // DLOWELL use output for splice test
            in_host.data(), // conv_res_host.data(), //DLOWELL use conv for splice test
            bn_res_host.data(),
            scale.data(),
            bias.data(),
            epsilon,
            runningMean.data(),
            runningVariance.data());
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        miopenBNActiveBNSpatialFwdInferHost(
            inputTensor,    // outputTensor, // DLOWELL use output for splice test
            in_host.data(), // conv_res_host.data(), //DLOWELL use conv for splice test
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
void CBAInferFusionDriver<Tgpu, Tref>::runGPUConvFwdInference()
{

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    Timer t;
    START_TIME;

    int pad_h      = inflags.GetValueInt("pad_h");
    int pad_w      = inflags.GetValueInt("pad_w");
    int u          = inflags.GetValueInt("conv_stride_0");
    int v          = inflags.GetValueInt("conv_stride_1");
    int dilation_h = inflags.GetValueInt("dilation_h");
    int dilation_w = inflags.GetValueInt("dilation_w");

#if 1
    miopen::DirectConvInference(miopen::deref(GetHandle()),
                                &alpha,
                                miopen::deref(inputTensor),
                                in_dev->GetMem(),
                                miopen::deref(weightTensor),
                                wei_dev->GetMem(),
                                &beta,
                                miopen::deref(outputTensor),
                                conv_res_dev->GetMem(),
                                pad_h,
                                pad_w,
                                u,
                                v,
                                dilation_h,
                                dilation_w,
                                bias_mode,
                                bias_mode != 0 ? b_dev->GetMem() : nullptr);

#else
    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results(request_algo_count);

    FindConvForward(ret_algo_count, request_algo_count, perf_results);

    miopenConvolutionForward(GetHandle(),
                             &alpha,
                             inputTensor,
                             in_dev->GetMem(),
                             weightTensor,
                             wei_dev->GetMem(),
                             convDesc,
                             perf_results[0].fwd_algo, // use the fastest algo
                             &beta,
                             outputTensor,
                             conv_res_dev->GetMem(),
                             (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem() : nullptr,
                             (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetSize() : 0);
#endif

    // conv_res_dev->FromGPU(GetStream(), conv_res.data());

    return;
}

template <typename Tgpu, typename Tref>
int CBAInferFusionDriver<Tgpu, Tref>::RunForwardCPU()
{
    runCPUConvFwdInference();
    if(fusion_mode < 3)
        runCPUBNFwdInference();
    if(fusion_mode != 5)
        runCPUActivFwdInference();

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
    RunForwardCPU();

    double allowedEps = std::numeric_limits<Tgpu>::epsilon() * 80;

    int match = 1;

    // match &= miopenInferVerify(conv_res.size(), conv_res_host.data(), conv_res.data(),
    // allowedEps);

    // match &= miopenInferVerify(bn_res.size(), bn_res_host.data(), bn_res.data(), allowedEps);

    match &= miopenInferVerify(out.size(), out_host.data(), out.data(), allowedEps);

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
