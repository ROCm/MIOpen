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

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen_BatchNormHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include "rocrand_wrapper.hpp"

#include "../test/verify.hpp"
#include "../test/fusionHost.hpp"

#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include "miopen/batch_norm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <numeric>
#include <vector>

#define MIO_BN_DEBUG 0
#define MIO_BN_MAX_DEBUGLOOP 65536

#define EPSILON 1e-3

#define ERRTOL_FP32 1e-4
#define ERRTOL_FP16 0.5e-3
#define RMSTOL_FP32 1e-4
#define RMSTOL_FP16 0.5e-3

#define MIO_DRIVER_BN_REFERENCE_COMPUTE_3D_AS_2D 1 // Resolves issue #1974

//========================

template <typename Tgpu>
class GpumemTensor
{
    std::unique_ptr<GPUMem> dev;
    tensor<Tgpu> host;
    bool is_gpualloc = false;

public:
    void SetGpuallocMode(bool v) { is_gpualloc = v; }
    tensor<Tgpu>& GetTensor() { return host; }

    void AllocOnHost(miopenTensorDescriptor_t t)
    {
        host = tensor<Tgpu>(miopen::deref(t));
        if(is_gpualloc) // We do not need host data.
        {
            host.data.clear();
            host.data.shrink_to_fit(); // To free host memory.
        }
    }
    template <typename T>
    void AllocOnHost(tensor<T> t)
    {
        AllocOnHost(&t.desc);
    }

    std::vector<Tgpu>& GetVector()
    {
        if(is_gpualloc)
            MIOPEN_THROW("[MIOpenDriver] GpumemTensor::GetVector should not be called in "
                         "'--gpualloc 1' mode");
        return host.data;
    }

    Tgpu* GetVectorData() { return is_gpualloc ? nullptr : host.data.data(); }
    std::size_t GetVectorSize() const { return is_gpualloc ? 0 : host.data.size(); }

    void
    InitHostData(const size_t sz,     //
                 const bool do_write, // If set to false, then only generate random data. This is
                                      // necessary to reproduce values in input buffers even if some
                                      // directions are skipped. For example, inputs for Backward
                                      // will be the same for both "-F 0" and "-F 2".
                 std::function<Tgpu()> generator)
    {
        if(is_gpualloc)
        {
            /// In gpualloc mode, we do not care about reproducibility of results, because
            /// validation is not used. Therefore, we do not have to always generate random value
            /// (\ref move_rand)
            return;
        }

        for(size_t i = 0; i < sz; ++i)
        {
            /// \anchor move_rand
            /// Generate random value, even if buffer is unused. This provides the same
            /// initialization of input buffers regardless of which kinds of
            /// convolutions are currently selectedfor testing (see the "-F" option).
            /// Verification cache would be broken otherwise.
            auto val = generator();
            if(do_write)
                GetVector()[i] = val;
        }
    }

    status_t AllocOnDevice(stream, context_t ctx, const size_t sz)
    {
        dev = std::make_unique<GPUMem>(ctx, sz, sizeof(Tgpu));
        return STATUS_SUCCESS;
    }

    status_t AllocOnDeviceAndInit(stream q, context_t ctx, const size_t sz)
    {
        AllocOnDevice(q, ctx, sz);
        if(is_gpualloc)
        {
            /// \anchor gpualloc_random_init
            /// In gpualloc mode, we do not want to leave input buffers uninitialized, because
            /// there could be NaNs and Infs, which may affect the performance (which we are
            /// interested to evaluate in this mode). Initialization with all 0's is not the
            /// best choice as well, because GPU HW may optimize out computations with 0's and
            /// that could affect performance of kernels too. That is why we are using
            /// rocrand to initialize input buffers.
            ///
            /// However we do not care about precision in gpualloc mode, because validation
            /// is not used. Therefore, range (0,1] is fine.
            return gpumemrand::gen_0_1(static_cast<Tgpu*>(GetDevicePtr()), sz);
        }
        return dev->ToGPU(q, GetVectorData());
    }

    template <typename T>
    status_t AllocOnDevice(stream, context_t ctx, const size_t sz, std::vector<T>&)
    {
        static_assert(std::is_same<T, float>::value           //
                          || std::is_same<T, int32_t>::value, //
                      "Before enabling more types, check thoroughly.");
        dev = std::make_unique<GPUMem>(ctx, sz, sizeof(T));
        return STATUS_SUCCESS;
    }

    template <typename T>
    status_t AllocOnDeviceAndInit(stream q, context_t ctx, const size_t sz, std::vector<T>& init)
    {
        AllocOnDevice(q, ctx, sz, init);
        if(is_gpualloc)
        {
            /// \ref gpualloc_random_init
            return gpumemrand::gen_0_1(static_cast<Tgpu*>(GetDevicePtr()), sz);
        }
        return dev->ToGPU(q, init.data());
    }

    status_t CopyFromDeviceToHost(stream q)
    {
        return is_gpualloc ? STATUS_SUCCESS : dev->FromGPU(q, GetVectorData());
    }

    template <typename T>
    status_t CopyFromDeviceToHost(stream q, tensor<T>& t)
    {
        return is_gpualloc ? STATUS_SUCCESS : dev->FromGPU(q, t.data.data());
    }

    template <typename T>
    status_t CopyFromDeviceToHost(stream q, std::vector<T>& v)
    {
        return is_gpualloc ? STATUS_SUCCESS : dev->FromGPU(q, v.data());
    }

    auto GetDevicePtr() -> auto { return dev->GetMem(); }
};
//========================

//#define BN_RUNFOR_PROFILER

template <typename Tgpu, typename Tref, typename Tmix = Tgpu>
class BatchNormDriver : public Driver
{
public:
    BatchNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        // miopenCreateTensorDescriptor(&biasScaleTensor);
        // miopenCreateTensorDescriptor(&dxOutputTensor);
        // miopenCreateTensorDescriptor(&dyInputTensor);

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

    void runCPUFwdInference(Tref epsilon);
    void runCPUFwdTrain(Tref epsilon, Tref eAF);

    int VerifyBackward() override;
    int VerifyForward() override;

    ~BatchNormDriver() override
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        // miopenDestroyTensorDescriptor(biasScaleTensor);
        // miopenDestroyTensorDescriptor(dxOutputTensor);
        // miopenDestroyTensorDescriptor(dyInputTensor);
    }

private:
    miopenBatchNormMode_t bn_mode;
    miopenActivationMode_t activ_mode = miopenActivationRELU;

    bool saveMeanVar;
    bool bsaveMeanVar;
    bool keepRunningMeanVar;
    bool estimatedMeanVar;

    int forw;
    int back;

    bool isFwdInfer = false;
    bool isFwdTrain = false;
    bool isBwd      = false;

    InputFlags inflags;
    bool isDepthSpecified = false;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    GpumemTensor<Tgpu> in;
    GpumemTensor<Tgpu> out;
    tensor<Tgpu> out_ref;

    // forward
    GpumemTensor<Tgpu> scale;
    GpumemTensor<Tgpu> bias;

    // forward inference
    GpumemTensor<Tmix> estMean;
    GpumemTensor<Tmix> estVariance;

    // forward training
    GpumemTensor<Tmix> savedMean;
    tensor<Tmix> savedMean_ref;
    GpumemTensor<Tmix> savedVariance;
    tensor<Tmix> savedVariance_ref;
    GpumemTensor<Tmix> runMean;
    tensor<Tmix> runMean_ref;
    GpumemTensor<Tmix> runVariance;
    tensor<Tmix> runVariance_ref;

    // backward
    GpumemTensor<Tgpu> bnScale;

    GpumemTensor<Tmix> dy;
    GpumemTensor<Tmix> dScale;
    tensor<Tmix> dScale_ref;
    GpumemTensor<Tmix> dBias;
    tensor<Tmix> dBias_ref;
    GpumemTensor<Tmix> savedInvVar;

    Tref maxval;

    miopenTensorLayout_t bn_layout;
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

    in.AllocOnHost(tensor<Tgpu>{bn_layout, in_len});
    out.AllocOnHost(tensor<Tgpu>{bn_layout, in_len});
    auto derivedBnDesc = miopen::TensorDescriptor{};
    miopen::DeriveBNTensorDescriptor(derivedBnDesc, in.GetTensor().desc, bn_mode);
    if(isFwdInfer || isFwdTrain)
    {
        scale.AllocOnHost(tensor<Tgpu>{bn_layout, derivedBnDesc.GetLengths()});
        bias.AllocOnHost(tensor<Tgpu>{bn_layout, derivedBnDesc.GetLengths()});
    }
    if(isFwdInfer)
    {
        estMean.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
        estVariance.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
    }
    else if(isFwdTrain)
    {
        savedMean.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
        savedVariance.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
        runMean.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
        runVariance.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
    }
    else if(isBwd)
    {
        bnScale.AllocOnHost(tensor<Tgpu>{bn_layout, derivedBnDesc.GetLengths()});
        dy.AllocOnHost(tensor<Tref>{bn_layout, in_len});

        dScale.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
        dBias.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
        savedMean.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
        savedInvVar.AllocOnHost(tensor<Tref>{bn_layout, derivedBnDesc.GetLengths()});
    }
    else
    {
        std::cout << "\nUnknown batch norm state!\n";
        exit(EXIT_FAILURE);
    }
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

    inflags.AddInputFlag("layout",
                         'L',
                         "NCHW",
                         "Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);

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

    std::string layout = inflags.GetValueStr("layout");

    if(layout == "NCHW")
    {
        bn_layout = miopenTensorNCHW;
    }
    else if(layout == "NHWC")
    {
        bn_layout = miopenTensorNHWC;
    }
    else
    {
        std::cout << "Cannot handle layout : " << layout << "\n";
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

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

    if(forw == 1)
    {
        isFwdTrain = true;
    }
    else if(forw == 2)
    {
        isFwdInfer = true;
    }
    else
    {
        isBwd = true;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tmix>
int BatchNormDriver<Tgpu, Tref, Tmix>::AllocateBuffersAndCopy()
{
    status_t status = STATUS_SUCCESS;
    DEFINE_CONTEXT(ctx);
#if MIOPEN_BACKEND_OPENCL
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif
    status |= in.AllocOnDeviceAndInit(q, ctx, in.GetTensor().desc.GetElementSpace());
    status |= out.AllocOnDeviceAndInit(q, ctx, out.GetTensor().desc.GetElementSpace());
    out_ref = out.GetTensor();
    if(isFwdInfer || isFwdTrain)
    {
        status |= scale.AllocOnDeviceAndInit(q, ctx, scale.GetTensor().desc.GetElementSpace());
        status |= bias.AllocOnDeviceAndInit(q, ctx, bias.GetTensor().desc.GetElementSpace());
    }
    if(isFwdInfer)
    {
        status |= estMean.AllocOnDeviceAndInit(q, ctx, estMean.GetTensor().desc.GetElementSpace());
        status |= estVariance.AllocOnDeviceAndInit(
            q, ctx, estVariance.GetTensor().desc.GetElementSpace());
    }
    if(isFwdTrain)
    {
        status |=
            savedMean.AllocOnDeviceAndInit(q, ctx, savedMean.GetTensor().desc.GetElementSpace());
        status |= savedVariance.AllocOnDeviceAndInit(
            q, ctx, savedVariance.GetTensor().desc.GetElementSpace());
        status |= runMean.AllocOnDeviceAndInit(q, ctx, runMean.GetTensor().desc.GetElementSpace());
        status |= runVariance.AllocOnDeviceAndInit(
            q, ctx, runVariance.GetTensor().desc.GetElementSpace());

        savedMean_ref     = savedMean.GetTensor();
        savedVariance_ref = savedVariance.GetTensor();
        runMean_ref       = runMean.GetTensor();
        runVariance_ref   = runVariance.GetTensor();
    }
    if(isBwd)
    {
        status |= bnScale.AllocOnDeviceAndInit(q, ctx, bnScale.GetTensor().desc.GetElementSpace());
        status |= dy.AllocOnDeviceAndInit(q, ctx, dy.GetTensor().desc.GetElementSpace());

        status |= dScale.AllocOnDeviceAndInit(q, ctx, dScale.GetTensor().desc.GetElementSpace());
        status |= dBias.AllocOnDeviceAndInit(q, ctx, dBias.GetTensor().desc.GetElementSpace());
        status |=
            savedMean.AllocOnDeviceAndInit(q, ctx, savedMean.GetTensor().desc.GetElementSpace());
        status |= savedInvVar.AllocOnDeviceAndInit(
            q, ctx, savedInvVar.GetTensor().desc.GetElementSpace());

        dScale_ref = dScale.GetTensor();
        dBias_ref  = dBias.GetTensor();
    }

    if(status != STATUS_SUCCESS)
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
                                                 &in.GetTensor().desc,
                                                 in.GetDevicePtr(),
                                                 &out.GetTensor().desc,
                                                 out.GetDevicePtr(),
                                                 &scale.GetTensor().desc,
                                                 scale.GetDevicePtr(),
                                                 bias.GetDevicePtr(),
                                                 estMean.GetDevicePtr(),
                                                 estVariance.GetDevicePtr(),
                                                 epsilon);
    }
    else
    { // recalculate mean and variance
        miopenBatchNormalizationForwardInference(GetHandle(),
                                                 bn_mode,
                                                 &alpha,
                                                 &beta,
                                                 &in.GetTensor().desc,
                                                 in.GetDevicePtr(),
                                                 &out.GetTensor().desc,
                                                 out.GetDevicePtr(),
                                                 &scale.GetTensor().desc,
                                                 scale.GetDevicePtr(),
                                                 bias.GetDevicePtr(),
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
                                                &in.GetTensor().desc,
                                                in.GetDevicePtr(),
                                                &out.GetTensor().desc,
                                                out.GetDevicePtr(),
                                                &scale.GetTensor().desc,
                                                scale.GetDevicePtr(),
                                                bias.GetDevicePtr(),
                                                eAF,
                                                runMean.GetDevicePtr(),
                                                runVariance.GetDevicePtr(),
                                                epsilon,
                                                savedMean.GetDevicePtr(),
                                                savedVariance.GetDevicePtr());
    }
    else if(saveMeanVar)
    {
        miopenBatchNormalizationForwardTraining(GetHandle(),
                                                bn_mode,
                                                &alpha,
                                                &beta,
                                                &in.GetTensor().desc,
                                                in.GetDevicePtr(),
                                                &out.GetTensor().desc,
                                                out.GetDevicePtr(),
                                                &scale.GetTensor().desc,
                                                scale.GetDevicePtr(),
                                                bias.GetDevicePtr(),
                                                eAF,
                                                nullptr,
                                                nullptr,
                                                epsilon,
                                                savedMean.GetDevicePtr(),
                                                savedVariance.GetDevicePtr());
    }
    else if(keepRunningMeanVar)
    {
        miopenBatchNormalizationForwardTraining(GetHandle(),
                                                bn_mode,
                                                &alpha,
                                                &beta,
                                                &in.GetTensor().desc,
                                                in.GetDevicePtr(),
                                                &out.GetTensor().desc,
                                                out.GetDevicePtr(),
                                                &scale.GetTensor().desc,
                                                scale.GetDevicePtr(),
                                                bias.GetDevicePtr(),
                                                eAF,
                                                runMean.GetDevicePtr(),
                                                runVariance.GetDevicePtr(),
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
                                                &in.GetTensor().desc,
                                                in.GetDevicePtr(),
                                                &out.GetTensor().desc,
                                                out.GetDevicePtr(),
                                                &scale.GetTensor().desc,
                                                scale.GetDevicePtr(),
                                                bias.GetDevicePtr(),
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
                                            &in.GetTensor().desc,
                                            in.GetDevicePtr(),
                                            &out.GetTensor().desc,
                                            out.GetDevicePtr(),
                                            &scale.GetTensor().desc,
                                            scale.GetDevicePtr(),
                                            bias.GetDevicePtr(),
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
void BatchNormDriver<Tgpu, Tref, Tmix>::runCPUFwdInference(Tref epsilon)
{

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        // handle 3d case
        batchNormPerActivHostInference(in.GetTensor(),
                                       out_ref,
                                       scale.GetTensor(),
                                       bias.GetTensor(),
                                       epsilon,
                                       estMean.GetTensor(),
                                       estVariance.GetTensor());
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        batchNormSpatialHostInference(in.GetTensor(),
                                      out_ref,
                                      scale.GetTensor(),
                                      bias.GetTensor(),
                                      epsilon,
                                      estMean.GetTensor(),
                                      estVariance.GetTensor());
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
void BatchNormDriver<Tgpu, Tref, Tmix>::runCPUFwdTrain(Tref epsilon, Tref eAF)
{

    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        batchNormPerActHostFwdTrain(in.GetTensor(),
                                    out_ref,
                                    scale.GetTensor(),
                                    bias.GetTensor(),
                                    static_cast<double>(epsilon),
                                    static_cast<double>(eAF),
                                    savedMean_ref,
                                    savedVariance_ref,
                                    runMean_ref,
                                    runVariance_ref);
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        batchNormSpatialHostFwdTrain(in.GetTensor(),
                                     out_ref,
                                     scale.GetTensor(),
                                     bias.GetTensor(),
                                     static_cast<double>(epsilon),
                                     static_cast<double>(eAF),
                                     savedMean_ref,
                                     savedVariance_ref,
                                     runMean_ref,
                                     runVariance_ref);
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
    //	T alpha = 0., beta  = 0.;
    Tref epsilon = static_cast<Tref>(EPSILON);
    Tref eAF     = static_cast<Tref>(1.0);

    if(forw == 1)
    { // training only
        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            eAF = static_cast<Tref>(1.0) / (static_cast<Tref>(i) + static_cast<Tref>(1.0));
            runCPUFwdTrain(epsilon, eAF /* alpha, beta,*/);
        }
    }
    else if(forw == 2)
    { // inference only
        runCPUFwdInference(epsilon);
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
                                             &in.GetTensor().desc,
                                             in.GetDevicePtr(),
                                             &dy.GetTensor().desc,
                                             dy.GetDevicePtr(),
                                             &out.GetTensor().desc,
                                             out.GetDevicePtr(),
                                             &bnScale.GetTensor().desc,
                                             bnScale.GetDevicePtr(),
                                             dScale.GetDevicePtr(),
                                             dBias.GetDevicePtr(),
                                             epsilon,
                                             savedMean.GetDevicePtr(),
                                             savedInvVar.GetDevicePtr());
        }
        else
        {
            miopenBatchNormalizationBackward(GetHandle(),
                                             bn_mode,
                                             &alphaDataDiff,
                                             &betaDataDiff,
                                             &alphaParamDiff,
                                             &betaParamDiff,
                                             &in.GetTensor().desc,
                                             in.GetDevicePtr(),
                                             &dy.GetTensor().desc,
                                             dy.GetDevicePtr(),
                                             &out.GetTensor().desc,
                                             out.GetDevicePtr(),
                                             &bnScale.GetTensor().desc,
                                             bnScale.GetDevicePtr(),
                                             dScale.GetDevicePtr(),
                                             dBias.GetDevicePtr(),
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
            runMean.CopyFromDeviceToHost(GetStream());
            runVariance.CopyFromDeviceToHost(GetStream());

            auto errorRunMean = miopen::rms_range(runMean_ref.data, runMean.GetVector());
            if(!std::isfinite(errorRunMean) || errorRunMean > maxrms)
            {
                std::cout << "Forward train batch norm verification FAILED on running mean: "
                          << errorRunMean << std::endl;
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < runMean.GetVector().size() && i < runMean_ref.data.size() &&
                               i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(Tmix(fabs(runMean.GetVector()[i]) - fabs(runMean_ref.data[i])));
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "rm[" << i << "]: " << runMean.GetVector()[i];
                        std::cout << ", rm_host[" << i << "]: " << runMean_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << Tmix(fabs(runMean.GetVector()[i]) - fabs(runMean_ref.data[i]))
                                  << std::endl;
                    }
                }
#endif
            }
            else
            {
                std::cout << "Forward train batch norm verification passed on running mean ("
                          << errorRunMean << ')' << std::endl;
            }

            auto errorRunVar = miopen::rms_range(runVariance_ref.data, runVariance.GetVector());
            if(!std::isfinite(errorRunVar) || errorRunVar > maxrms)
            {
                std::cout << "Forward train batch norm verification FAILED on running variance: "
                          << errorRunVar << std::endl;
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < runVariance.GetVector().size() &&
                               i < runVariance_ref.data.size() && i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(
                        Tmix(fabs(runVariance.GetVector()[i]) - fabs(runVariance_ref.data[i])));
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "rv[" << i << "]: " << runVariance.GetVector()[i];
                        std::cout << ", rv_host[" << i << "]: " << runVariance_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << Tmix(fabs(runVariance.GetVector()[i]) -
                                          fabs(runVariance_ref.data[i]))
                                  << std::endl;
                    }
                }
#endif
            }
            else
            {
                std::cout << "Forward train batch norm verification passed on running variance ("
                          << errorRunVar << ')' << std::endl;
            }
        } // end if(keepRunningMeanVar)

        if(saveMeanVar)
        { // copy back for verification
            // saveMean_dev->FromGPU(GetStream(), savedMean.data());
            // saveInvVariance_dev->FromGPU(GetStream(), savedInvVar.data());

            savedMean.CopyFromDeviceToHost(GetStream());
            savedVariance.CopyFromDeviceToHost(GetStream());

            maxval             = static_cast<Tref>(0.0);
            auto errorSaveMean = miopen::rms_range(savedVariance_ref.data, savedMean.GetVector());
            if(!std::isfinite(errorSaveMean) || errorSaveMean > maxrms)
            {
                std::cout << "Forward train batch norm verification FAILED on saved mean: "
                          << errorSaveMean << std::endl;
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < savedMean.GetVector().size() &&
                               i < savedVariance_ref.data.size() && i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(
                        Tmix(fabs(savedMean.GetVector()[i]) - fabs(savedVariance_ref.data[i])));
                    maxval = maxval < diff ? diff : maxval;
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "sm[" << i << "]: " << savedMean.GetVector()[i];
                        std::cout << ", sm_host[" << i << "]: " << savedVariance_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << Tmix(fabs(savedMean.GetVector()[i]) -
                                          fabs(savedVariance_ref.data[i]))
                                  << std::endl;
                    }
                }
#endif
                std::cout << "max difference in saved mean: " << maxval << std::endl;
            }
            else
            {
                std::cout << "Forward train batch norm verification passed on saved mean ("
                          << errorSaveMean << ')' << std::endl;
            }

            auto errorSaveVar =
                miopen::rms_range(savedVariance_ref.data, savedVariance.GetVector());
            if(!std::isfinite(errorSaveVar) || errorSaveVar > maxrms)
            {
                std::cout
                    << "Forward train batch norm verification FAILED on saved inverse variance: "
                    << errorSaveVar << std::endl;
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < savedVariance.GetVector().size() &&
                               i < savedVariance_ref.data.size() && i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(
                        Tmix(fabs(savedVariance.GetVector()[i]) - fabs(savedVariance_ref.data[i])));
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "sv[" << i << "]: " << savedVariance.GetVector()[i];
                        std::cout << ", sv_host[" << i << "]: " << savedVariance_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << Tmix(fabs(savedVariance.GetVector()[i]) -
                                          fabs(savedVariance_ref.data[i]))
                                  << std::endl;
                    }
                }
#endif
            }
            else
            {
                std::cout
                    << "Forward train batch norm verification passed on saved inverse variance ("
                    << errorSaveVar << ')' << std::endl;
            }
        } // end if(saveMeanVar)
    }

    // Check output tensor error
    // out_dev->FromGPU(GetStream(), out.data());
    out.CopyFromDeviceToHost(GetStream());

    maxval        = static_cast<Tref>(0.0);
    auto errorOut = miopen::rms_range(out_ref.data, out.GetVector());
    if(!std::isfinite(errorOut) || errorOut > maxrms)
    {
        std::cout << "Forward batch norm verification FAILED on output: " << errorOut << std::endl;
        anError = true;
#if(MIO_BN_DEBUG == 1)
        unsigned int count = 0;
        for(int i = 0; i < out.GetVector().size() && i < out_ref.data.size(); i++)
        {
            if(std::isnan(out.GetVector()[i]))
            {
                std::cout << "out[" << i << "] produced a nan: " << out.GetVector()[i] << std::endl;
            }
            if(std::isnan(out_ref.data[i]))
            {
                std::cout << "out_ref[" << i << "] produced a nan: " << out_ref.data[i]
                          << std::endl;
            }
            diff   = Tref(fabs(out.GetVector()[i]) - fabs(out_ref.data[i]));
            maxval = maxval < diff ? diff : maxval;
            if(!std::isfinite(diff) || diff > tolerance)
            {
                std::cout << "out[" << i << "]: " << out.GetVector()[i];
                std::cout << ", out_ref.data[" << i << "]: " << out_ref.data[i];
                std::cout << ", diff[" << i << "]: " << Tref(out.GetVector()[i] - out_ref.data[i])
                          << std::endl;
                count++;
            }
        }

        std::cout << "Number of elements: " << out.GetVector().size() << std::endl;
        std::cout << "Number of bad elements: " << count << std::endl;
        std::cout << "max difference in output: " << maxval << std::endl;
#endif
    }
    else
    {
        std::cout << "Forward batch norm verification passed on output (" << errorOut << ')'
                  << std::endl;
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

    //	T alphaDiff = 1, betaDiff = 0;
    //	T alphaParam = 1, betaParam = 0;
    double alpha = static_cast<double>(1), beta = static_cast<double>(0),
           gamma = static_cast<double>(1);

    if(bn_mode == miopenBNPerActivation)
    {
        // 1xCxHxW
        batchNormActivSpatialHostBwdTrain(activ_mode,
                                          gamma,
                                          beta,
                                          alpha,
                                          in.GetTensor(),
                                          out.GetTensor(),
                                          out_ref,
                                          bnScale.GetTensor(),
                                          dy.GetTensor(),
                                          dBias.GetTensor(),
                                          dScale_ref,
                                          dBias_ref,
                                          savedMean.GetTensor(),
                                          savedInvVar.GetTensor());
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        batchNormSpatialHostBwdTrain(in.GetTensor(),
                                     dy.GetTensor(),
                                     out_ref,
                                     bnScale.GetTensor(),
                                     dScale_ref,
                                     dBias_ref,
                                     savedMean.GetTensor(),
                                     savedInvVar.GetTensor());
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

    out.CopyFromDeviceToHost(GetStream());
    dScale.CopyFromDeviceToHost(GetStream());
    dBias.CopyFromDeviceToHost(GetStream());

#if(MIO_BN_DEBUG == 1)
    const Tref tolerance =
        static_cast<Tref>(1000 * (sizeof(Tgpu) == 4) ? ERRTOL_FP32 : ERRTOL_FP16);
    Tref diff = static_cast<Tref>(0.0);
#endif
    maxval          = static_cast<Tref>(0.0);
    auto errordxout = miopen::rms_range(out_ref.data, out.GetVector());
    if(!std::isfinite(errordxout) || errordxout > maxrms)
    {
        std::cout << "Backwards prop batch norm verification FAILED on dx: " << errordxout
                  << std::endl;
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < out_ref.data.size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff   = fabs(Tgpu(fabs(out_ref.data[i]) - fabs(out.GetVector()[i])));
            maxval = maxval < diff ? diff : maxval;
            if(!std::isfinite(diff) || diff > tolerance)
            {
                std::cout << "out_ref[" << i << "]: " << out_ref.data[i];
                std::cout << "\tout.GetVector()[" << i << "]: " << out.GetVector()[i];
                std::cout << "\tdiff[" << i
                          << "]: " << Tgpu(fabs(out_ref.data[i]) - fabs(out.GetVector()[i]));
                std::cout << "\tratioH: "
                          << fabs(fabs(out_ref.data[i]) - fabs(out.GetVector()[i])) /
                                 fabs(out.GetVector()[i])
                          << std::endl;
            }
        }
#endif
        std::cout << "max difference in dxout: " << maxval << std::endl;
    }
    else
    {
        std::cout << "Backwards prop batch norm verification passed on dx (" << errordxout << ')'
                  << std::endl;
    }

    maxval           = static_cast<Tref>(0.0);
    auto errordscale = miopen::rms_range(dScale_ref.data, dScale.GetVector());
    if(!std::isfinite(errordscale) || errordscale > maxrms)
    {
        std::cout << "Backwards prop batch norm verification FAILED on dscale: " << errordscale
                  << std::endl;
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < dScale.GetVector().size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff   = fabs(Tmix(fabs(dScale.GetVector()[i]) - fabs(dScale_ref.data[i])));
            maxval = maxval < diff ? diff : maxval;
            if(!std::isfinite(diff) || diff > tolerance)
            {
                std::cout << "dscale[" << i << "]: " << dScale.GetVector()[i];
                std::cout << "\tdscale_host[" << i << "]: " << dScale_ref.data[i];
                std::cout << "\tdiff[" << i
                          << "]: " << Tmix(fabs(dScale.GetVector()[i]) - fabs(dScale_ref.data[i]));
                std::cout << "\tratioH: "
                          << fabs(fabs(dScale.GetVector()[i]) - fabs(dScale_ref.data[i])) /
                                 fabs(dScale_ref.data[i])
                          << std::endl;
            }
        }
#endif
        std::cout << "max difference in dscale: " << maxval << std::endl;
    }
    else
    {
        std::cout << "Backwards prop batch norm verification passed on dscale (" << errordscale
                  << ')' << std::endl;
    }

    auto errordbias = miopen::rms_range(dBias_ref.data, dBias.GetVector());
    if(!std::isfinite(errordbias) || errordbias > maxrms)
    {
        std::cout << "Backwards prop batch norm verification FAILED on dbias: " << errordbias
                  << std::endl;
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < dBias.GetVector().size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff = fabs(Tmix(fabs(dBias.GetVector()[i]) - fabs(dBias_ref.data[i])));
            if(!std::isfinite(diff) || diff > tolerance)
            {
                std::cout << "dbias[" << i << "]: " << dBias.GetVector()[i];
                std::cout << "\tdbias_host[" << i << "]: " << dBias_ref.data[i];
                std::cout << "\tdiff[" << i
                          << "]: " << Tmix(fabs(dBias.GetVector()[i]) - fabs(dBias_ref.data[i]));
                std::cout << "\tratioH: "
                          << fabs(fabs(dBias.GetVector()[i]) - fabs(dBias_ref.data[i])) /
                                 fabs(dBias_ref.data[i])
                          << std::endl;
            }
        }
#endif
    }
    else
    {
        std::cout << "Backwards prop batch norm verification passed on dbias (" << errordbias << ')'
                  << std::endl;
    }

    if(!anError)
        std::cout << "Backwards Prop Batch Norm Verifies on CPU and GPU." << std::endl;

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BN_DRIVER_HPP
