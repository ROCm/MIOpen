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
#include "../test/random.hpp"
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
#define RMSTOL_FP16 2e-3

#define MIO_DRIVER_BN_REFERENCE_COMPUTE_3D_AS_2D 1 // Resolves issue #1974

//#define BN_RUNFOR_PROFILER

template <typename TInput,
          typename Tref,
          typename TAcc       = TInput,
          typename TScaleBias = TInput,
          typename TOut       = TInput>
class BatchNormDriver : public Driver
{
public:
    BatchNormDriver() : Driver() { data_type = (sizeof(TInput) == 4) ? miopenFloat : miopenHalf; }

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

    // Helper function to check the Layout type short names
    bool ChkLayout_ShortName();
    // function to validate the Layout type parameters.
    // layout parameter value to std (NCHW/NHWC/NCDHW/NDHWC) values,
    // defined in MIOpen lib.
    void ValidateLayoutInputParameters(std::string layout_type);

    ~BatchNormDriver() override {}

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

    GpumemTensor<TInput> in;
    GpumemTensor<TInput> out; // for forward the output maches input type.
    tensor<Tref> out_ref;

    // forward
    GpumemTensor<TScaleBias> scale;
    GpumemTensor<TScaleBias> bias;

    // forward inference
    GpumemTensor<TAcc> estMean;
    GpumemTensor<TAcc> estVariance;

    GpumemTensor<TAcc> savedMean;
    tensor<Tref> savedMean_ref;

    // forward training
    GpumemTensor<TAcc> savedVariance;
    GpumemTensor<TAcc> runMean;
    GpumemTensor<TAcc> runVariance;
    // ref
    tensor<Tref> savedVariance_ref;
    tensor<Tref> runMean_ref;
    tensor<Tref> runVariance_ref;

    // backward needed different type for bwd.
    GpumemTensor<TOut> out_bwd;

    GpumemTensor<TScaleBias> bnScale;
    GpumemTensor<TAcc> dScale;
    GpumemTensor<TAcc> dBias;
    // savedMean declared above as TAcc as well
    GpumemTensor<TAcc> savedInvVar;
    GpumemTensor<TOut> dy;

    tensor<Tref> dBias_ref;
    tensor<Tref> dScale_ref;

    Tref maxval;

    miopenTensorLayout_t bn_layout;
};

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::GetandSetData()
{

    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    SetBNParametersFromCmdLineArgs();

    in.AllocOnHost(tensor<TInput>{bn_layout, in_len});
    for(size_t i = 0; i < in.GetVector().size(); i++)
    {
        in.GetVector()[i] = prng::gen_canonical<TInput>();
    }

    auto derivedBnDesc = miopen::TensorDescriptor{};
    miopen::DeriveBNTensorDescriptor(derivedBnDesc, in.GetTensor().desc, bn_mode);

    if(isFwdInfer || isFwdTrain)
    {
        out.AllocOnHost(tensor<TInput>{bn_layout, in_len});
        scale.AllocOnHost(tensor<TScaleBias>{bn_layout, derivedBnDesc.GetLengths()});
        bias.AllocOnHost(tensor<TScaleBias>{bn_layout, derivedBnDesc.GetLengths()});

        for(int i = 0; i < scale.GetVector().size(); i++)
        {
            scale.GetVector()[i] = prng::gen_canonical<TInput>();
            bias.GetVector()[i]  = prng::gen_canonical<TInput>();
        }
    }
    if(isFwdInfer)
    {
        estMean.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        estVariance.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});

        auto gen_value_emean = [](auto...) { return prng::gen_descreet_unsigned<TAcc>(1e-2, 100); };
        estMean.InitHostData(estMean.GetTensor().desc.GetElementSize(), true, gen_value_emean);
    }
    else if(isFwdTrain)
    {
        savedMean.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        savedVariance.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        runMean.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        runVariance.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});

        for(int i = 0; i < runVariance.GetVector().size(); i++)
        {
            runMean.GetVector()[i]     = prng::gen_canonical<TAcc>();
            runVariance.GetVector()[i] = prng::gen_canonical<TAcc>();
        }
    }
    else if(isBwd)
    {
        out_bwd.AllocOnHost(tensor<TOut>{bn_layout, in_len});

        bnScale.AllocOnHost(tensor<TScaleBias>{bn_layout, derivedBnDesc.GetLengths()});
        dy.AllocOnHost(tensor<TOut>{bn_layout, in_len});

        auto gen_var_bwd = [](auto...) {
            return static_cast<TOut>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };

        dy.InitHostData(dy.GetTensor().desc.GetElementSize(), true, gen_var_bwd);

        dScale.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        dBias.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        savedMean.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        savedInvVar.AllocOnHost(tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});

        auto gen_value = [](auto...) { return prng::gen_descreet_unsigned<TScaleBias>(1e-2, 100); };
        bnScale.InitHostData(bnScale.GetTensor().desc.GetElementSize(), true, gen_value);

        auto gen_in_var = [](auto...) {
            return static_cast<TAcc>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
        savedMean.InitHostData(savedMean.GetTensor().desc.GetElementSize(), true, gen_in_var);
        savedInvVar.InitHostData(savedInvVar.GetTensor().desc.GetElementSize(), true, gen_in_var);
    }
    else
    {
        std::cout << "\nUnknown batch norm state!\n";
        exit(EXIT_FAILURE);
    }

    return miopenStatusSuccess;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::AddCmdLineArgs()
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

    inflags.AddInputFlag(
        "layout", 'L', "", "Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)", "string", true);

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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
std::vector<int>
BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::GetInputTensorLengthsFromCmdLine()
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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
bool BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::ChkLayout_ShortName()
{
    // check for short name of layout type
    if(inflags.FindShortName("layout") == 'L')
    {
        // do noting
        // found valid short names
        return true;
    }
    else
    {
        std::cerr << "Error:Invalid Short Name for layout!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
void BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::ValidateLayoutInputParameters(
    std::string layout_value)
{
    if(!ChkLayout_ShortName())
    {
        std::cerr << "Invalid Layout Short Name = " << inflags.FindShortName("layout") << std::endl;
        exit(EXIT_FAILURE);
    }
    if((layout_value.compare("NCHW") != 0) && (layout_value.compare("NHWC") != 0) &&
       (layout_value.compare("NCDHW") != 0) && (layout_value.compare("NDHWC") != 0))
    {
        std::cerr << "Invalid Layout Parameter Value - " << layout_value << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::SetBNParametersFromCmdLineArgs()
{

    //    	double bnAlpha = inflags.GetValueDouble("alpha");
    //    	double bnBeta = inflags.GetValueDouble("beta");

    const std::string default_layout = isDepthSpecified ? "NCDHW" : "NCHW";

    // inflags value is empty, default value is used
    // if it is supplied via cmd line, check the value.
    if(inflags.GetValueStr("layout").empty())
    {
        inflags.SetValue("layout", default_layout);
    }
    else
    {
        std::string layoutValue = inflags.GetValueStr("layout");
        ValidateLayoutInputParameters(layoutValue);
        inflags.SetValue("layout", layoutValue);
    }

    std::string layout = inflags.GetValueStr("layout");

    if(layout == "NCHW")
    {
        bn_layout = miopenTensorNCHW;
    }
    else if(layout == "NHWC")
    {
        bn_layout = miopenTensorNHWC;
    }
    else if(layout == "NCDHW")
    {
        bn_layout = miopenTensorNCDHW;
    }
    else if(layout == "NDHWC")
    {
        bn_layout = miopenTensorNDHWC;
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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::AllocateBuffersAndCopy()
{
    status_t status = STATUS_SUCCESS;
    DEFINE_CONTEXT(ctx);
#if MIOPEN_BACKEND_OPENCL
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif
    status |= in.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&in.GetTensor().desc));

    if(isFwdInfer || isFwdTrain)
    {
        status |= out.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&out.GetTensor().desc));
        out_ref =
            tensor<Tref>{out.GetTensor().desc.GetLayout_t(), out.GetTensor().desc.GetLengths()};
        status |= scale.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&scale.GetTensor().desc));
        status |= bias.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&bias.GetTensor().desc));
    }
    if(isFwdInfer)
    {
        status |= estMean.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&estMean.GetTensor().desc));
        status |=
            estVariance.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&estVariance.GetTensor().desc));
    }
    if(isFwdTrain)
    {
        status |=
            savedMean.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&savedMean.GetTensor().desc));
        status |= savedVariance.AllocOnDeviceAndInit(
            q, ctx, GetTensorSize(&savedVariance.GetTensor().desc));
        status |= runMean.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&runMean.GetTensor().desc));
        status |=
            runVariance.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&runVariance.GetTensor().desc));

        savedMean_ref = tensor<Tref>{savedMean.GetTensor().desc.GetLayout_t(),
                                     savedMean.GetTensor().desc.GetLengths()};

        savedVariance_ref = tensor<Tref>{savedVariance.GetTensor().desc.GetLayout_t(),
                                         savedVariance.GetTensor().desc.GetLengths()};

        runMean_ref = tensor<Tref>{runMean.GetTensor().desc.GetLayout_t(),
                                   runMean.GetTensor().desc.GetLengths()};

        runVariance_ref = tensor<Tref>{runVariance.GetTensor().desc.GetLayout_t(),
                                       runVariance.GetTensor().desc.GetLengths()};
    }
    if(isBwd)
    {
        status |= out_bwd.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&out_bwd.GetTensor().desc));

        out_ref = tensor<Tref>{out_bwd.GetTensor().desc.GetLayout_t(),
                               out_bwd.GetTensor().desc.GetLengths()};

        status |= bnScale.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&bnScale.GetTensor().desc));
        status |= dy.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&dy.GetTensor().desc));

        status |= dScale.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&dScale.GetTensor().desc));
        status |= dBias.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&dBias.GetTensor().desc));
        status |=
            savedMean.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&savedMean.GetTensor().desc));
        status |=
            savedInvVar.AllocOnDeviceAndInit(q, ctx, GetTensorSize(&savedInvVar.GetTensor().desc));

        dScale_ref = tensor<Tref>{dScale.GetTensor().desc.GetLayout_t(),
                                  dScale.GetTensor().desc.GetLengths()};

        dBias_ref =
            tensor<Tref>{dBias.GetTensor().desc.GetLayout_t(), dBias.GetTensor().desc.GetLengths()};
    }

    for(size_t i = 0; i < runMean.GetVector().size(); ++i)
    {
        runMean_ref.data[i] = static_cast<Tref>(runMean.GetVector()[i]);
    }

    for(size_t i = 0; i < runVariance.GetVector().size(); ++i)
    {
        runVariance_ref.data[i] = static_cast<Tref>(runVariance.GetVector()[i]);
    }

    if(status != STATUS_SUCCESS)
        printf("Fatal: Error copying data to GPU\nExiting...\n\n");

    return miopenStatusSuccess;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
void BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::runGPUFwdInference(Tref epsilon,
                                                                               float alpha,
                                                                               float beta)
{

    if(keepRunningMeanVar)
    { // use precalculated mean and variance
        miopenBatchNormalizationForwardInference_V2(GetHandle(),
                                                    bn_mode,
                                                    &alpha,
                                                    &beta,
                                                    &in.GetTensor().desc,
                                                    in.GetDevicePtr(),
                                                    &out.GetTensor().desc,
                                                    out.GetDevicePtr(),
                                                    &scale.GetTensor().desc,
                                                    &bias.GetTensor().desc,
                                                    &estMean.GetTensor().desc,
                                                    &estVariance.GetTensor().desc,
                                                    scale.GetDevicePtr(),
                                                    bias.GetDevicePtr(),
                                                    estMean.GetDevicePtr(),
                                                    estVariance.GetDevicePtr(),
                                                    epsilon);
    }
    else
    { // recalculate mean and variance
        miopenBatchNormalizationForwardInference_V2(GetHandle(),
                                                    bn_mode,
                                                    &alpha,
                                                    &beta,
                                                    &in.GetTensor().desc,
                                                    in.GetDevicePtr(),
                                                    &out.GetTensor().desc,
                                                    out.GetDevicePtr(),
                                                    &scale.GetTensor().desc,
                                                    &bias.GetTensor().desc,
                                                    &estMean.GetTensor().desc,
                                                    &estVariance.GetTensor().desc,
                                                    scale.GetDevicePtr(),
                                                    bias.GetDevicePtr(),
                                                    nullptr,
                                                    nullptr,
                                                    epsilon);
    }

    return;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
void BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::runGPUFwdTrain(Tref epsilon,
                                                                           Tref eAF,
                                                                           float alpha,
                                                                           float beta)
{
    if(saveMeanVar && keepRunningMeanVar)
    {
        miopenBatchNormalizationForwardTraining_V2(GetHandle(),
                                                   bn_mode,
                                                   &alpha,
                                                   &beta,
                                                   &in.GetTensor().desc,
                                                   in.GetDevicePtr(),
                                                   &out.GetTensor().desc,
                                                   out.GetDevicePtr(),
                                                   &scale.GetTensor().desc,
                                                   &bias.GetTensor().desc,
                                                   &savedMean.GetTensor().desc,
                                                   &savedVariance.GetTensor().desc,
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
        miopenBatchNormalizationForwardTraining_V2(GetHandle(),
                                                   bn_mode,
                                                   &alpha,
                                                   &beta,
                                                   &in.GetTensor().desc,
                                                   in.GetDevicePtr(),
                                                   &out.GetTensor().desc,
                                                   out.GetDevicePtr(),
                                                   &scale.GetTensor().desc,
                                                   &bias.GetTensor().desc,
                                                   &savedMean.GetTensor().desc,
                                                   &savedVariance.GetTensor().desc,
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
        miopenBatchNormalizationForwardTraining_V2(GetHandle(),
                                                   bn_mode,
                                                   &alpha,
                                                   &beta,
                                                   &in.GetTensor().desc,
                                                   in.GetDevicePtr(),
                                                   &out.GetTensor().desc,
                                                   out.GetDevicePtr(),
                                                   &scale.GetTensor().desc,
                                                   &bias.GetTensor().desc,
                                                   &savedMean.GetTensor().desc,
                                                   &savedVariance.GetTensor().desc,
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
        miopenBatchNormalizationForwardTraining_V2(GetHandle(),
                                                   bn_mode,
                                                   &alpha,
                                                   &beta,
                                                   &in.GetTensor().desc,
                                                   in.GetDevicePtr(),
                                                   &out.GetTensor().desc,
                                                   out.GetDevicePtr(),
                                                   &scale.GetTensor().desc,
                                                   &bias.GetTensor().desc,
                                                   &savedMean.GetTensor().desc,
                                                   &savedVariance.GetTensor().desc,
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
    miopenBatchNormalizationForwardTraining_V2(GetHandle(),
                                               bn_mode,
                                               &alpha,
                                               &beta,
                                               &in.GetTensor().desc,
                                               in.GetDevicePtr(),
                                               &out.GetTensor().desc,
                                               out.GetDevicePtr(),
                                               &scale.GetTensor().desc,
                                               &bias.GetTensor().desc,
                                               &savedMean.GetTensor().desc,
                                               &savedVariance.GetTensor().desc,
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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::RunForwardGPU()
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
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(in.GetTensor().desc.GetLengths());
        size_t M                         = in_n * in_c * in_h * in_w;
        size_t dataSz = (M + 2 * in_c) * miopen::GetTypeSize(in.GetTensor().desc.GetType());
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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
void BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::runCPUFwdInference(Tref epsilon)
{
    int size{0};
    miopenGetTensorDescriptorSize(&in.GetTensor().desc, &size);

    if(size == 5)
    {
        in.GetTensor().desc    = miopen::BuildReshaped4DTensorDescriptor(in.GetTensor().desc);
        out_ref.desc           = miopen::BuildReshaped4DTensorDescriptor(out_ref.desc);
        scale.GetTensor().desc = miopen::BuildReshaped4DTensorDescriptor(scale.GetTensor().desc);
        bias.GetTensor().desc  = miopen::BuildReshaped4DTensorDescriptor(bias.GetTensor().desc);
        estMean.GetTensor().desc =
            miopen::BuildReshaped4DTensorDescriptor(estMean.GetTensor().desc);
        estVariance.GetTensor().desc =
            miopen::BuildReshaped4DTensorDescriptor(estVariance.GetTensor().desc);
    }

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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
void BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::runCPUFwdTrain(Tref epsilon, Tref eAF)
{
    int size{0};
    miopenGetTensorDescriptorSize(&in.GetTensor().desc, &size);
    if(size == 5)
    {
        in.GetTensor().desc    = miopen::BuildReshaped4DTensorDescriptor(in.GetTensor().desc);
        out_ref.desc           = miopen::BuildReshaped4DTensorDescriptor(out_ref.desc);
        scale.GetTensor().desc = miopen::BuildReshaped4DTensorDescriptor(scale.GetTensor().desc);
        bias.GetTensor().desc  = miopen::BuildReshaped4DTensorDescriptor(bias.GetTensor().desc);
        savedMean_ref.desc     = miopen::BuildReshaped4DTensorDescriptor(savedMean_ref.desc);
        savedVariance_ref.desc = miopen::BuildReshaped4DTensorDescriptor(savedVariance_ref.desc);
        runMean_ref.desc       = miopen::BuildReshaped4DTensorDescriptor(runMean_ref.desc);
        runVariance_ref.desc   = miopen::BuildReshaped4DTensorDescriptor(runVariance_ref.desc);
    }
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

        if(forw == 2 && !keepRunningMeanVar)
        {
            tensor<Tref> empty_tensor;
            batchNormSpatialHostFwdTrain(in.GetTensor(),
                                         out_ref,
                                         scale.GetTensor(),
                                         bias.GetTensor(),
                                         static_cast<double>(epsilon),
                                         static_cast<double>(eAF),
                                         empty_tensor,  // savedMean_ref
                                         empty_tensor,  // savedVariance_ref
                                         empty_tensor,  // runMean_ref
                                         empty_tensor); // runVariance_ref
        }
        else
        {
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
    }
    else
    {
        printf("Something went wrong.\nBad batch normalization mode in host kernel "
               "selection.\nExiting...\n\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::RunForwardCPU()
{
    //	T alpha = 0., beta  = 0.;
    Tref epsilon = static_cast<Tref>(EPSILON);
    Tref eAF     = static_cast<Tref>(1.0);

    if(forw == 1 || (forw == 2 && !keepRunningMeanVar))
    { // training only
        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            eAF = static_cast<Tref>(1.0) / (static_cast<Tref>(i) + static_cast<Tref>(1.0));
            runCPUFwdTrain(epsilon, eAF /* alpha, beta,*/);
        }
    }
    else if(forw == 2 && keepRunningMeanVar)
    {
        // inference only
        runCPUFwdInference(epsilon);
    }
    else
    {
        printf("Unsupported forward cpu run state.\nExiting...\n\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    return miopenStatusSuccess;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::RunBackwardGPU()
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
            miopenBatchNormalizationBackward_V2(GetHandle(),
                                                bn_mode,
                                                &alphaDataDiff,
                                                &betaDataDiff,
                                                &alphaParamDiff,
                                                &betaParamDiff,
                                                &in.GetTensor().desc,
                                                in.GetDevicePtr(),
                                                &dy.GetTensor().desc,
                                                dy.GetDevicePtr(),
                                                &out_bwd.GetTensor().desc,
                                                out_bwd.GetDevicePtr(),
                                                &bnScale.GetTensor().desc,
                                                &dBias.GetTensor().desc,
                                                &savedMean.GetTensor().desc,
                                                &savedInvVar.GetTensor().desc,
                                                bnScale.GetDevicePtr(),
                                                dScale.GetDevicePtr(),
                                                dBias.GetDevicePtr(),
                                                epsilon,
                                                savedMean.GetDevicePtr(),
                                                savedInvVar.GetDevicePtr());
        }
        else
        {
            miopenBatchNormalizationBackward_V2(GetHandle(),
                                                bn_mode,
                                                &alphaDataDiff,
                                                &betaDataDiff,
                                                &alphaParamDiff,
                                                &betaParamDiff,
                                                &in.GetTensor().desc,
                                                in.GetDevicePtr(),
                                                &dy.GetTensor().desc,
                                                dy.GetDevicePtr(),
                                                &out_bwd.GetTensor().desc,
                                                out_bwd.GetDevicePtr(),
                                                &bnScale.GetTensor().desc,
                                                &dBias.GetTensor().desc,
                                                &savedMean.GetTensor().desc,
                                                &savedInvVar.GetTensor().desc,
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
            std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(in.GetTensor().desc.GetLengths());
            size_t M                         = in_n * in_c * in_h * in_w;
            size_t dataSz = (M + 2 * in_c) * miopen::GetTypeSize(in.GetTensor().desc.GetType());
            float rdCnt   = 2.0;
            float wrCnt   = 1.0;
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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::VerifyForward()
{

    // jump out since we are forcing forward off when doing backwards.
    if(!forw)
        return miopenStatusSuccess;

    const Tref maxrms = static_cast<Tref>((sizeof(TInput) == 4) ? RMSTOL_FP32 : RMSTOL_FP16);

#if(MIO_BN_DEBUG == 1)
    const Tref tolerance = static_cast<Tref>((sizeof(TInput) == 4) ? ERRTOL_FP32 : ERRTOL_FP16);
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
                    diff = fabs(TAcc(fabs(runMean.GetVector()[i]) - fabs(runMean_ref.data[i])));
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "rm[" << i << "]: " << runMean.GetVector()[i];
                        std::cout << ", rm_host[" << i << "]: " << runMean_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << TAcc(fabs(runMean.GetVector()[i]) - fabs(runMean_ref.data[i]))
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
                        TAcc(fabs(runVariance.GetVector()[i]) - fabs(runVariance_ref.data[i])));
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "rv[" << i << "]: " << runVariance.GetVector()[i];
                        std::cout << ", rv_host[" << i << "]: " << runVariance_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << TAcc(fabs(runVariance.GetVector()[i]) -
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
            savedMean.CopyFromDeviceToHost(GetStream());
            savedVariance.CopyFromDeviceToHost(GetStream());
            maxval             = static_cast<Tref>(0.0);
            auto errorSaveMean = miopen::rms_range(savedMean_ref.data, savedMean.GetVector());
            if(!std::isfinite(errorSaveMean) || errorSaveMean > maxrms)
            {
                std::cout << "Forward train batch norm verification FAILED on saved mean: "
                          << errorSaveMean << std::endl;
                anError = true;
#if(MIO_BN_DEBUG == 1)
                for(int i = 0; i < savedMean.GetVector().size() && i < savedMean_ref.data.size() &&
                               i < MIO_BN_MAX_DEBUGLOOP;
                    i++)
                {
                    diff = fabs(TAcc(fabs(savedMean.GetVector()[i]) - fabs(savedMean_ref.data[i])));
                    maxval = maxval < diff ? diff : maxval;
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "sm[" << i << "]: " << savedMean.GetVector()[i];
                        std::cout << ", sm_host[" << i << "]: " << savedMean_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << TAcc(fabs(savedMean.GetVector()[i]) -
                                          fabs(savedMean_ref.data[i]))
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
                        TAcc(fabs(savedVariance.GetVector()[i]) - fabs(savedVariance_ref.data[i])));
                    if(!std::isfinite(diff) || diff > tolerance)
                    {
                        std::cout << "sv[" << i << "]: " << savedVariance.GetVector()[i];
                        std::cout << ", sv_host[" << i << "]: " << savedVariance_ref.data[i];
                        std::cout << ", diff[" << i << "]: "
                                  << TAcc(fabs(savedVariance.GetVector()[i]) -
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

    out.CopyFromDeviceToHost(GetStream());

    maxval = static_cast<Tref>(0.0);

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

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::RunBackwardCPU()
{

    if(!back)
        return miopenStatusSuccess;

    //	T alphaDiff = 1, betaDiff = 0;
    //	T alphaParam = 1, betaParam = 0;
    double alpha = static_cast<double>(1), beta = static_cast<double>(0),
           gamma = static_cast<double>(1);

    // float alphaDataDiff = static_cast<float>(1), betaDataDiff = static_cast<float>(0);
    // float alphaParamDiff = static_cast<float>(1), betaParamDiff = static_cast<float>(0);
    int size{0};
    miopenGetTensorDescriptorSize(&in.GetTensor().desc, &size);
    if(size == 5)
    {
        in.GetTensor().desc = miopen::BuildReshaped4DTensorDescriptor(in.GetTensor().desc);
        dy.GetTensor().desc = miopen::BuildReshaped4DTensorDescriptor(dy.GetTensor().desc);
        out_bwd.GetTensor().desc =
            miopen::BuildReshaped4DTensorDescriptor(out_bwd.GetTensor().desc);
        out_ref.desc = miopen::BuildReshaped4DTensorDescriptor(out_ref.desc);
        bnScale.GetTensor().desc =
            miopen::BuildReshaped4DTensorDescriptor(bnScale.GetTensor().desc);
        dBias.GetTensor().desc = miopen::BuildReshaped4DTensorDescriptor(dBias.GetTensor().desc);
        dScale_ref.desc        = miopen::BuildReshaped4DTensorDescriptor(dScale_ref.desc);
        dBias_ref.desc         = miopen::BuildReshaped4DTensorDescriptor(dBias_ref.desc);
        savedMean.GetTensor().desc =
            miopen::BuildReshaped4DTensorDescriptor(savedMean.GetTensor().desc);
        savedInvVar.GetTensor().desc =
            miopen::BuildReshaped4DTensorDescriptor(savedInvVar.GetTensor().desc);
    }

    if(bn_mode == miopenBNPerActivation)
    {
        // 1xCxHxW
        batchNormActivSpatialHostBwdTrain(activ_mode,
                                          gamma,
                                          beta,
                                          alpha,
                                          in.GetTensor(),
                                          dy.GetTensor(),
                                          out.GetTensor(),
                                          out_ref,
                                          bnScale.GetTensor(),
                                          dBias.GetTensor(),
                                          dScale_ref,
                                          dBias_ref,
                                          savedMean.GetTensor(),
                                          savedInvVar.GetTensor());
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        if(saveMeanVar)
        {

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
            tensor<Tref> empty_tensor;
            batchNormSpatialHostBwdTrain(in.GetTensor(),
                                         dy.GetTensor(),
                                         out_ref,
                                         bnScale.GetTensor(),
                                         dScale_ref,
                                         dBias_ref,
                                         empty_tensor,
                                         empty_tensor);
        }
    }
    else
    {
        printf("Something went wrong.\nBad batch normalization mode in host kernel "
               "selection.\nExiting...\n\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    return miopenStatusSuccess;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>::VerifyBackward()
{

    if(!back)
        return miopenStatusSuccess;

    const Tref maxrms =
        static_cast<Tref>(((sizeof(TInput) == 4) ? RMSTOL_FP32 : RMSTOL_FP16) * 1000);
    bool anError = false;

    RunBackwardCPU();

    out_bwd.CopyFromDeviceToHost(GetStream());
    dScale.CopyFromDeviceToHost(GetStream());
    dBias.CopyFromDeviceToHost(GetStream());

#if(MIO_BN_DEBUG == 1)
    const Tref tolerance =
        static_cast<Tref>(1000 * (sizeof(TInput) == 4) ? ERRTOL_FP32 : ERRTOL_FP16);
    Tref diff = static_cast<Tref>(0.0);
#endif
    maxval          = static_cast<Tref>(0.0);
    auto errordxout = miopen::rms_range(out_ref.data, out_bwd.GetVector());
    if(!std::isfinite(errordxout) || errordxout > maxrms)
    {
        std::cout << "Backwards prop batch norm verification FAILED on dx: " << errordxout
                  << std::endl;
        anError = true;
#if(MIO_BN_DEBUG == 1)
        for(int i = 0; i < out_ref.data.size() && i < MIO_BN_MAX_DEBUGLOOP; i++)
        {
            diff   = fabs(TOut(fabs(out_ref.data[i]) - fabs(out_bwd.GetVector()[i])));
            maxval = maxval < diff ? diff : maxval;
            if(!std::isfinite(diff) || diff > tolerance)
            {
                std::cout << "out_ref[" << i << "]: " << out_ref.data[i];
                std::cout << "\tout_bwd.GetVector()[" << i << "]: " << out_bwd.GetVector()[i];
                std::cout << "\tdiff[" << i
                          << "]: " << TOut(fabs(out_ref.data[i]) - fabs(out_bwd.GetVector()[i]));
                std::cout << "\tratioH: "
                          << fabs(fabs(out_ref.data[i]) - fabs(out_bwd.GetVector()[i])) /
                                 fabs(out_bwd.GetVector()[i])
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
            auto diff = fabs(TAcc(fabs(dScale.GetVector()[i]) - fabs(dScale_ref.data[i])));
            maxval    = maxval < diff ? diff : maxval;
            if(!std::isfinite(diff) || diff > tolerance)
            {
                std::cout << "dscale[" << i << "]: " << dScale.GetVector()[i];
                std::cout << "\tdscale_host[" << i << "]: " << dScale_ref.data[i];
                std::cout << "\tdiff[" << i
                          << "]: " << TAcc(fabs(dScale.GetVector()[i]) - fabs(dScale_ref.data[i]));
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
            diff = fabs(TAcc(fabs(dBias.GetVector()[i]) - fabs(dBias_ref.data[i])));
            if(!std::isfinite(diff) || diff > tolerance)
            {
                std::cout << "dbias[" << i << "]: " << dBias.GetVector()[i];
                std::cout << "\tdbias_host[" << i << "]: " << dBias_ref.data[i];
                std::cout << "\tdiff[" << i
                          << "]: " << TAcc(fabs(dBias.GetVector()[i]) - fabs(dBias_ref.data[i]));
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
