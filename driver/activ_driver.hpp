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
#ifndef GUARD_MIOPEN_ACTIV_DRIVER_HPP
#define GUARD_MIOPEN_ACTIV_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloNeuronHost.hpp"
#include "tensor_driver.hpp"
#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include "random.hpp"
#include "timer.hpp"

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

template <typename Tgpu, typename Tref>
class ActivationDriver : public Driver
{
    public:
    ActivationDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);

        miopenCreateActivationDescriptor(&activDesc);
        data_type = (sizeof(Tgpu) == 4) ? miopenFloat : miopenHalf;
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetActivationDescriptorFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    int VerifyBackward() override;
    int VerifyForward() override;
    ~ActivationDriver() override
    {

        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyActivationDescriptor(activDesc);
    }

    private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> scale_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    miopenActivationDescriptor_t activDesc;

    miopenTensorDescriptor_t dInputTensor;
    miopenTensorDescriptor_t dOutputTensor;

    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> dout_dev;

    std::vector<Tgpu> din;
    std::vector<Tgpu> dout;
    std::vector<Tref> dinhost;
};

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len, data_type);

    SetActivationDescriptorFromCmdLineArgs();

    SetTensor4d(outputTensor, in_len, data_type);

    SetTensor4d(dInputTensor, in_len, data_type);
    SetTensor4d(dOutputTensor, in_len, data_type);
    return (0);
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward LRN Normalization (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "3", "Activation Mode (relu,..., see spec) (Default=3(relu))", "int");
    inflags.AddInputFlag("alpha", 'A', "1", "Activation alpha (Default=1)", "double");
    inflags.AddInputFlag("beta", 'B', "1", "Activation beta (Default=1)", "double");
    inflags.AddInputFlag("gamma", 'G', "1", "Activation gamma (Default=1)", "double");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> ActivationDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::SetActivationDescriptorFromCmdLineArgs()
{

    miopenActivationMode_t mode;
    double Alpha = inflags.GetValueDouble("alpha");
    double Beta  = inflags.GetValueDouble("beta");
    double Gamma = inflags.GetValueDouble("gamma");
    mode         = static_cast<miopenActivationMode_t>(inflags.GetValueInt("mode"));

    return (miopenSetActivationDescriptor(activDesc, mode, Alpha, Beta, Gamma));
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{

    size_t in_sz  = GetTensorSpace(inputTensor);
    size_t out_sz = GetTensorSpace(outputTensor);
#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    din     = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dout    = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    dinhost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

    miopenActivationMode_t activation_mode;
    double alpha, beta, gamma;

    miopenGetActivationDescriptor(activDesc, &activation_mode, &alpha, &beta, &gamma);

    for(int i = 0; i < in_sz; i++)
    {
        switch(activation_mode)
        {
        case MIOPEN_NEURON_PASTHRU:
            in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            break;
        case MIOPEN_NEURON_LOGISTIC:
            in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            break;
        case MIOPEN_NEURON_TANH:
            in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            break;
        case MIOPEN_NEURON_RELU:
            in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            break;
        case MIOPEN_NEURON_SOFTRELU:
            in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            break;
        case MIOPEN_NEURON_ABS:
            in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(2.0));
            break;
        case MIOPEN_NEURON_POWER: {
            double v = -alpha / beta;
            in[i]    = i % 2 ? RAN_GEN<Tgpu>(static_cast<Tgpu>((v + 0.005) / beta),
                                          static_cast<Tgpu>((v + 2.0) / beta))
                          : RAN_GEN<Tgpu>(static_cast<Tgpu>((v - 2.0) / beta),
                                          static_cast<Tgpu>((v - 0.005) / beta));
            break;
        }
        case MIOPEN_NEURON_CLIPPED_RELU:
            if(i % 3 == 0)
                in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-1.0 * alpha),
                                      static_cast<Tgpu>(-0.005 * alpha));
            else if(i % 3 == 1)
                in[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(0.005 * alpha),
                                      static_cast<Tgpu>(0.995 * alpha));
            else
                in[i] =
                    RAN_GEN<Tgpu>(static_cast<Tgpu>(1.005 * alpha), static_cast<Tgpu>(2.0 * alpha));

            break;
        case MIOPEN_NEURON_LEAKY_RELU:
            in[i] = i % 2 ? RAN_GEN<Tgpu>(static_cast<Tgpu>(-1.0), static_cast<Tgpu>(-0.005))
                          : RAN_GEN<Tgpu>(static_cast<Tgpu>(-0.005), static_cast<Tgpu>(1.0));
            break;
        case MIOPEN_NEURON_ELU:
            in[i] = i % 2 ? RAN_GEN<Tgpu>(static_cast<Tgpu>(0.005), static_cast<Tgpu>(2.0))
                          : RAN_GEN<Tgpu>(static_cast<Tgpu>(-2.0), static_cast<Tgpu>(-0.005));
            break;
        }
    }

    for(int i = 0; i < out_sz; i++)
    {
        dout[i] = RAN_GEN<Tgpu>(static_cast<Tgpu>(-0.5), static_cast<Tgpu>(0.5));
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

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::RunForwardGPU()
{

    float alpha = 1, beta = 0;
    double fulltime = 0.;
    float avgtime   = 0.0f;
    float lowtime   = 100000000.0f;
    int iters       = inflags.GetValueInt("iter");
    Timer t;

    for(int i = 0; i < iters; i++)
    {
        START_TIME

        miopenActivationForward(GetHandle(),
                                activDesc,
                                &alpha,
                                inputTensor,
                                in_dev->GetMem(),
                                &beta,
                                outputTensor,
                                out_dev->GetMem());

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

        if(inflags.GetValueInt("time") == 1)
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
        printf("Wall-clock Time Forward GPU Activation Elapsed: %f ms, for %d iterations.\n",
               (iters == 1) ? t.gettime_ms() : (fulltime / float(iters - 1)),
               (iters > 1) ? iters - 1 : 1);
    }

    if(inflags.GetValueInt("time") == 1)
    {
        printf("GPU Kernel Min Time Forward Activation Elapsed: %f ms\n", lowtime);
        if(iters > 1)
            printf("GPU Kernel Avg Time Forward Activation Elapsed: %f ms, for %d iterations.\n",
                   avgtime / (iters - 1),
                   iters - 1);
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
        size_t dataSz =
            in_n * in_c * in_h * in_w * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());

        // layer, readbytes, writebytes, BG/s, timeMS
        printf("stats: name, bytesRead, bytesWritten, GB/s, timeMs\n");
        printf("stats: fwd-activ, %zu, %zu, %f, %f\n",
               dataSz,
               dataSz,
               2 * dataSz / lowtime / 1e6,
               avgtime / (iters - 1));
    }

    out_dev->FromGPU(GetStream(), out.data());
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::RunForwardCPU()
{
    return (0);
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float alpha = 1, beta = 0;
    double fulltime = 0.;
    float avgtime   = 0.0f;
    float lowtime   = 100000000.0f;
    int iters       = inflags.GetValueInt("iter");
    Timer t;

    for(int i = 0; i < iters; i++)
    {
        START_TIME

        miopenActivationBackward(GetHandle(),
                                 activDesc,
                                 &alpha,
                                 outputTensor,
                                 out_dev->GetMem(),
                                 dOutputTensor,
                                 dout_dev->GetMem(),
                                 inputTensor,
                                 in_dev->GetMem(),
                                 &beta,
                                 dInputTensor,
                                 din_dev->GetMem());

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

        if(inflags.GetValueInt("time") == 1)
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
        printf("Wall-clock Time Backward GPU Activation Elapsed: %f ms, for %d iterations.\n",
               (iters == 1) ? t.gettime_ms() : (fulltime / float(iters - 1)),
               (iters > 1) ? iters - 1 : 1);
    }

    if(inflags.GetValueInt("time") == 1)
    {
        printf("GPU Kernel Min Time Backward Activation Elapsed: %f ms\n", lowtime);
        if(iters > 1)
            printf("GPU Kernel Avg Time Backward Activation Elapsed: %f ms, for %d iterations.\n",
                   avgtime / (iters - 1),
                   iters - 1);
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
        size_t dataSz =
            in_n * in_c * in_h * in_w * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());

        // layer, readbytes, writebytes, BG/s, timeMS
        printf("stats: name, bytesRead, bytesWritten, GB/s, timeMs\n");
        printf("stats: bwd-activ, %zu, %zu, %f, %f\n",
               dataSz,
               dataSz,
               2 * dataSz / lowtime / 1e6,
               avgtime / (iters - 1));
    }

    din_dev->FromGPU(GetStream(), din.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::VerifyForward()
{

    double allowedEps = std::numeric_limits<Tgpu>::epsilon() * 80;
    miopenActivationMode_t v_mode;
    double v_Alpha;
    double v_Beta;
    double v_Gamma;

    miopenGetActivationDescriptor(activDesc, &v_mode, &v_Alpha, &v_Beta, &v_Gamma);

    int match = 1;
    match     = mloNeuronForwardRunHostAndVerify<Tgpu, Tref>(v_mode,
                                                         static_cast<Tref>(v_Gamma),
                                                         static_cast<Tref>(v_Beta),
                                                         static_cast<Tref>(v_Alpha),
                                                         in.size(),
                                                         in.data(),
                                                         out.data(),
                                                         static_cast<Tref>(allowedEps));

    if(match)
        printf("Forward Activation Verifies on CPU and GPU\n");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::RunBackwardCPU()
{

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ActivationDriver<Tgpu, Tref>::VerifyBackward()
{

    double allowedEps = std::numeric_limits<Tgpu>::epsilon() * 80;
    miopenActivationMode_t v_mode;
    double v_Alpha;
    double v_Beta;
    double v_Gamma;

    miopenGetActivationDescriptor(activDesc, &v_mode, &v_Alpha, &v_Beta, &v_Gamma);

    int match = 1;
    match     = mloNeuronBackwardRunHostAndVerify<Tgpu, Tref>(v_mode,
                                                          static_cast<Tref>(v_Gamma),
                                                          static_cast<Tref>(v_Beta),
                                                          static_cast<Tref>(v_Alpha),
                                                          dinhost.size(),
                                                          in.data(),
                                                          out.data(),
                                                          din.data(),
                                                          dout.data(),
                                                          static_cast<Tref>(allowedEps));
    if(match)
        printf("Backward Activation Verifies on CPU and GPU\n");
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_ACTIV_DRIVER_HPP
