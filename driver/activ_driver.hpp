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

#ifdef MIOPEN_BACKEND_HIP
#ifndef CL_SUCCESS
#define CL_SUCCESS 0
#endif
#endif

template <typename T>
class ActivationDriver : public Driver
{
    public:
    ActivationDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateActivationDescriptor(&activDesc);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetActivationDescriptorFromCmdLineArgs();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU();
    int RunBackwardCPU();

    int VerifyBackward();
    int VerifyForward();
    ~ActivationDriver()
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

    std::vector<T> in;
    std::vector<T> out;
    std::vector<T> outhost;

    miopenActivationDescriptor_t activDesc;

    miopenTensorDescriptor_t dInputTensor;
    miopenTensorDescriptor_t dOutputTensor;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> dout_dev;

    std::vector<T> din;
    std::vector<T> dout;
    std::vector<T> dinhost;
};

template <typename T>
int ActivationDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename T>
int ActivationDriver<T>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len);

    SetActivationDescriptorFromCmdLineArgs();

    SetTensor4d(outputTensor, in_len);

    SetTensor4d(dInputTensor, in_len);
    SetTensor4d(dOutputTensor, in_len);
    return (0);
}

template <typename T>
int ActivationDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward LRN Normalization (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "3", "Activation Mode (relu,..., see spec) (Default=3(relu))", "int");
    inflags.AddInputFlag("alpha", 'A', "0.0", "Activation shift (Default=0.0)", "double");
    inflags.AddInputFlag("beta", 'B', "0.0", "Activation scale (Default=0.0)", "double");
    inflags.AddInputFlag("power", 'P', "1.0", "Activation power (Default=1.0)", "double");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");

    return 0;
}

template <typename T>
std::vector<int> ActivationDriver<T>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename T>
int ActivationDriver<T>::SetActivationDescriptorFromCmdLineArgs()
{

    miopenActivationMode_t mode;
    double Alpha = inflags.GetValueDouble("alpha");
    double Beta  = inflags.GetValueDouble("beta");
    double Power = inflags.GetValueDouble("power");
    mode         = static_cast<miopenActivationMode_t>(inflags.GetValueInt("mode"));

    miopenSetActivationDescriptor(activDesc, mode, Alpha, Beta, Power);
    return (0);
}

template <typename T>
int ActivationDriver<T>::AllocateBuffersAndCopy()
{

    size_t in_sz  = GetTensorSize(inputTensor);
    size_t out_sz = GetTensorSize(outputTensor);
#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

    in      = std::vector<float>(in_sz);
    out     = std::vector<float>(out_sz, 0);
    outhost = std::vector<float>(out_sz, 0);

    din     = std::vector<float>(in_sz);
    dout    = std::vector<float>(out_sz, 0);
    dinhost = std::vector<float>(in_sz, 0);

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = static_cast<T>(static_cast<double>(rand()) * (1.0 / RAND_MAX));
    }

    for(int i = 0; i < out_sz; i++)
    {
        dout[i] = static_cast<T>(static_cast<double>((rand()) * (1.0 / RAND_MAX) - 0.5) * 0.001);
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

template <typename T>
int ActivationDriver<T>::RunForwardGPU()
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

    return miopenStatusSuccess;
}

template <typename T>
int ActivationDriver<T>::RunForwardCPU()
{
    return (0);
}

template <typename T>
int ActivationDriver<T>::RunBackwardGPU()
{
    float alpha = 1., beta = 0.;

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

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        printf("GPU Kernel Time Backward Activation Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());
    return (0);
}

template <typename T>
int ActivationDriver<T>::VerifyForward()
{
    const double allowedEps = (1 << 2);
    miopenActivationMode_t v_mode;
    double v_Alpha;
    double v_Beta;
    double v_Power;

    miopenGetActivationDescriptor(activDesc, &v_mode, &v_Alpha, &v_Beta, &v_Power);

    int match = 1;
    match     = mloNeuronForwardRunHostAndVerify<T>(v_mode,
                                                static_cast<T>(v_Power),
                                                static_cast<T>(v_Alpha),
                                                static_cast<T>(v_Beta),
                                                in.size(),
                                                in.data(),
                                                out.data(),
                                                allowedEps);

    if(match)
        printf("Forward Activation Verifies on CPU and GPU\n");
    return 0;
}

template <typename T>
int ActivationDriver<T>::RunBackwardCPU()
{

    return 0;
}

template <typename T>
int ActivationDriver<T>::VerifyBackward()
{

    const double allowedEps = (1 << 2);
    miopenActivationMode_t v_mode;
    double v_Alpha;
    double v_Beta;
    double v_Power;

    miopenGetActivationDescriptor(activDesc, &v_mode, &v_Alpha, &v_Beta, &v_Power);

    int match = 1;
    match     = mloNeuronBackwardRunHostAndVerify<T>(v_mode,
                                                 static_cast<T>(v_Power),
                                                 static_cast<T>(v_Alpha),
                                                 static_cast<T>(v_Beta),
                                                 dinhost.size(),
                                                 in.data(),
                                                 out.data(),
                                                 din.data(),
                                                 dout.data(),
                                                 allowedEps);
    if(match)
        printf("Backward Activation Verifies on CPU and GPU\n");
    return 0;
}

#endif // GUARD_MIOPEN_ACTIV_DRIVER_HPP
