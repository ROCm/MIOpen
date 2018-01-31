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
#ifndef GUARD_MIOPEN_SOFTMAX_DRIVER_HPP
#define GUARD_MIOPEN_SOFTMAX_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <../test/verify.hpp>
#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>

template <typename T>
class SoftmaxDriver : public Driver
{
    public:
    SoftmaxDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();
    int RunForwardCPU();

    int RunBackwardGPU();
    int RunBackwardCPU();

    int VerifyBackward();
    int VerifyForward();
    ~SoftmaxDriver()
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyTensorDescriptor(dOutputTensor);
        miopenDestroyTensorDescriptor(dInputTensor);
    }

    private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<T> in;
    std::vector<T> out;
    std::vector<T> outhost;

    miopenTensorDescriptor_t dInputTensor;
    miopenTensorDescriptor_t dOutputTensor;

    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> dout_dev;

    std::vector<T> din;
    std::vector<T> dout;
    std::vector<T> dinhost;
};

template <typename T>
int SoftmaxDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename T>
int SoftmaxDriver<T>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    SetTensor4d(inputTensor, in_len);
    SetTensor4d(dInputTensor, in_len);
    SetTensor4d(outputTensor, in_len);
    SetTensor4d(dOutputTensor, in_len);

    return (0);
}

template <typename T>
int SoftmaxDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Softmax (Default=0)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("alpha", 'A', "1.0", "Softmax shift (Default=0.0)", "double");
    inflags.AddInputFlag("beta", 'B', "0.0", "Softmax scale (Default=0.0)", "double");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return 0;
}

template <typename T>
std::vector<int> SoftmaxDriver<T>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    return std::vector<int>({in_n, in_c, in_h, in_w});
}

template <typename T>
int SoftmaxDriver<T>::AllocateBuffersAndCopy()
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

    din     = std::vector<T>(in_sz, 0);
    dout    = std::vector<T>(out_sz);
    dinhost = std::vector<T>(in_sz, 0);

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = static_cast<T>(static_cast<double>(rand()) * (1.0 / RAND_MAX));
    }

    for(int i = 0; i < out_sz; i++)
    {
        dout[i] = static_cast<double>((rand()) * (1.0 / RAND_MAX) - 0.5) * 0.001;
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
int SoftmaxDriver<T>::RunForwardGPU()
{

    float alpha = 1, beta = 0;

    miopenSoftmaxForward(
        GetHandle(), &alpha, inputTensor, in_dev->GetMem(), &beta, outputTensor, out_dev->GetMem());

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSoftmaxForward(GetHandle(),
                             &alpha,
                             inputTensor,
                             in_dev->GetMem(),
                             &beta,
                             outputTensor,
                             out_dev->GetMem());
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward Softmax Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Forward Softmax Elapsed: %f ms\n", time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename T>
int SoftmaxDriver<T>::RunForwardCPU()
{
    int n, c, h, w;
    miopenGet4dTensorDescriptorLengths(inputTensor, &n, &c, &h, &w);

    std::copy(in.begin(), in.end(), outhost.begin());
    std::vector<float> channel_max(n * h * w, -FLT_MAX);

    for(int i = 0; i < n; i++)
    {
        for(int s = 0; s < h * w; s++)
        {
            for(int j = 0; j < c; j++)
            {
                channel_max[i * h * w + s] =
                    std::max(outhost[(i * c + j) * h * w + s], channel_max[i * h * w + s]);
            }

            for(int j = 0; j < c; j++)
            {
                outhost[(i * c + j) * h * w + s] -= channel_max[i * h * w + s];
                outhost[(i * c + j) * h * w + s] = exp(outhost[(i * c + j) * h * w + s]);
            }

            channel_max[i * h * w + s] = 0.0;
            for(int j = 0; j < c; j++)
            {
                channel_max[i * h * w + s] += outhost[(i * c + j) * h * w + s];
            }

            for(int j = 0; j < c; j++)
            {
                outhost[(i * c + j) * h * w + s] /= channel_max[i * h * w + s];
            }
        }
    }

    return 0;
}

template <typename T>
int SoftmaxDriver<T>::RunBackwardGPU()
{
    float alpha = 1., beta = 0.;

    miopenSoftmaxBackward(GetHandle(),
                          &alpha,
                          outputTensor,
                          out_dev->GetMem(),
                          dOutputTensor,
                          dout_dev->GetMem(),
                          &beta,
                          dInputTensor,
                          din_dev->GetMem());

    Timer t;
    START_TIME;

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSoftmaxBackward(GetHandle(),
                              &alpha,
                              outputTensor,
                              out_dev->GetMem(),
                              dOutputTensor,
                              dout_dev->GetMem(),
                              &beta,
                              dInputTensor,
                              din_dev->GetMem());
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        STOP_TIME;
        if(WALL_CLOCK)
            printf("Wall-clock Time Backward Softmax Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));
        printf("GPU Kernel Time Backward Softmax Elapsed: %f ms\n", time);
    }

    din_dev->FromGPU(GetStream(), din.data());
    return (0);
}

template <typename T>
int SoftmaxDriver<T>::VerifyForward()
{
    RunForwardCPU();

    auto error             = miopen::rms_range(outhost, out);
    const double tolerance = 1e-6;
    if(error > tolerance)
    {
        std::cout << std::string("Forward Softmax Failed: ") << error << std::string("\n");
    }
    else
    {
        printf("Forward Softmax Verifies on CPU and GPU\n");
    }

    return 0;
}

template <typename T>
int SoftmaxDriver<T>::RunBackwardCPU()
{
    int n, c, h, w;
    miopenGet4dTensorDescriptorLengths(dOutputTensor, &n, &c, &h, &w);

    std::copy(dout.begin(), dout.end(), dinhost.begin());
    std::vector<float> channel_dot(n * h * w, 0.0);

    for(int i = 0; i < n; i++)
    {
        for(int s = 0; s < h * w; s++)
        {
            for(int j = 0; j < c; j++)
            {
                channel_dot[i * h * w + s] +=
                    out[(i * c + j) * h * w + s] * dinhost[(i * c + j) * h * w + s];
            }

            for(int j = 0; j < c; j++)
            {
                dinhost[(i * c + j) * h * w + s] -= channel_dot[i * h * w + s];
                dinhost[(i * c + j) * h * w + s] =
                    out[(i * c + j) * h * w + s] * dinhost[(i * c + j) * h * w + s];
            }
        }
    }

    return 0;
}

template <typename T>
int SoftmaxDriver<T>::VerifyBackward()
{
    RunBackwardCPU();

    auto error             = miopen::rms_range(dinhost, din);
    const double tolerance = 1e-6;
    if(error > tolerance)
    {
        std::cout << std::string("Backward Softmax Failed: ") << error << std::string("\n");
    }
    else
    {
        printf("Backward Softmax Verifies on CPU and GPU\n");
    }

    return 0;
}

#endif // GUARD_MIOPEN_SOFTMAX_DRIVER_HPP
