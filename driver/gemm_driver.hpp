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
#ifndef GUARD_MIOPEN_GEMM_DRIVER_HPP
#define GUARD_MIOPEN_GEMM_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <miopen/miopen.h>
#include <numeric>
#include <vector>

miopenStatus_t miopenGemm(miopenHandle_t handle,
                          bool isDataColMajor,
                          bool transA,
                          bool transB,
                          int M,
                          int N,
                          int K,
                          const void* alpha,
                          const void* A,
                          int lda,
                          const void* B,
                          int ldb,
                          const void* beta,
                          void* C,
                          int ldc,
                          int find);

template <typename T>
class GemmDriver : public Driver
{
    public:
    GemmDriver() : Driver() {}

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy();

    int RunForwardGPU();

    int RunBackwardGPU();

    int VerifyBackward();
    int VerifyForward();
    ~GemmDriver() {}

    private:
    InputFlags inflags;

    std::unique_ptr<GPUMem> a_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::unique_ptr<GPUMem> c_dev;

    std::vector<T> a;
    std::vector<T> b;
    std::vector<T> c;
    std::vector<T> chost;

    int M, N, K; // Netlib BLAS dims, user inputs
    bool transA, transB;
    T alpha, beta;

    int lda, ldb;
};

template <typename T>
int GemmDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Gemm (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'b', "1", "batch size for Gemm (Default=1)", "int");
    inflags.AddInputFlag("a_h", 'm', "256", "Height of A matrix (Default=256)", "int");
    inflags.AddInputFlag("a_w", 'k', "256", "Width of A matrix (Default=256)", "int");
    inflags.AddInputFlag("b_w", 'n', "256", "Width of B matrix (Default=256)", "int");
    inflags.AddInputFlag("alpha", 'A', "1.0", "Gemm alpha (Default=1.0)", "double");
    inflags.AddInputFlag("beta", 'B', "0.0", "Gemm beta (Default=0.0)", "double");
    inflags.AddInputFlag("transA", 'u', "0", "Transpose A matrix (Default=0)", "int");
    inflags.AddInputFlag("transB", 'v', "0", "Transpose B matrix (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");

    return 0;
}

template <typename T>
int GemmDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return 0;
}

template <typename T>
int GemmDriver<T>::GetandSetData()
{

    M = inflags.GetValueInt("a_h");
    K = inflags.GetValueInt("a_w");
    N = inflags.GetValueInt("b_w");

    transA = inflags.GetValueInt("transA");
    transB = inflags.GetValueInt("transB");

    alpha = inflags.GetValueDouble("alpha");
    beta  = inflags.GetValueDouble("beta");

    lda = transA == 0 ? K : M;
    ldb = transB == 0 ? N : K;

    return (0);
}

template <typename T>
int GemmDriver<T>::AllocateBuffersAndCopy()
{

    size_t a_sz = M * K;
    size_t b_sz = K * N;
    size_t c_sz = M * N;
#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    a_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, a_sz, sizeof(T)));
    b_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(T)));
    c_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, c_sz, sizeof(T)));

    a     = std::vector<T>(a_sz);
    b     = std::vector<T>(b_sz);
    c     = std::vector<T>(c_sz, 0.);
    chost = std::vector<T>(c_sz, 0.);

    for(int i = 0; i < a_sz; i++)
    {
        a[i] = static_cast<T>(static_cast<double>(rand()) * (1.0 / RAND_MAX));
    }

    for(int i = 0; i < b_sz; i++)
    {
        b[i] = static_cast<double>((rand()) * (1.0 / RAND_MAX) - 0.5) * 0.001;
    }
#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
    int status;
#endif
    status = a_dev->ToGPU(q, a.data());
    status |= b_dev->ToGPU(q, b.data());
    status |= c_dev->ToGPU(q, c.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename T>
int GemmDriver<T>::RunForwardGPU()
{

#if MIOPEN_USE_MIOPENGEMM
    miopenGemm(GetHandle(),
               false, // isDataColMajor
               transA,
               transB,
               M,
               N,
               K,
               &alpha,
               a_dev->GetMem(),
               lda,
               b_dev->GetMem(),
               ldb,
               &beta,
               c_dev->GetMem(),
               N,
               1); // find needs to be on to compile the kernel

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        printf("GPU Kernel Time Gemm Elapsed: %f ms\n", time);
    }

    c_dev->FromGPU(GetStream(), c.data());
#else
    std::cerr << "GEMM is not supported\n";
#endif
    return miopenStatusSuccess;
}

template <typename T>
int GemmDriver<T>::RunBackwardGPU()
{
    return (0);
}

template <typename T>
int GemmDriver<T>::VerifyForward()
{
    return 0;
}

template <typename T>
int GemmDriver<T>::VerifyBackward()
{
    return 0;
}

#endif // GUARD_MIOPEN_GEMM_DRIVER_HPP
