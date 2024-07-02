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

#include <miopen/config.h>

#if MIOPEN_USE_GEMM
#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "util_driver.hpp"

#include <../test/verify.hpp>

#include <miopen/gemm_v2.hpp>
#include <miopen/miopen.h>

#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <memory>
#include <numeric>
#include <vector>

#define GEMM_DRIVER_DEBUG 0

template <typename T>
void callCpuGemmStridedBatched(bool isColMajor,
                               bool transA,
                               bool transB,
                               int m,
                               int n,
                               int k,
                               T alpha,
                               const void* A,
                               int a_offset,
                               int lda,
                               long long int strideA,
                               const void* B,
                               int b_offset,
                               int ldb,
                               long long int strideB,
                               T beta,
                               void* C,
                               int c_offset,
                               int ldc,
                               long long int strideC,
                               int batch_count)
{
    const T* a_ptr = static_cast<const T*>(A);
    const T* b_ptr = static_cast<const T*>(B);
    T* c_ptr       = static_cast<T*>(C);

    // our cpu GEMM logic is row-major
    if(isColMajor)
    {
        isColMajor = false;
        std::swap(a_ptr, b_ptr);
        std::swap(a_offset, b_offset);
        std::swap(transA, transB);
        std::swap(m, n);
        std::swap(lda, ldb);
        std::swap(strideA, strideB);
    }

    for(int bi = 0; bi < batch_count; ++bi)
    {
        for(int mi = 0; mi < m; ++mi)
        {
            for(int ni = 0; ni < n; ++ni)
            {
                double y = 0;
                for(int ki = 0; ki < k; ++ki)
                {
                    int aindex = transA ? a_offset + strideA * bi + lda * ki + mi
                                        : a_offset + strideA * bi + lda * mi + ki;
                    int bindex = transB ? b_offset + strideB * bi + ldb * ni + ki
                                        : b_offset + strideB * bi + ldb * ki + ni;
                    y += static_cast<double>(a_ptr[aindex]) * static_cast<double>(b_ptr[bindex]);
                }
                int cindex = c_offset + strideC * bi + ldc * mi + ni;
                c_ptr[cindex] =
                    static_cast<T>(static_cast<double>(alpha) * y +
                                   static_cast<double>(beta) * static_cast<double>(c_ptr[cindex]));
            }
        }
    }
}

template <typename T>
class GemmDriver : public Driver
{
public:
    GemmDriver() : Driver() {}

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;

    int RunForwardCPU();

    int RunBackwardGPU() override;

    int VerifyBackward() override;
    int VerifyForward() override;
    ~GemmDriver() override {}

private:
    InputFlags inflags;

    std::unique_ptr<GPUMem> a_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::unique_ptr<GPUMem> c_dev;

    std::vector<T> a;
    std::vector<T> b;
    std::vector<T> c;
    std::vector<T> chost;

    T alpha, beta;

    miopen::GemmDescriptor gemm_desc = {
        false, false, false, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0f, 0.0f, miopenFloat, false};
};

template <typename T>
int GemmDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Gemm (Default=1)", "int");
    inflags.AddInputFlag("batch_count", 'b', "1", "batch count for Gemm (Default=1)", "int");
    inflags.AddInputFlag(
        "isColMajor", 'C', "0", "Are matrices in column major? (Default=0)", "int");
    inflags.AddInputFlag("a_h", 'm', "256", "Height of A matrix (Default=256)", "int");
    inflags.AddInputFlag("a_w", 'k', "256", "Width of A matrix (Default=256)", "int");
    inflags.AddInputFlag("b_w", 'n', "256", "Width of B matrix (Default=256)", "int");
    inflags.AddInputFlag("alpha", 'A', "1.0", "Gemm alpha (Default=1.0)", "float");
    inflags.AddInputFlag("beta", 'B', "0.0", "Gemm beta (Default=0.0)", "float");
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
    if constexpr(std::is_same_v<T, float>)
    {
        gemm_desc.dataType = miopenFloat;
    }
    else if constexpr(std::is_same_v<T, float16>)
    {
        gemm_desc.dataType = miopenHalf;
    }
    else
    {
        static_assert(!"unsupported type");
    }

    gemm_desc.a_cast_type = data_type;
    gemm_desc.b_cast_type = data_type;

    gemm_desc.isColMajor = inflags.GetValueInt("isColMajor") != 0;
    gemm_desc.m          = inflags.GetValueInt("a_h");
    gemm_desc.k          = inflags.GetValueInt("a_w");
    gemm_desc.n          = inflags.GetValueInt("b_w");

    gemm_desc.transA = inflags.GetValueInt("transA") != 0;
    gemm_desc.transB = inflags.GetValueInt("transB") != 0;

    gemm_desc.alpha = inflags.GetValueDouble("alpha");
    gemm_desc.beta  = inflags.GetValueDouble("beta");

    // we are assuming: each matrix is saved in continuous memory, no empty memory
    // between batches of matrices
    if(gemm_desc.isColMajor)
    {
        gemm_desc.lda = gemm_desc.transA == 0 ? gemm_desc.m : gemm_desc.k;
        gemm_desc.ldb = gemm_desc.transB == 0 ? gemm_desc.k : gemm_desc.n;
        gemm_desc.ldc = gemm_desc.m; // C is never transposed
    }
    else
    {
        gemm_desc.lda = gemm_desc.transA == 0 ? gemm_desc.k : gemm_desc.m;
        gemm_desc.ldb = gemm_desc.transB == 0 ? gemm_desc.n : gemm_desc.k;
        gemm_desc.ldc = gemm_desc.n; // C is never transposed
    }

    gemm_desc.batch_count = inflags.GetValueInt("batch_count");

#if GEMM_DRIVER_DEBUG
    gemm_desc.strideA = 0;
#else
    gemm_desc.strideA = gemm_desc.m * gemm_desc.k;
#endif
    gemm_desc.strideB = gemm_desc.k * gemm_desc.n;
    gemm_desc.strideC = gemm_desc.m * gemm_desc.n;

    gemm_desc.deterministic = false;
    return (0);
}

template <typename T>
int GemmDriver<T>::AllocateBuffersAndCopy()
{
    size_t a_sz = gemm_desc.m * gemm_desc.k + (gemm_desc.batch_count - 1) * gemm_desc.strideA;
    size_t b_sz = gemm_desc.k * gemm_desc.n + (gemm_desc.batch_count - 1) * gemm_desc.strideB;
    size_t c_sz = gemm_desc.m * gemm_desc.n + (gemm_desc.batch_count - 1) * gemm_desc.strideC;

    DEFINE_CONTEXT(ctx);
#if MIOPEN_BACKEND_OPENCL
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif
    a_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, a_sz, sizeof(T)));
    b_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(T)));
    c_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, c_sz, sizeof(T)));

    a = std::vector<T>(a_sz);
    b = std::vector<T>(b_sz);
#if GEMM_DRIVER_DEBUG
    c = std::vector<T>(c_sz, static_cast<T>(1));
#else

    c = std::vector<T>(c_sz, static_cast<T>(0));
#endif
    chost = c;

    for(int i = 0; i < a_sz; i++)
    {
#if GEMM_DRIVER_DEBUG
        a[i] = static_cast<T>(i);
#else
        a[i] = prng::gen_canonical<T>();
#endif
    }

    for(int i = 0; i < b_sz; i++)
    {
#if GEMM_DRIVER_DEBUG
        b[i] = static_cast<T>(i);
#else
        b[i] = prng::gen_A_to_B(static_cast<T>(-0.5), static_cast<T>(0.5));
#endif
    }
    status_t status;
    status = a_dev->ToGPU(q, a.data());
    status |= b_dev->ToGPU(q, b.data());
    status |= c_dev->ToGPU(q, c.data());

    if(status != STATUS_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename T>
int GemmDriver<T>::RunForwardGPU()
{
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
#if GEMM_DRIVER_DEBUG
        {
            std::cout << std::endl;

            std::vector<T> a_tmp(a_dev->sz);
            a_dev->FromGPU(GetStream(), a_tmp.data());
            std::cout << __func__ << "before GEMM, a_tmp: " << a_tmp << std::endl;

            std::vector<T> b_tmp(b_dev->sz);
            b_dev->FromGPU(GetStream(), b_tmp.data());
            std::cout << __func__ << "before GEMM, b_tmp: " << b_tmp << std::endl;

            std::vector<T> c_tmp(c_dev->sz);
            c_dev->FromGPU(GetStream(), c_tmp.data());
            std::cout << __func__ << "before GEMM, c_tmp: " << c_tmp << std::endl;
        }
#endif

        if(gemm_desc.batch_count > 1)
            CallGemmStridedBatched(miopen::deref(GetHandle()),
                                   gemm_desc,
                                   a_dev->GetMem(),
                                   0,
                                   b_dev->GetMem(),
                                   0,
                                   c_dev->GetMem(),
                                   0);
        else
            CallGemm(miopen::deref(GetHandle()),
                     gemm_desc,
                     a_dev->GetMem(),
                     0,
                     b_dev->GetMem(),
                     0,
                     c_dev->GetMem(),
                     0);

#if GEMM_DRIVER_DEBUG
        {
            std::cout << std::endl;

            std::vector<T> a_tmp(a_dev->sz);
            a_dev->FromGPU(GetStream(), a_tmp.data());
            std::cout << __func__ << ": after GEMM, a_tmp: " << a_tmp << std::endl;

            std::vector<T> b_tmp(b_dev->sz);
            b_dev->FromGPU(GetStream(), b_tmp.data());
            std::cout << __func__ << ": after GEMM, b_tmp: " << b_tmp << std::endl;

            std::vector<T> c_tmp(c_dev->sz);
            c_dev->FromGPU(GetStream(), c_tmp.data());
            std::cout << __func__ << ": after_GEMM, c_tmp: " << c_tmp << std::endl;
        }
#endif
    }

    if(inflags.GetValueInt("time") == 1)
    {
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        printf("GPU Kernel Time Gemm Elapsed: %f ms\n", time);
    }

    c_dev->FromGPU(GetStream(), c.data());
    return miopenStatusSuccess;
}

template <typename T>
int GemmDriver<T>::RunForwardCPU()
{
    callCpuGemmStridedBatched<T>(gemm_desc.isColMajor,
                                 gemm_desc.transA,
                                 gemm_desc.transB,
                                 gemm_desc.m,
                                 gemm_desc.n,
                                 gemm_desc.k,
                                 static_cast<T>(gemm_desc.alpha),
                                 a.data(),
                                 0,
                                 gemm_desc.lda,
                                 gemm_desc.strideA,
                                 b.data(),
                                 0,
                                 gemm_desc.ldb,
                                 gemm_desc.strideB,
                                 static_cast<T>(gemm_desc.beta),
                                 chost.data(),
                                 0,
                                 gemm_desc.ldc,
                                 gemm_desc.strideC,
                                 gemm_desc.batch_count);

    return 0;
}

template <typename T>
int GemmDriver<T>::RunBackwardGPU()
{
    return (0);
}

template <typename T>
int GemmDriver<T>::VerifyForward()
{
    RunForwardCPU();

    c_dev->FromGPU(GetStream(), c.data());

#if GEMM_DRIVER_DEBUG
    {
        float sum_c = std::accumulate(c.begin(), c.end(), float(0), std::plus<float>());
        std::cout << __func__ << ": chost: " << chost << std::endl;
        std::cout << __func__ << ": c    : " << c << std::endl;
        std::cout << __func__ << ": sum_c " << sum_c << std::endl;
    }
#endif

    auto error = miopen::rms_range(chost, c);
    const double tolerance =
        ((sizeof(T) == 4) ? static_cast<double>(1e-6) : static_cast<double>(7e-2));
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << std::string("Forward GEMM FAILED: ") << error << std::endl;
    }
    else
    {
        printf("Forward GEMM Verifies on CPU and GPU (err=%f)\n", error);
    }

    return 0;
}

template <typename T>
int GemmDriver<T>::VerifyBackward()
{
    return 0;
}

#endif // MIOPEN_USE_GEMM
#endif // GUARD_MIOPEN_GEMM_DRIVER_HPP
