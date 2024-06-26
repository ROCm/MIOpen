/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/config.h>

#if MIOPEN_USE_HIPBLASLT

#include "get_handle.hpp"
#include "../workspace.hpp"

#include <gtest/gtest.h>
#include <gtest/gtest_common.hpp>

#include <miopen/gemm_v2.hpp>

#include <half/half.hpp>
#include <hip_float8.hpp>

using float16 = half_float::half;
using float8  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

using namespace miopen;

namespace hipblaslt_gemm {

struct TestCase
{
    bool isColMajor;
    int m, n, k;
    bool transA, transB;
    float alpha, beta;
    int batch_count;
};

template <typename T>
static void callCpuGemmStridedBatched(bool isColMajor,
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

static std::vector<TestCase> GetTestCases()
{
    return {{false, 256, 512, 1024, false, false, 1.0f, 0.0, 1},
            {false, 256, 512, 1024, true, false, 1.0f, 0.0, 1},
            {false, 256, 512, 1024, false, true, 1.0f, 0.0, 1},
            {false, 256, 512, 1024, false, true, 1.0f, 0.0, 1},
            {false, 256, 512, 1024, false, false, 1.0f, 0.0, 10},
            {false, 256, 512, 1024, true, false, 1.0f, 0.0, 10},
            {false, 256, 512, 1024, false, true, 1.0f, 0.0, 10},
            {false, 256, 512, 1024, false, true, 1.0f, 0.0, 10},
            {true, 256, 512, 1024, false, false, 1.0f, 0.0, 1},
            {true, 256, 512, 1024, true, false, 1.0f, 0.0, 1},
            {true, 256, 512, 1024, false, true, 1.0f, 0.0, 1},
            {true, 256, 512, 1024, false, true, 1.0f, 0.0, 1},
            {true, 256, 512, 1024, false, false, 1.0f, 0.0, 10},
            {true, 256, 512, 1024, true, false, 1.0f, 0.0, 10},
            {true, 256, 512, 1024, false, true, 1.0f, 0.0, 10},
            {true, 256, 512, 1024, false, true, 1.0f, 0.0, 10},
            {false, 256, 512, 1024, false, true, 1.0f, 1.0f, 10}
            };
}

class HipBLASLtGEMMTestFloat : public testing::TestWithParam<TestCase>{};

class HipBLASLtGEMMTestHalf : public testing::TestWithParam<TestCase>{};

class HipBLASLtGEMMTestBFloat16 : public testing::TestWithParam<TestCase>{};

class HipBLASLtGEMMTestFloat8 : public testing::TestWithParam<TestCase>{};

class HipBLASLtGEMMTestBFloat8 : public testing::TestWithParam<TestCase>{};

static GemmDescriptor GetGemmDescriptor(const TestCase& testCase, miopenDataType_t dataType)
{
    int lda = 0;
    int ldb = 0;
    int ldc = 0;

    if(testCase.isColMajor)
    {
        lda = testCase.transA == 0 ? testCase.m : testCase.k;
        ldb = testCase.transB == 0 ? testCase.k : testCase.n;
        ldc = testCase.m; // C is never transposed
    }
    else
    {
        lda = testCase.transA == 0 ? testCase.k : testCase.m;
        ldb = testCase.transB == 0 ? testCase.n : testCase.k;
        ldc = testCase.n; // C is never transposed
    }

    size_t strideA = testCase.m * testCase.k;
    size_t strideB = testCase.k * testCase.n;
    size_t strideC = testCase.m * testCase.n;

    return {testCase.isColMajor, testCase.transA, testCase.transB, testCase.m, testCase.n, testCase.k, lda, ldb, ldc, testCase.batch_count, strideA, strideB, strideC, testCase.alpha, testCase.beta, dataType, false};
}

template <typename T, typename disabled_mask, typename enabled_mask>
static void RunGemmDescriptors(const TestCase& testCase, miopenDataType_t dataType)
{
    if(IsTestSupportedForDevMask<disabled_mask, enabled_mask>())
    {
        GemmDescriptor desc = GetGemmDescriptor(testCase, dataType);

        size_t aSize = desc.batch_count * desc.strideA;
        size_t bSize = desc.batch_count * desc.strideB;
        size_t cSize = desc.batch_count * desc.strideC;

        Workspace workspaceA_device(aSize * sizeof(T));
        Workspace workspaceB_device(bSize * sizeof(T));
        Workspace workspaceC_device(cSize * sizeof(T));

        std::vector<T> workspaceA_host(aSize);
        std::vector<T> workspaceB_host(bSize);
        std::vector<T> workspaceC_host(cSize, static_cast<T>(1));

        for(auto& index : workspaceA_host)
        {
            index = prng::gen_canonical<T>();
        }

        for(auto& index : workspaceB_host)
        {
            index = prng::gen_A_to_B(static_cast<T>(-0.5), static_cast<T>(0.5f));
        }

        workspaceA_device.Write(workspaceA_host);
        workspaceB_device.Write(workspaceB_host);
        workspaceC_device.Write(workspaceC_host);

        Handle& handle = get_handle();

        if(desc.batch_count == 1)
        {
            EXPECT_EQUAL(CallGemm(handle, desc, workspaceA_device.ptr(), 0, workspaceB_device.ptr(), 0, workspaceC_device.ptr(), 0, GemmBackend_t::hipblaslt), miopenStatus_t::miopenStatusSuccess);
        }
        else
        {
            EXPECT_EQUAL(CallGemmStridedBatched(handle, desc, workspaceA_device.ptr(), 0, workspaceB_device.ptr(), 0, workspaceC_device.ptr(), 0, GemmBackend_t::hipblaslt), miopenStatus_t::miopenStatusSuccess);
        }

        callCpuGemmStridedBatched<T>(desc.isColMajor,
                                    desc.transA,
                                    desc.transB,
                                    desc.m,
                                    desc.n,
                                    desc.k,
                                    static_cast<T>(desc.alpha),
                                    workspaceA_host.data(),
                                    0,
                                    desc.lda,
                                    desc.strideA,
                                    workspaceB_host.data(),
                                    0,
                                    desc.ldb,
                                    desc.strideB,
                                    static_cast<T>(desc.beta),
                                    workspaceC_host.data(),
                                    0,
                                    desc.ldc,
                                    desc.strideC,
                                    desc.batch_count);

        auto error = miopen::rms_range(workspaceC_host, workspaceC_device.Read<std::vector<T>>());
        EXPECT_TRUE(std::isfinite(error));

        const double tolerance = ((sizeof(T) == 4) ? static_cast<double>(1e-6) : static_cast<double>(7e-2));
        EXPECT_LT(error, tolerance);
    }
    else
    {
        GTEST_SKIP();
    }
}

}
using namespace hipblaslt_gemm; 

TEST_P(HipBLASLtGEMMTestFloat, RunHipBLASLtGEMM)
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx90A, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx103X,Gpu::gfx900, Gpu::gfx906, Gpu::gfx908>;
    RunGemmDescriptors<float, d_mask, e_mask>(GetParam(), miopenDataType_t::miopenFloat);
};
INSTANTIATE_TEST_SUITE_P(HipBLASLtGEMMTestSet, HipBLASLtGEMMTestFloat, testing::ValuesIn(GetTestCases()));

TEST_P(HipBLASLtGEMMTestHalf, RunHipBLASLtGEMM)
{ 
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx90A, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx103X,Gpu::gfx900, Gpu::gfx906, Gpu::gfx908>;
    RunGemmDescriptors<float16, d_mask, e_mask>(GetParam(), miopenDataType_t::miopenHalf);
};
INSTANTIATE_TEST_SUITE_P(HipBLASLtGEMMTestSet, HipBLASLtGEMMTestHalf, testing::ValuesIn(GetTestCases()));

TEST_P(HipBLASLtGEMMTestBFloat16, RunHipBLASLtGEMM)
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx90A, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx103X,Gpu::gfx900, Gpu::gfx906, Gpu::gfx908>;
    RunGemmDescriptors<bfloat16, d_mask, e_mask>(GetParam(), miopenDataType_t::miopenBFloat16);
};
INSTANTIATE_TEST_SUITE_P(HipBLASLtGEMMTestSet, HipBLASLtGEMMTestBFloat16, testing::ValuesIn(GetTestCases()));

TEST_P(HipBLASLtGEMMTestFloat8, RunHipBLASLtGEMM)
{
    using e_mask = enabled<Gpu::gfx94X>;
    using d_mask = disabled<Gpu::gfx103X,Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A, Gpu::gfx110X>;
    RunGemmDescriptors<bfloat16, d_mask, e_mask>(GetParam(), miopenDataType_t::miopenFloat8);
};
INSTANTIATE_TEST_SUITE_P(HipBLASLtGEMMTestSet, HipBLASLtGEMMTestFloat8, testing::ValuesIn(GetTestCases()));

TEST_P(HipBLASLtGEMMTestBFloat8, RunHipBLASLtGEMM)
{
    using e_mask = enabled<Gpu::gfx94X>;
    using d_mask = disabled<Gpu::gfx103X,Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A, Gpu::gfx110X>;
    RunGemmDescriptors<bfloat16, d_mask, e_mask>(GetParam(), miopenDataType_t::miopenBFloat8);
};
INSTANTIATE_TEST_SUITE_P(HipBLASLtGEMMTestSet, HipBLASLtGEMMTestBFloat8, testing::ValuesIn(GetTestCases()));

#endif
