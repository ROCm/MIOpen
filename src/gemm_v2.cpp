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
#include <miopen/gemm_v2.hpp>
#include <miopen/logger.hpp>

#if MIOPEN_USE_ROCBLAS
#include <rocblas.h>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/perf_field.hpp>
#elif MIOPEN_USE_MIOPENGEMM
#include <miopen/miopengemm.hpp>
#endif

namespace miopen {

void CallGemm(Handle& handle,
              GemmDescriptor gemm_desc,
              ConstData_t A,
              int a_offset,
              ConstData_t B,
              int b_offset,
              Data_t C,
              int c_offset,
              std::string* kcache_key)
{
#if MIOPEN_USE_ROCBLAS
    MIOPEN_LOG_FUNCTION("rocBLAS");

    if(!gemm_desc.isColMajor)
    {
        // gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    if(handle.IsProfilingEnabled())
    {
        start = make_hip_event();
        stop  = make_hip_event();
        hipEventRecord(start.get(), nullptr);
    }

    rocblas_sgemm(handle.rhandle.get(),
                  gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                  gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                  gemm_desc.m,
                  gemm_desc.n,
                  gemm_desc.k,
                  &gemm_desc.alpha,
                  static_cast<const float*>(A) + a_offset,
                  gemm_desc.lda,
                  static_cast<const float*>(B) + b_offset,
                  gemm_desc.ldb,
                  &gemm_desc.beta,
                  static_cast<float*>(C) + c_offset,
                  gemm_desc.ldc);

    if(handle.IsProfilingEnabled())
    {
        hipEventRecord(stop.get(), nullptr);
        hipEventSynchronize(stop.get());
        float mS = 0;
        hipEventElapsedTime(&mS, start.get(), stop.get());
        handle.ResetKernelTime();
        handle.AccumKernelTime(mS);
    }

    if(kcache_key != nullptr)
        *kcache_key = FindDbData::GetUnusedKCacheKey();

#elif MIOPEN_USE_MIOPENGEMM
    MIOPEN_LOG_FUNCTION("MIOpenGEMM");

    // do row-to-column major conversion here
    if(!gemm_desc.isColMajor)
    {
        // isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    // making network configs for MIOpenGEMM kernel(s),
    //   using necessary and minimal info,
    //   based on info that's always true:
    //      column-major,
    //      C is not transposed,
    //      workSpace is 0,
    //      fp32
    auto gemm_desc_to_string = [&gemm_desc]() {
        return std::to_string(static_cast<int>(gemm_desc.transA)) + "_" +
               std::to_string(static_cast<int>(gemm_desc.transB)) + "_" +
               std::to_string(gemm_desc.lda) + "_" + std::to_string(gemm_desc.ldb) + "_" +
               std::to_string(gemm_desc.ldc) + "_" + std::to_string(gemm_desc.m) + "_" +
               std::to_string(gemm_desc.n) + "_" + std::to_string(gemm_desc.k);
    };

    const std::string algorithm_name = "MIOpenGEMM";
    const std::string network_config = gemm_desc_to_string();

    if(kcache_key != nullptr)
        *kcache_key = network_config;

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);

    if(kernels.empty())
    {
        MIOpenGEMM::Geometry mgg(true,
                                 gemm_desc.transA,
                                 gemm_desc.transB,
                                 false,
                                 gemm_desc.lda,
                                 gemm_desc.ldb,
                                 gemm_desc.ldc,
                                 gemm_desc.m,
                                 gemm_desc.n,
                                 gemm_desc.k,
                                 0,
                                 'f');

        AddMiopengemmSolution(handle, algorithm_name, network_config, mgg, A, B, C, 0.003, false);

        auto&& new_kernels = handle.GetKernels(algorithm_name, network_config);

        RunMiopengemmSolution(handle,
                              new_kernels,
                              gemm_desc.alpha,
                              A,
                              a_offset,
                              B,
                              b_offset,
                              gemm_desc.beta,
                              C,
                              c_offset);
    }
    else
    {
        RunMiopengemmSolution(handle,
                              kernels,
                              gemm_desc.alpha,
                              A,
                              a_offset,
                              B,
                              b_offset,
                              gemm_desc.beta,
                              C,
                              c_offset);
    }

#else
    MIOPEN_THROW("No GEMM backend");
#endif
}

void CallGemmStridedBatched(Handle& handle,
                            GemmDescriptor gemm_desc,
                            ConstData_t A,
                            int a_offset,
                            ConstData_t B,
                            int b_offset,
                            Data_t C,
                            int c_offset,
                            std::string* kcache_key)
{
#if MIOPEN_USE_ROCBLAS
    MIOPEN_LOG_FUNCTION("rocBLAS");

    if(!gemm_desc.isColMajor)
    {
        // gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    if(handle.IsProfilingEnabled())
    {
        start = make_hip_event();
        stop  = make_hip_event();
        hipEventRecord(start.get(), nullptr);
    }

    rocblas_sgemm_strided_batched(
        handle.rhandle.get(),
        gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
        gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
        gemm_desc.m,
        gemm_desc.n,
        gemm_desc.k,
        &gemm_desc.alpha,
        static_cast<const float*>(A) + a_offset,
        gemm_desc.lda,
        gemm_desc.strideA,
        static_cast<const float*>(B) + b_offset,
        gemm_desc.ldb,
        gemm_desc.strideB,
        &gemm_desc.beta,
        static_cast<float*>(C) + c_offset,
        gemm_desc.ldc,
        gemm_desc.strideC,
        gemm_desc.batch_count);

    if(handle.IsProfilingEnabled())
    {
        hipEventRecord(stop.get(), nullptr);
        hipEventSynchronize(stop.get());
        float mS = 0;
        hipEventElapsedTime(&mS, start.get(), stop.get());
        handle.ResetKernelTime();
        handle.AccumKernelTime(mS);
    }

    if(kcache_key != nullptr)
        *kcache_key = FindDbData::GetUnusedKCacheKey();

#elif MIOPEN_USE_MIOPENGEMM

    CallGemmStridedBatchedSequential(
        handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, kcache_key);
#else
    (void)handle;
    (void)gemm_desc;
    (void)A;
    (void)a_offset;
    (void)B;
    (void)b_offset;
    (void)C;
    (void)c_offset;
    (void)kcache_key;

    MIOPEN_THROW("No GEMM backend");
#endif
}

void CallGemmStridedBatchedSequential(Handle& handle,
                                      GemmDescriptor gemm_desc,
                                      ConstData_t A,
                                      int a_offset,
                                      ConstData_t B,
                                      int b_offset,
                                      Data_t C,
                                      int c_offset,
                                      std::string* kcache_key)
{
#if MIOPEN_USE_ROCBLAS
    MIOPEN_LOG_FUNCTION("rocBLAS");

    if(!gemm_desc.isColMajor)
    {
        // gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    if(handle.IsProfilingEnabled())
    {
        start = make_hip_event();
        stop  = make_hip_event();
        hipEventRecord(start.get(), nullptr);
    }

    for(int i = 0; i < gemm_desc.batch_count; ++i)
    {
        rocblas_sgemm(handle.rhandle.get(),
                      gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                      gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                      gemm_desc.m,
                      gemm_desc.n,
                      gemm_desc.k,
                      &gemm_desc.alpha,
                      static_cast<const float*>(A) + a_offset + i * gemm_desc.strideA,
                      gemm_desc.lda,
                      static_cast<const float*>(B) + b_offset + i * gemm_desc.strideB,
                      gemm_desc.ldb,
                      &gemm_desc.beta,
                      static_cast<float*>(C) + c_offset + i * gemm_desc.strideC,
                      gemm_desc.ldc);
    }

    if(handle.IsProfilingEnabled())
    {
        hipEventRecord(stop.get(), nullptr);
        hipEventSynchronize(stop.get());
        float mS = 0;
        hipEventElapsedTime(&mS, start.get(), stop.get());
        handle.ResetKernelTime();
        handle.AccumKernelTime(mS);
    }

    if(kcache_key != nullptr)
        *kcache_key = FindDbData::GetUnusedKCacheKey();

#elif MIOPEN_USE_MIOPENGEMM
    MIOPEN_LOG_FUNCTION("MIOpenGEMM");

    if(!gemm_desc.isColMajor)
    {
        // gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    // making network configs for MIOpenGEMM kernel(s),
    //   using necessary and minimal info,
    //   based on info that's always true:
    //      column-major,
    //      C is not transposed,
    //      workSpace is 0,
    //      fp32
    auto gemm_desc_to_string = [&gemm_desc]() {
        return std::to_string(static_cast<int>(gemm_desc.transA)) + "_" +
               std::to_string(static_cast<int>(gemm_desc.transB)) + "_" +
               std::to_string(gemm_desc.lda) + "_" + std::to_string(gemm_desc.ldb) + "_" +
               std::to_string(gemm_desc.ldc) + "_" + std::to_string(gemm_desc.m) + "_" +
               std::to_string(gemm_desc.n) + "_" + std::to_string(gemm_desc.k);
    };

    const std::string algorithm_name = "MIOpenGEMM";
    const std::string network_config = gemm_desc_to_string();

    if(kcache_key != nullptr)
        *kcache_key = network_config;

    auto&& old_kernels = handle.GetKernels(algorithm_name, network_config);

    if(old_kernels.empty())
    {
        MIOpenGEMM::Geometry mgg(true,
                                 gemm_desc.transA,
                                 gemm_desc.transB,
                                 false,
                                 gemm_desc.lda,
                                 gemm_desc.ldb,
                                 gemm_desc.ldc,
                                 gemm_desc.m,
                                 gemm_desc.n,
                                 gemm_desc.k,
                                 0,
                                 'f');

        AddMiopengemmSolution(handle, algorithm_name, network_config, mgg, A, B, C, 0.003, false);

        auto&& new_kernels = handle.GetKernels(algorithm_name, network_config);

        float gemm_time = 0;

        for(int i = 0; i < gemm_desc.batch_count; ++i)
        {
            RunMiopengemmSolution(handle,
                                  new_kernels,
                                  gemm_desc.alpha,
                                  A,
                                  a_offset + i * gemm_desc.strideA,
                                  B,
                                  b_offset + i * gemm_desc.strideB,
                                  gemm_desc.beta,
                                  C,
                                  c_offset + i * gemm_desc.strideC);

            if(handle.IsProfilingEnabled())
            {
                if(i == gemm_desc.batch_count - 1)
                    handle.AccumKernelTime(gemm_time);
                else
                    gemm_time += handle.GetKernelTime();
            }
        }
    }
    else
    {
        float gemm_time = 0;

        for(int i = 0; i < gemm_desc.batch_count; ++i)
        {
            RunMiopengemmSolution(handle,
                                  old_kernels,
                                  gemm_desc.alpha,
                                  A,
                                  a_offset + i * gemm_desc.strideA,
                                  B,
                                  b_offset + i * gemm_desc.strideB,
                                  gemm_desc.beta,
                                  C,
                                  c_offset + i * gemm_desc.strideC);

            if(handle.IsProfilingEnabled())
            {
                if(i == gemm_desc.batch_count - 1)
                    handle.AccumKernelTime(gemm_time);
                else
                    gemm_time += handle.GetKernelTime();
            }
        }
    }
#else
    (void)handle;
    (void)gemm_desc;
    (void)A;
    (void)a_offset;
    (void)B;
    (void)b_offset;
    (void)C;
    (void)c_offset;
    (void)kcache_key;

    MIOPEN_THROW("No GEMM backend");
#endif
}

// y = w * Im2Col(x)
GemmDescriptor CreateGemmDescriptorConvFwd(const TensorDescriptor& wDesc,
                                           const TensorDescriptor& xDesc,
                                           const TensorDescriptor& yDesc)
{
    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n;
    int n                 = out_h * out_w;
    int k                 = in_c * wei_h * wei_w;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

// dx = Col2Im(transpose(w) * dy)
GemmDescriptor CreateGemmDescriptorConvBwdData(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& dxDesc)
{
    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = in_c * wei_h * wei_w;
    int n                 = out_h * out_w;
    int k                 = wei_n;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

// dw = dy * transpose(Im2Col(x))
GemmDescriptor CreateGemmDescriptorConvBwdWeight(const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& xDesc,
                                                 const TensorDescriptor& dwDesc)
{
    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = true;
    int m                 = wei_n;
    int n                 = in_c * wei_h * wei_w;
    int k                 = out_h * out_w;
    int lda               = k;
    int ldb               = k;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

// y = CNHW2NCHW(w * NCHW2CNHW(x))
GemmDescriptor CreateGemmDescriptorConvCNHWFwd(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& yDesc)
{
    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n;
    int n                 = in_n * out_h * out_w;
    int k                 = in_c;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
GemmDescriptor CreateGemmDescriptorConvCNHWBwdData(const TensorDescriptor& wDesc,
                                                   const TensorDescriptor& dyDesc,
                                                   const TensorDescriptor& dxDesc)
{
    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = in_c;
    int n                 = in_n * out_h * out_w;
    int k                 = wei_n;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

// y[i] = w * x[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedParamConv1x1Fwd(const TensorDescriptor& wDesc,
                                                       const TensorDescriptor& xDesc,
                                                       const TensorDescriptor& yDesc)
{
    (void)yDesc;

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n;
    int n                 = in_h * in_w;
    int k                 = in_c;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = in_n;
    long long int strideA = 0;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

// dx[i] = transpose(w) * dy[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedParamConv1x1BwdData(const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& dyDesc,
                                                           const TensorDescriptor& dxDesc)
{
    (void)dyDesc;

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = in_c;
    int n                 = in_h * in_w;
    int k                 = wei_n;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = in_n;
    long long int strideA = 0;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

// dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
GemmDescriptor CreateGemmStridedBatchedParamConv1x1BwdWeight(const TensorDescriptor& dyDesc,
                                                             const TensorDescriptor& xDesc,
                                                             const TensorDescriptor& dwDesc)
{
    (void)dyDesc;

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(dwDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = true;
    int m                 = wei_n;
    int n                 = in_c;
    int k                 = in_h * in_w;
    int lda               = k;
    int ldb               = k;
    int ldc               = n;
    int batch_count       = in_n;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta};
}

} // namespace miopen
