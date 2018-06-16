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

#if MIOPEN_USE_ROCBLAS
#include <rocblas.h>
#elif MIOPEN_USE_MIOPENGEMM
#include <miopen/miopengemm.hpp>
#endif

#define GEMM_V2_CPP_DEBUG 0
#define GEMM_V2_CPP_PROFILE 0

namespace miopen {
#if GEMM_V2_CPP_DEBUG
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vs)
{
    os << "{ size: " << vs.size() << ", entries: ";
    for(auto& v : vs)
        os << v << " ";
    os << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const GemmParam& gemm_param)
{
    os << "{ "
       << "isColMajor " << gemm_param.isColMajor << ", "
       << "transA " << gemm_param.transA << ", "
       << "transB " << gemm_param.transB << ", "
       << "m " << gemm_param.m << ", "
       << "n " << gemm_param.n << ", "
       << "k " << gemm_param.k << ", "
       << "lda " << gemm_param.lda << ", "
       << "ldb " << gemm_param.ldb << ", "
       << "ldc " << gemm_param.ldc << ", "
       << "batch_count " << gemm_param.batch_count << ", "
       << "strideA " << gemm_param.strideA << ", "
       << "strideB " << gemm_param.strideB << ", "
       << "strideC " << gemm_param.strideC << ", "
       << "alpha" << gemm_param.alpha << ", "
       << "beta" << gemm_param.beta << " }";
    return os;
}
#endif

void CallGemm(Handle& handle,
              GemmParam gemm_param,
              ConstData_t A,
              int a_offset,
              ConstData_t B,
              int b_offset,
              Data_t C,
              int c_offset)
{
#if MIOPEN_USE_ROCBLAS
#if GEMM_V2_CPP_DEBUG
    std::cout << std::endl << __func__ << ": rocBLAS" << std::endl;
#endif

    if(!gemm_param.isColMajor)
    {
        // gemm_param.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_param.transA, gemm_param.transB);
        std::swap(gemm_param.m, gemm_param.n);
        std::swap(gemm_param.lda, gemm_param.ldb);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, nullptr);

    rocblas_sgemm(handle.rhandle.get(),
                  gemm_param.transA ? rocblas_operation_transpose : rocblas_operation_none,
                  gemm_param.transB ? rocblas_operation_transpose : rocblas_operation_none,
                  gemm_param.m,
                  gemm_param.n,
                  gemm_param.k,
                  &gemm_param.alpha,
                  static_cast<const float*>(A) + a_offset,
                  gemm_param.lda,
                  static_cast<const float*>(B) + b_offset,
                  gemm_param.ldb,
                  &gemm_param.beta,
                  static_cast<float*>(C) + c_offset,
                  gemm_param.ldc);

    hipEventRecord(stop, nullptr);
    hipDeviceSynchronize();
    float mS = 0;
    hipEventElapsedTime(&mS, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    handle.ResetKernelTime();
    handle.AccumKernelTime(mS);

#elif MIOPEN_USE_MIOPENGEMM
#if GEMM_V2_CPP_DEBUG
    std::cout << __func__ << ": MIOpenGEMM" << std::endl;
#endif

#if GEMM_V2_CPP_PROFILE
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // do row-to-column major conversion here
    if(!gemm_param.isColMajor)
    {
        // isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_param.transA, gemm_param.transB);
        std::swap(gemm_param.m, gemm_param.n);
        std::swap(gemm_param.lda, gemm_param.ldb);
    }

    // making network configs for MIOpenGEMM kernel(s),
    //   using necessary and minimal info,
    //   based on info that's always true:
    //      column-major,
    //      C is not transposed,
    //      workSpace is 0,
    //      fp32
    auto gemm_param_to_string = [&gemm_param]() {
        return std::to_string(gemm_param.transA) + "_" + std::to_string(gemm_param.transB) + "_" +
               std::to_string(gemm_param.lda) + "_" + std::to_string(gemm_param.ldb) + "_" +
               std::to_string(gemm_param.ldc) + "_" + std::to_string(gemm_param.m) + "_" +
               std::to_string(gemm_param.n) + "_" + std::to_string(gemm_param.k);
    };

    const std::string algorithm_name = "MIOpenGEMM";
    const std::string network_config = gemm_param_to_string();

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);

    if(kernels.empty())
    {
        MIOpenGEMM::Geometry mgg(true,
                                 gemm_param.transA,
                                 gemm_param.transB,
                                 false,
                                 gemm_param.lda,
                                 gemm_param.ldb,
                                 gemm_param.ldc,
                                 gemm_param.m,
                                 gemm_param.n,
                                 gemm_param.k,
                                 0,
                                 'f');

        AddMiopengemmSolution(handle, algorithm_name, network_config, mgg, A, B, C, 0.003, false);

        auto&& new_kernels = handle.GetKernels(algorithm_name, network_config);

        RunMiopengemmSolution(handle,
                              new_kernels,
                              gemm_param.alpha,
                              A,
                              a_offset,
                              B,
                              b_offset,
                              gemm_param.beta,
                              C,
                              c_offset);
    }
    else
    {
        RunMiopengemmSolution(handle,
                              kernels,
                              gemm_param.alpha,
                              A,
                              a_offset,
                              B,
                              b_offset,
                              gemm_param.beta,
                              C,
                              c_offset);
    }

#if GEMM_V2_CPP_PROFILE
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << __func__ << ": time: " << std::chrono::duration<double>(t_end - t_start).count()
              << " seconds." << std::endl;
#endif

#else
    MIOPEN_THROW("No GEMM backend");
#endif
}

void CallGemmStridedBatched(Handle& handle,
                            GemmParam gemm_param,
                            ConstData_t A,
                            int a_offset,
                            ConstData_t B,
                            int b_offset,
                            Data_t C,
                            int c_offset)
{
#if MIOPEN_USE_ROCBLAS
#if GEMM_V2_CPP_DEBUG
    std::cout << std::endl << __func__ << ": rocBLAS" << std::endl;
#endif

    if(!gemm_param.isColMajor)
    {
        // gemm_param.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_param.transA, gemm_param.transB);
        std::swap(gemm_param.m, gemm_param.n);
        std::swap(gemm_param.lda, gemm_param.ldb);
        std::swap(gemm_param.strideA, gemm_param.strideB);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, nullptr);
    rocblas_sgemm_strided_batched(
        handle.rhandle.get(),
        gemm_param.transA ? rocblas_operation_transpose : rocblas_operation_none,
        gemm_param.transB ? rocblas_operation_transpose : rocblas_operation_none,
        gemm_param.m,
        gemm_param.n,
        gemm_param.k,
        &gemm_param.alpha,
        static_cast<const float*>(A) + a_offset,
        gemm_param.lda,
        gemm_param.strideA,
        static_cast<const float*>(B) + b_offset,
        gemm_param.ldb,
        gemm_param.strideB,
        &gemm_param.beta,
        static_cast<float*>(C) + c_offset,
        gemm_param.ldc,
        gemm_param.strideC,
        gemm_param.batch_count);
    hipEventRecord(stop, nullptr);
    hipDeviceSynchronize();
    float mS = 0;
    hipEventElapsedTime(&mS, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    handle.ResetKernelTime();
    handle.AccumKernelTime(mS);
#elif MIOPEN_USE_MIOPENGEMM

    CallGemmStridedBatchedSequential(handle, gemm_param, A, a_offset, B, b_offset, C, c_offset);
#else
    (void)handle;
    (void)gemm_param;
    (void)A;
    (void)a_offset;
    (void)B;
    (void)b_offset;
    (void)C;
    (void)c_offset;

    MIOPEN_THROW("No GEMM backend");
#endif
}

void CallGemmStridedBatchedSequential(Handle& handle,
                                      GemmParam gemm_param,
                                      ConstData_t A,
                                      int a_offset,
                                      ConstData_t B,
                                      int b_offset,
                                      Data_t C,
                                      int c_offset)
{
#if MIOPEN_USE_MIOPENGEMM

#if GEMM_V2_CPP_DEBUG
    std::cout << __func__ << ": MIOpenGEMM" << std::endl;
#endif

    if(!gemm_param.isColMajor)
    {
        // gemm_param.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_param.transA, gemm_param.transB);
        std::swap(gemm_param.m, gemm_param.n);
        std::swap(gemm_param.lda, gemm_param.ldb);
        std::swap(gemm_param.strideA, gemm_param.strideB);
    }

    // making network configs for MIOpenGEMM kernel(s),
    //   using necessary and minimal info,
    //   based on info that's always true:
    //      column-major,
    //      C is not transposed,
    //      workSpace is 0,
    //      fp32
    auto gemm_param_to_string = [&gemm_param]() {
        return std::to_string(gemm_param.transA) + "_" + std::to_string(gemm_param.transB) + "_" +
               std::to_string(gemm_param.lda) + "_" + std::to_string(gemm_param.ldb) + "_" +
               std::to_string(gemm_param.ldc) + "_" + std::to_string(gemm_param.m) + "_" +
               std::to_string(gemm_param.n) + "_" + std::to_string(gemm_param.k);
    };

    const std::string algorithm_name = "MIOpenGEMM";
    const std::string network_config = gemm_param_to_string();

    auto&& old_kernels = handle.GetKernels(algorithm_name, network_config);

    if(old_kernels.empty())
    {
        MIOpenGEMM::Geometry mgg(true,
                                 gemm_param.transA,
                                 gemm_param.transB,
                                 false,
                                 gemm_param.lda,
                                 gemm_param.ldb,
                                 gemm_param.ldc,
                                 gemm_param.m,
                                 gemm_param.n,
                                 gemm_param.k,
                                 0,
                                 'f');

        AddMiopengemmSolution(handle, algorithm_name, network_config, mgg, A, B, C, 0.003, false);

        auto&& new_kernels = handle.GetKernels(algorithm_name, network_config);

        float gemm_time = 0;

        for(int i = 0; i < gemm_param.batch_count; ++i)
        {
            RunMiopengemmSolution(handle,
                                  new_kernels,
                                  gemm_param.alpha,
                                  A,
                                  a_offset + i * gemm_param.strideA,
                                  B,
                                  b_offset + i * gemm_param.strideB,
                                  gemm_param.beta,
                                  C,
                                  c_offset + i * gemm_param.strideC);

            if(handle.IsProfilingEnabled())
            {
                if(i == gemm_param.batch_count - 1)
                    handle.AccumKernelTime(gemm_time);
                else
                    gemm_time += handle.GetKernelTime();
            }
        }
    }
    else
    {
        float gemm_time = 0;

        for(int i = 0; i < gemm_param.batch_count; ++i)
        {
            RunMiopengemmSolution(handle,
                                  old_kernels,
                                  gemm_param.alpha,
                                  A,
                                  a_offset + i * gemm_param.strideA,
                                  B,
                                  b_offset + i * gemm_param.strideB,
                                  gemm_param.beta,
                                  C,
                                  c_offset + i * gemm_param.strideC);

            if(handle.IsProfilingEnabled())
            {
                if(i == gemm_param.batch_count - 1)
                    handle.AccumKernelTime(gemm_time);
                else
                    gemm_time += handle.GetKernelTime();
            }
        }
    }
#else
    (void)handle;
    (void)gemm_param;
    (void)A;
    (void)a_offset;
    (void)B;
    (void)b_offset;
    (void)C;
    (void)c_offset;

    MIOPEN_THROW("No GEMM backend");
#endif
}

// y = w * Im2Col(x)
GemmParam CreateGemmParamConvFwd(const TensorDescriptor& wDesc,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& yDesc)
{
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": yDesc: " << yDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
GemmParam CreateGemmParamConvBwdData(const TensorDescriptor& wDesc,
                                     const TensorDescriptor& dyDesc,
                                     const TensorDescriptor& dxDesc)
{
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ":  wDesc: " <<  wDesc << std::endl;
    std::cout << __func__ << ": dxDesc: " << dxDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
GemmParam CreateGemmParamConvBwdWeight(const TensorDescriptor& dyDesc,
                                       const TensorDescriptor& xDesc,
                                       const TensorDescriptor& dwDesc)
{
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ": dwDesc: " << dwDesc << std::endl;
    std::cout << __func__ << ":  xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
GemmParam CreateGemmParamConvCNHWFwd(const TensorDescriptor& wDesc,
                                     const TensorDescriptor& xDesc,
                                     const TensorDescriptor& yDesc)
{
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": yDesc: " << yDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
GemmParam CreateGemmParamConvCNHWBwdData(const TensorDescriptor& wDesc,
                                         const TensorDescriptor& dyDesc,
                                         const TensorDescriptor& dxDesc)
{
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ":  wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": dxDesc: " << dxDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
GemmParam CreateGemmStridedBatchedParamConv1x1Fwd(const TensorDescriptor& wDesc,
                                                  const TensorDescriptor& xDesc,
                                                  const TensorDescriptor& yDesc)
{
    (void)yDesc;
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": yDesc: " << yDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
GemmParam CreateGemmStridedBatchedParamConv1x1BwdData(const TensorDescriptor& wDesc,
                                                      const TensorDescriptor& dyDesc,
                                                      const TensorDescriptor& dxDesc)
{
    (void)dyDesc;
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": dxDesc: " << dxDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
GemmParam CreateGemmStridedBatchedParamConv1x1BwdWeight(const TensorDescriptor& dyDesc,
                                                        const TensorDescriptor& xDesc,
                                                        const TensorDescriptor& dwDesc)
{
#if 0
    std::cout << std::endl << __func__ << std::endl;
    std::cout << __func__ << ": dwDesc: " << dwDesc << std::endl;
    std::cout << __func__ << ":  xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;
#endif

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

    return GemmParam{isColMajor,
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
