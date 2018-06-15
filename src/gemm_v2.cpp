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
#if 0
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

    if(handle.GetKernels(algorithm_name, network_config).empty())
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
    }

    RunMiopengemmSolution(handle,
                          algorithm_name,
                          network_config,
                          gemm_param.alpha,
                          A,
                          a_offset,
                          B,
                          b_offset,
                          gemm_param.beta,
                          C,
                          c_offset);

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
#if 0
    std::cout << std::endl << __func__ << ": rocBLAS" << std::endl;
#endif

#if GEMM_V2_CPP_DEBUG
    std::cout << __func__ << ": gemm_param before swap" << std::endl;
    std::cout << gemm_param << std::endl;

    const float* A_old       = static_cast<const float*>(A);
    const float* B_old       = static_cast<const float*>(B);
    int a_offset_old         = a_offset;
    int b_offset_old         = b_offset;
    GemmParam gemm_param_old = gemm_param;
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

#if GEMM_V2_CPP_DEBUG
    {
        std::cout << __func__ << ": gemm_param after swap" << std::endl;
        std::cout << gemm_param << std::endl;

        std::size_t a_sz = a_offset_old + gemm_param_old.m * gemm_param_old.k +
                           (gemm_param_old.batch_count - 1) * gemm_param_old.strideA;
        std::size_t b_sz = b_offset_old + gemm_param_old.k * gemm_param_old.n +
                           (gemm_param_old.batch_count - 1) * gemm_param_old.strideB;
        std::size_t c_sz = c_offset + gemm_param_old.m * gemm_param_old.n +
                           (gemm_param_old.batch_count - 1) * gemm_param_old.strideC;

        std::vector<float> tmp_a(a_sz, 0.);
        std::vector<float> tmp_b(b_sz, 0.);
        std::vector<float> tmp_c(c_sz, 0.);

        hipMemcpy(tmp_a.data(), A_old, a_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_b.data(), B_old, b_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_c.data(), C, c_sz * sizeof(float), hipMemcpyHostToDevice);

        std::cout << std::endl;
        std::cout << __func__ << ": A before call rocblas: " << tmp_a << std::endl;
        std::cout << __func__ << ": B before call rocblas: " << tmp_b << std::endl;
        std::cout << __func__ << ": C before call rocblas: " << tmp_c << std::endl;

        float sum_c = std::accumulate(tmp_c.begin(), tmp_c.end(), float(0), std::plus<float>());
        std::cout << __func__ << ": sum_c before call rocblas" << sum_c << std::endl;
    }
#endif

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

#if GEMM_V2_CPP_DEBUG
    {
        std::size_t a_sz = a_offset_old + gemm_param_old.m * gemm_param_old.k +
                           (gemm_param_old.batch_count - 1) * gemm_param_old.strideA;
        std::size_t b_sz = b_offset_old + gemm_param_old.k * gemm_param_old.n +
                           (gemm_param_old.batch_count - 1) * gemm_param_old.strideB;
        std::size_t c_sz = c_offset + gemm_param_old.m * gemm_param_old.n +
                           (gemm_param_old.batch_count - 1) * gemm_param_old.strideC;

        std::vector<float> tmp_a(a_sz, 0.);
        std::vector<float> tmp_b(b_sz, 0.);
        std::vector<float> tmp_c(c_sz, 0.);

        hipMemcpy(tmp_a.data(), A_old, a_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_b.data(), B_old, b_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_c.data(), C, c_sz * sizeof(float), hipMemcpyHostToDevice);

        std::cout << std::endl;
        std::cout << __func__ << ": A after call rocblas: " << tmp_a << std::endl;
        std::cout << __func__ << ": B after call rocblas: " << tmp_b << std::endl;
        std::cout << __func__ << ": C after call rocblas: " << tmp_c << std::endl;

        float sum_c = std::accumulate(tmp_c.begin(), tmp_c.end(), float(0), std::plus<float>());
        std::cout << __func__ << ": sum_c after call rocblas" << sum_c << std::endl;
    }
#endif

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

// y = w * x
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

// dx = transpose(w) * dy
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

} // namespace miopen
