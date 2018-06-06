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
#endif

void CallGemm(Handle& handle,
              bool isColMajor,
              bool transA,
              bool transB,
              int m,
              int n,
              int k,
              const void* alpha,
              ConstData_t A,
              int a_offset,
              int lda,
              ConstData_t B,
              int b_offset,
              int ldb,
              const void* beta,
              Data_t C,
              int c_offset,
              int ldc)
{
#if MIOPEN_USE_ROCBLAS
#if 0
    std::cout << std::endl << __func__ << ": rocBLAS" << std::endl;
#endif

    if(!isColMajor)
    {
        isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(transA, transB);
        std::swap(m, n);
        std::swap(lda, ldb);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float alpha_local = *static_cast<const float*>(alpha);
    float beta_local  = *static_cast<const float*>(beta);
    hipEventRecord(start, nullptr);

    rocblas_sgemm(handle.rhandle.get(),
                  transA ? rocblas_operation_transpose : rocblas_operation_none,
                  transB ? rocblas_operation_transpose : rocblas_operation_none,
                  m,
                  n,
                  k,
                  &alpha_local,
                  static_cast<const float*>(A) + a_offset,
                  lda,
                  static_cast<const float*>(B) + b_offset,
                  ldb,
                  &beta_local,
                  static_cast<float*>(C) + c_offset,
                  ldc);

    hipEventRecord(stop, nullptr);
    hipDeviceSynchronize();
    float mS = 0;
    hipEventElapsedTime(&mS, start, stop);
    handle.ResetKernelTime();
    handle.AccumKernelTime(mS);

#elif MIOPEN_USE_MIOPENGEMM
#if 0
    std::cout << __func__ << ": MIOpenGEMM" << std::endl;
#endif
    // do row-to-column major conversion here
    if(!isColMajor)
    {
        isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(transA, transB);
        std::swap(m, n);
        std::swap(lda, ldb);
    }

    // TODO: save a map for MIOpenGENN::Geometry in miopenHandle,
    //  so we don't need to construct a new geometry every time
    MIOpenGEMM::Geometry mgg(true, transA, transB, false, lda, ldb, ldc, m, n, k, 0, 'f');

    const std::string algorithm_name = "MIOpenGEMM";
    const std::string network_config = mgg.get_networkconfig_string();

    if(handle.GetKernels(algorithm_name, network_config).empty())
    {
        FindMiopengemmSolution(handle, mgg, A, B, C, 0.003, false);
    }

    float alpha_local = *static_cast<const float*>(alpha);
    float beta_local  = *static_cast<const float*>(beta);

    RunMiopengemmSolution(
        handle, mgg, alpha_local, A, a_offset, B, b_offset, beta_local, C, c_offset);

#else
    MIOPEN_THROW("No GEMM backend");
#endif
}

void CallGemmStridedBatched(Handle& handle,
                            bool isColMajor,
                            bool transA,
                            bool transB,
                            int m,
                            int n,
                            int k,
                            const void* alpha,
                            ConstData_t A,
                            int a_offset,
                            int lda,
                            long long int strideA,
                            ConstData_t B,
                            int b_offset,
                            int ldb,
                            long long int strideB,
                            const void* beta,
                            Data_t C,
                            int c_offset,
                            int ldc,
                            long long int strideC,
                            int batch_count)
{
#if MIOPEN_USE_ROCBLAS
#if 0
    std::cout << std::endl << __func__ << ": rocBLAS" << std::endl;

    {
        std::cout << __func__ << ": gemm desc before swap" << std::endl;
        std::cout << "{ "
                  << "isColMajor " << isColMajor << ", "
                  << "transA " << transA << ", "
                  << "transB " << transB << ", "
                  << "m " << m << ", "
                  << "n " << n << ", "
                  << "k " << k << ", "
                  << "lda " << lda << ", "
                  << "ldb " << ldb << ", "
                  << "ldc " << ldc << ", "
                  << "strideA " << strideA << ", "
                  << "strideB " << strideB << ", "
                  << "strideC " << strideC << ", "
                  << "batch_count " << batch_count << " }" << std::endl;
    }

    const float* A_old = static_cast<const float*>(A);
    const float* B_old = static_cast<const float*>(B);
    int a_offset_old   = a_offset;
    int b_offset_old   = b_offset;
    int m_old          = m;
    int n_old          = n;
    int k_old          = k;
    int strideA_old    = strideA;
    int strideB_old    = strideB;
#endif

    if(!isColMajor)
    {
        isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(transA, transB);
        std::swap(m, n);
        std::swap(lda, ldb);
        std::swap(strideA, strideB);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float alpha_local = *static_cast<const float*>(alpha);
    float beta_local  = *static_cast<const float*>(beta);

#if GEMM_V2_CPP_DEBUG
    {
        std::cout << __func__ << ": alpha_local " << alpha_local << ", beta_local " << beta_local
                  << std::endl;

        std::cout << __func__ << ": gemm desc after swap" << std::endl;
        std::cout << "{ "
                  << "isColMajor " << isColMajor << ", "
                  << "transA " << transA << ", "
                  << "transB " << transB << ", "
                  << "m " << m << ", "
                  << "n " << n << ", "
                  << "k " << k << ", "
                  << "lda " << lda << ", "
                  << "ldb " << ldb << ", "
                  << "ldc " << ldc << ", "
                  << "strideA " << strideA << ", "
                  << "strideB " << strideB << ", "
                  << "strideC " << strideC << ", "
                  << "batch_count " << batch_count << " }" << std::endl;

        std::size_t a_sz = a_offset_old + m_old * k_old + (batch_count - 1) * strideA_old;
        std::size_t b_sz = b_offset_old + k_old * n_old + (batch_count - 1) * strideB_old;
        std::size_t c_sz = c_offset + m_old * n_old + (batch_count - 1) * strideC;

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
    rocblas_sgemm_strided_batched(handle.rhandle.get(),
                                  transA ? rocblas_operation_transpose : rocblas_operation_none,
                                  transB ? rocblas_operation_transpose : rocblas_operation_none,
                                  m,
                                  n,
                                  k,
                                  &alpha_local,
                                  static_cast<const float*>(A) + a_offset,
                                  lda,
                                  strideA,
                                  static_cast<const float*>(B) + b_offset,
                                  ldb,
                                  strideB,
                                  &beta_local,
                                  static_cast<float*>(C) + c_offset,
                                  ldc,
                                  strideC,
                                  batch_count);
    hipEventRecord(stop, nullptr);
    hipDeviceSynchronize();
    float mS = 0;
    hipEventElapsedTime(&mS, start, stop);
    handle.ResetKernelTime();
    handle.AccumKernelTime(mS);

#if GEMM_V2_CPP_DEBUG
    {
        std::size_t a_sz = a_offset_old + m_old * k_old + (batch_count - 1) * strideA_old;
        std::size_t b_sz = b_offset_old + k_old * n_old + (batch_count - 1) * strideB_old;
        std::size_t c_sz = c_offset + m_old * n_old + (batch_count - 1) * strideC;

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
        std::cout << __func__ << ": sum_c after call rocblas" << sum_c << std::endl;
    }
#endif

#else
    (void)handle;
    (void)isColMajor;
    (void)transA;
    (void)transB;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)A;
    (void)a_offset;
    (void)lda;
    (void)strideA;
    (void)B;
    (void)b_offset;
    (void)ldb;
    (void)strideB;
    (void)beta;
    (void)C;
    (void)c_offset;
    (void)ldc;
    (void)strideC;
    (void)batch_count;

    MIOPEN_THROW("No GEMM backend");
#endif
}

// y = w * Im2Col(x)
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvFwd(const TensorDescriptor& wDesc,
                             const TensorDescriptor& xDesc,
                             const TensorDescriptor& yDesc)
{
    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    float alpha, beta;

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

    isColMajor = false;
    transA     = false;
    transB     = false;
    m          = wei_n;
    n          = out_h * out_w;
    k          = in_c * wei_h * wei_w;
    lda        = k;
    ldb        = n;
    ldc        = n;
    alpha      = 1.;
    beta       = 0.;

    return std::make_tuple(isColMajor, transA, transB, m, n, k, lda, ldb, ldc, alpha, beta);
}

// dx = Col2Im(transpose(w) * dy)
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvBwdData(const TensorDescriptor& wDesc,
                                 const TensorDescriptor& dyDesc,
                                 const TensorDescriptor& dxDesc)
{
    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    float alpha, beta;

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

    isColMajor = false;
    transA     = true;
    transB     = false;
    m          = in_c * wei_h * wei_w;
    n          = out_h * out_w;
    k          = wei_n;
    lda        = m;
    ldb        = n;
    ldc        = n;
    alpha      = 1.;
    beta       = 0.;

    return std::make_tuple(isColMajor, transA, transB, m, n, k, lda, ldb, ldc, alpha, beta);
}

// dw = dy * transpose(Im2Col(x))
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvBwdWeight(const TensorDescriptor& dyDesc,
                                   const TensorDescriptor& xDesc,
                                   const TensorDescriptor& dwDesc)
{
    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    float alpha, beta;

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

    isColMajor = false;
    transA     = false;
    transB     = true;
    m          = wei_n;
    n          = in_c * wei_h * wei_w;
    k          = out_h * out_w;
    lda        = k;
    ldb        = k;
    ldc        = n;
    alpha      = 1.;
    beta       = 1.;

    return std::make_tuple(isColMajor, transA, transB, m, n, k, lda, ldb, ldc, alpha, beta);
}

// y = CNHW2NCHW(w * NCHW2CNHW(x))
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvCNHWFwd(const TensorDescriptor& wDesc,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& yDesc)
{
    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    float alpha, beta;

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

    isColMajor = false;
    transA     = false;
    transB     = false;
    m          = wei_n;
    n          = in_n * out_h * out_w;
    k          = in_c;
    lda        = k;
    ldb        = n;
    ldc        = n;
    alpha      = 1.;
    beta       = 0.;

    return std::make_tuple(isColMajor, transA, transB, m, n, k, lda, ldb, ldc, alpha, beta);
}

// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvCNHWBwdData(const TensorDescriptor& wDesc,
                                     const TensorDescriptor& dyDesc,
                                     const TensorDescriptor& dxDesc)
{
    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    float alpha, beta;

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

    isColMajor = false;
    transA     = true;
    transB     = false;
    m          = in_c;
    n          = in_n * out_h * out_w;
    k          = wei_n;
    lda        = m;
    ldb        = n;
    ldc        = n;
    alpha      = 1.;
    beta       = 0.;

    return std::make_tuple(isColMajor, transA, transB, m, n, k, lda, ldb, ldc, alpha, beta);
}

// y = w * x
std::tuple<bool,
           bool,
           bool,
           int,
           int,
           int,
           int,
           int,
           int,
           long long int,
           long long int,
           long long int,
           int,
           float,
           float>
CreateGemmStridedBatchedDescriptionConv1x1Fwd(const TensorDescriptor& wDesc,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& yDesc)
{
    (void)yDesc;

    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    long long int strideA, strideB, strideC;
    int batch_count;
    float alpha, beta;

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

    isColMajor  = false;
    transA      = false;
    transB      = false;
    m           = wei_n;
    n           = in_h * in_w;
    k           = in_c;
    lda         = k;
    ldb         = n;
    ldc         = n;
    strideA     = 0;
    strideB     = k * n;
    strideC     = m * n;
    batch_count = in_n;
    alpha       = 1.;
    beta        = 0.;

    return std::make_tuple(isColMajor,
                           transA,
                           transB,
                           m,
                           n,
                           k,
                           lda,
                           ldb,
                           ldc,
                           strideA,
                           strideB,
                           strideC,
                           batch_count,
                           alpha,
                           beta);
}

// dx = transpose(w) * dy
std::tuple<bool,
           bool,
           bool,
           int,
           int,
           int,
           int,
           int,
           int,
           long long int,
           long long int,
           long long int,
           int,
           float,
           float>
CreateGemmStridedBatchedDescriptionConv1x1BwdData(const TensorDescriptor& wDesc,
                                                  const TensorDescriptor& dyDesc,
                                                  const TensorDescriptor& dxDesc)
{
    (void)dyDesc;

    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    long long int strideA, strideB, strideC;
    int batch_count;
    float alpha, beta;

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

    isColMajor  = false;
    transA      = true;
    transB      = false;
    m           = in_c;
    n           = in_h * in_w;
    k           = wei_n;
    lda         = m;
    ldb         = n;
    ldc         = n;
    strideA     = 0;
    strideB     = k * n;
    strideC     = m * n;
    batch_count = in_n;
    alpha       = 1.;
    beta        = 0;

    return std::make_tuple(isColMajor,
                           transA,
                           transB,
                           m,
                           n,
                           k,
                           lda,
                           ldb,
                           ldc,
                           strideA,
                           strideB,
                           strideC,
                           batch_count,
                           alpha,
                           beta);
}

} // namespace miopen
