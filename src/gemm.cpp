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
#include <miopen/gemm.hpp>
#include <miopen/handle.hpp>

namespace miopen {
// for debugging
#if 1
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

#if MIOPEN_USE_MIOPENGEMM
GemmGeometry CreateMIOpenGemmGeometry(bool isColMajor,
                                      bool transA,
                                      bool transB,
                                      int m,
                                      int n,
                                      int k,
                                      int lda,
                                      int ldb,
                                      int ldc,
                                      float alpha,
                                      float beta)
{
    MIOpenGEMM::Geometry tgg{};

    // Assuming we are using miopengemm as only col major
    // Therefore, if the user provides data in col. major
    // then no transformations are requrired and vice versa
    if(isColMajor)
    {
        tgg = MIOpenGEMM::Geometry(
            true, transA, transB, false, lda, ldb, ldc, m, n, k, 0, 'f'); // jn : added 0 for no
                                                                          // workspace,
                                                                          // 'f' for single prec.

        return GemmGeometry{"miopenGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(
            true, transB, transA, false, ldb, lda, ldc, n, m, k, 0, 'f'); // jn : added 0 for no
                                                                          // workspace,
                                                                          // 'f' for single prec.

        return GemmGeometry{"miopenGEMM", alpha, beta, tgg};
    }
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
    std::cout << std::endl << __func__ << ": going to call rocblas" << std::endl;

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
    std::cout << __func__ << ": going to call miopengemm" << std::endl;

    // do row-to-column major conversion here,
    //   so CreateMIOpenGemmGeometry would not need to do it
    if(!isColMajor)
    {
        isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(transA, transB);
        std::swap(m, n);
        std::swap(lda, ldb);
    }

    GemmGeometry gg = CreateMIOpenGemmGeometry(true,
                                               transA,
                                               transB,
                                               m,
                                               n,
                                               k,
                                               lda,
                                               ldb,
                                               ldc,
                                               *(static_cast<const float*>(alpha)),
                                               *(static_cast<const float*>(beta)));

    gg.RunGemmSimple(handle, A, B, C, a_offset, b_offset, c_offset);
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
    std::cout << std::endl << __func__ << ": going to call rocblas" << std::endl;

#if 1 // debug: output GEMM description
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
#endif

#if 1
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

    std::cout << __func__ << ": alpha_local " << alpha_local << ", beta_local " << beta_local
              << std::endl;

#if 1 // debug: output GEMM description
    {
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
    }
#endif

#if 1 // debug: output A, B, C
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
        // std::cout << __func__ << ": A before call rocblas: " << tmp_a << std::endl;
        // std::cout << __func__ << ": B before call rocblas: " << tmp_b << std::endl;
        // std::cout << __func__ << ": C before call rocblas: " << tmp_c << std::endl;

        float sum_c = std::accumulate(tmp_c.begin(), tmp_c.end(), float(0), std::plus<float>());
        std::cout << __func__ << ": sum_c before call rocblas" << sum_c << std::endl;
    }
#endif // debug: output A, B, C

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

#if 1 // debug: output A, B, C
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
        // std::cout << __func__ << ": A before call rocblas: " << tmp_a << std::endl;
        // std::cout << __func__ << ": B before call rocblas: " << tmp_b << std::endl;
        // std::cout << __func__ << ": C before call rocblas: " << tmp_c << std::endl;

        float sum_c = std::accumulate(tmp_c.begin(), tmp_c.end(), float(0), std::plus<float>());
        std::cout << __func__ << ": sum_c after call rocblas" << sum_c << std::endl;
    }
#endif // debug: output A, B, C

#else
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

    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": yDesc: " << yDesc << std::endl;

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

    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ":  wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": dxDesc: " << dxDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;

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

    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ": dwDesc: " << dwDesc << std::endl;
    std::cout << __func__ << ":  xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;

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

    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": yDesc: " << yDesc << std::endl;

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

    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ":  wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": dxDesc: " << dxDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;

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
    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    long long int strideA, strideB, strideC;
    int batch_count;
    float alpha, beta;

    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": yDesc: " << yDesc << std::endl;

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
    bool isColMajor, transA, transB;
    int m, n, k, lda, ldb, ldc;
    long long int strideA, strideB, strideC;
    int batch_count;
    float alpha, beta;

    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": dxDesc: " << dxDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;

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

#if MIOPEN_USE_MIOPENGEMM
namespace miopen {

GemmGeometry CreateGemmGeometryTranBwdData(const TensorDescriptor& dyDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& dxDesc,
                                           bool isDataColMajor,
                                           std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_c, wei_n, wei_h, wei_w;
    std::tie(wei_c, wei_n, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int K       = wei_n * wei_h * wei_w;
    int M       = wei_c;
    int N       = in_h * in_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = false;
    bool tB     = false;
    bool tC     = false;
    int lda     = K;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenTransposeBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenTransposeBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvBwdData(const TensorDescriptor& dyDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& dxDesc,
                                           bool isDataColMajor,
                                           std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int K       = wei_n;
    int N       = out_h * out_w;
    int M       = in_c * wei_h * wei_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = true;
    bool tB     = false;
    bool tC     = false;
    int lda     = M;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvBwdDataCNHW(const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& wDesc,
                                               const TensorDescriptor& dxDesc,
                                               bool isDataColMajor,
                                               std::string& network_config)
{
    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int K       = wei_n;
    int N       = in_n * out_h * out_w;
    int M       = in_c;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = true;
    bool tB     = false;
    bool tC     = false;
    int lda     = M;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvBwdWeights(const TensorDescriptor& dyDesc,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& dwDesc,
                                              bool isDataColMajor,
                                              std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int N       = in_c * wei_h * wei_w;
    int M       = wei_n;
    int K       = out_h * out_w;
    bool tA     = false;
    bool tB     = true;
    bool tC     = false;
    int lda     = K;
    int ldb     = K;
    int ldc     = N;
    float alpha = 1.0;
    float beta  = 1.0;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdWeightsAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdWeightsAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvFwd(const TensorDescriptor& xDesc,
                                       const TensorDescriptor& wDesc,
                                       const TensorDescriptor& yDesc,
                                       bool isDataColMajor,
                                       std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    // GEMM
    int K       = in_c * wei_h * wei_w;
    int M       = wei_n;
    int N       = out_h * out_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = false;
    bool tB     = false;
    bool tC     = false;
    int lda     = K;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvFwdCNHW(const TensorDescriptor& xDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& yDesc,
                                           bool isDataColMajor,
                                           std::string& network_config)
{
    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    // GEMM
    int K       = in_c;
    int M       = wei_n;
    int N       = in_n * out_h * out_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = false;
    bool tB     = false;
    bool tC     = false;
    int lda     = K;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry GetGemmGeometry(Handle& handle, std::string algorithm_name, std::string network_config)
{
    auto gemm_iterator = handle.geo_map.find(std::make_pair(algorithm_name, network_config));
    if(gemm_iterator != handle.geo_map.end())
    {
        return *gemm_iterator->second;
    }
    else
    {
        MIOPEN_THROW("looking for gemm kernel (does not exist): " + algorithm_name + ", " +
                     network_config);
    }
}

GemmGeometry CreateGemmGeometryRNN(int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   float beta,
                                   bool tA,
                                   bool tB,
                                   bool tC,
                                   int lda,
                                   int ldb,
                                   int ldc,
                                   bool isDataColMajor,
                                   std::string& network_config)
{
    // GEMM
    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    (void)isDataColMajor;

    tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
    gg  = GemmGeometry{"miopenRNNAlgoGEMM", alpha, beta, tgg};

    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry ScanGemmGeometryRNN(Handle& handle,
                                 ConstData_t A,
                                 ConstData_t B,
                                 Data_t C,
                                 int M,
                                 int N,
                                 int K,
                                 float alpha,
                                 float beta,
                                 bool tA,
                                 bool tB,
                                 bool tC,
                                 int lda,
                                 int ldb,
                                 int ldc,
                                 bool isDataColMajor,
                                 std::string& network_config,
                                 float timeout)
{

    auto gg = CreateGemmGeometryRNN(
        M, N, K, alpha, beta, tA, tB, tC, lda, ldb, ldc, isDataColMajor, network_config);

    auto gemm_iterator = handle.geo_map.find(std::make_pair("miopenRNNAlgoGEMM", network_config));
    if(gemm_iterator != handle.geo_map.end())
    {
        gg = *gemm_iterator->second;
    }
    else
    {
        gg.FindSolution(timeout, handle, A, B, C, false);
    }

    return gg;
}

void RunGemmGeometryRNN(Handle& handle,
                        ConstData_t A,
                        ConstData_t B,
                        Data_t C,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        float beta,
                        bool tA,
                        bool tB,
                        bool tC,
                        int lda,
                        int ldb,
                        int ldc,
                        int a_offset,
                        int b_offset,
                        int c_offset,
                        bool isDataColMajor,
                        std::string& network_config,
                        float timeout)
{

    auto gg = ScanGemmGeometryRNN(handle,
                                  A,
                                  B,
                                  C,
                                  M,
                                  N,
                                  K,
                                  alpha,
                                  beta,
                                  tA,
                                  tB,
                                  tC,
                                  lda,
                                  ldb,
                                  ldc,
                                  isDataColMajor,
                                  network_config,
                                  timeout);

    gg.RunGemm(handle, A, B, C, a_offset, b_offset, c_offset);
}

} // namespace miopen
#endif // MIOPEN_USE_MIOPENGEMM
