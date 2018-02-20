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
#include <miopen/errors.hpp>
#include <miopen/gemm.hpp>

#if MIOPEN_USE_ROCBLAS
#include <miopen/handle.hpp>
#include <miopen/manage_ptr.hpp>
#include <rocblas.h>

using rocblas_handle_ptr = MIOPEN_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);

rocblas_handle_ptr create_rocblas_handle_ptr(miopen::Handle& h)
{
    rocblas_handle x = nullptr;
    rocblas_create_handle(&x);
    auto result = rocblas_handle_ptr{x};
    rocblas_set_stream(result.get(), h.GetStream());
    return result;
}
#endif

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
                          int find)
{
#if MIOPEN_USE_ROCBLAS
    (void)isDataColMajor;
    (void)find;
    static rocblas_handle_ptr rhandle = create_rocblas_handle_ptr(miopen::deref(handle));
    float alpha_local                 = *static_cast<const float*>(alpha);
    float beta_local                  = *static_cast<const float*>(beta);
    rocblas_sgemm(rhandle.get(),
                  transA ? rocblas_operation_transpose : rocblas_operation_none,
                  transB ? rocblas_operation_transpose : rocblas_operation_none,
                  M,
                  N,
                  K,
                  &alpha_local,
                  static_cast<const float*>(A),
                  lda,
                  static_cast<const float*>(B),
                  ldb,
                  &beta_local,
                  static_cast<float*>(C),
                  ldc);
    return miopenStatusSuccess;
#else

    // JN make column major
    if(!isDataColMajor)
    {
        std::swap(transA, transB);
        std::swap(M, N);
        std::swap(lda, ldb);
        std::swap(A, B);
        isDataColMajor = true;
    }

    return miopen::try_([&] {
        miopen::GemmGeometry gg =
            miopen::CreateMIOpenGemmGeometry(M,
                                             N,
                                             K,
                                             lda,
                                             ldb,
                                             ldc,
                                             transA,
                                             transB,
                                             isDataColMajor,
                                             *(static_cast<const float*>(alpha)),
                                             *(static_cast<const float*>(beta)));

        if(find)
        {
            gg.FindSolution(
                .003, miopen::deref(handle), DataCast(A), DataCast(B), DataCast(C), false);

            gg.RunGemm(miopen::deref(handle), DataCast(A), DataCast(B), DataCast(C), 0, 0, 0);
        }
        else
            gg.RunGemm(miopen::deref(handle), DataCast(A), DataCast(B), DataCast(C), 0, 0, 0);
    });
#endif
}
