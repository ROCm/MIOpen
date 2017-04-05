#include <miopen/gemm.hpp>
#include <miopen/errors.hpp>

extern "C"
miopenStatus_t miopenGemm(
        miopenHandle_t          handle,
        bool                    isDataColMajor,
        bool                    transA, 
        bool                    transB, 
        int M, int N, int K, 
        const void *alpha, 
        const void *A, int lda, 
        const void *B, int ldb, 
        const void *beta, 
        void *C, int ldc )
{
    return miopen::try_([&] {
        miopen::GemmGeometry gg = miopen::CreateMIOpenGemmGeometry(M, N, K,
            lda, ldb, ldc,
            transA, transB,
            isDataColMajor,
            *(static_cast<const float*>(alpha)), *(static_cast<const float*>(beta)));

        gg.FindSolution(.003, miopen::deref(handle),
            DataCast(A),
            DataCast(B),
            DataCast(C),
            false);

        gg.RunGemm(miopen::deref(handle),
            DataCast(A),
            DataCast(B),
            DataCast(C),
            0, 0, 0);
    });
}

