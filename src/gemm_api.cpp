#include <mlopen/gemm.hpp>
#include <mlopen/errors.hpp>

extern "C"
mlopenStatus_t mlopenGemm(
		mlopenHandle_t			handle,
		bool					isDataColMajor,
		bool					transA, 
		bool					transB, 
		int M, int N, int K, 
		const void *alpha, 
		const void *A, int lda, 
		const void *B, int ldb, 
		const void *beta, 
		void *C, int ldc )
{
	return mlopen::try_([&] {
		mlopen::GemmGeometry gg = mlopen::CreateMLOpenGemmGeometry(M, N, K,
			lda, ldb, ldc,
			transA, transB,
			isDataColMajor,
			*((float*)(alpha)), *((float*)(beta)));

		gg.FindSolution(.003, mlopen::deref(handle),
			DataCast(A),
			DataCast(B),
			DataCast(C),
			false);

		gg.RunGemm(mlopen::deref(handle),
			DataCast(A),
			DataCast(B),
			DataCast(C),
			0, 0, 0);
	});
}

