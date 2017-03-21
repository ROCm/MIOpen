#include <mlopen/tensor.hpp>
#include <mlopen/tensor_ops.hpp>
#include <mlopen/errors.hpp>
#include <mlopen/logger.hpp>
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreateTensorDescriptor(
		mlopenTensorDescriptor_t *tensorDesc) {
	MLOPEN_LOG_FUNCTION(tensorDesc);
	return mlopen::try_([&] {
		mlopen::deref(tensorDesc) = new mlopen::TensorDescriptor();
	});
}

extern "C"
mlopenStatus_t mlopenSet4dTensorDescriptor(
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t dataType,
		int n,
		int c,
		int h,
		int w) {

	MLOPEN_LOG_FUNCTION(tensorDesc, dataType, n, c, h, w);
	return mlopen::try_([&] {
		std::initializer_list<int> lens = {n, c, h, w};
		mlopen::deref(tensorDesc) = mlopen::TensorDescriptor(dataType, lens.begin(), 4);
	});
}

extern "C"
mlopenStatus_t mlopenGet4dTensorDescriptor(
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t *dataType,
		int *n,
		int *c,
		int *h,
		int *w,
		int *nStride,
		int *cStride,
		int *hStride,
		int *wStride) {

	MLOPEN_LOG_FUNCTION(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
	return mlopen::try_([&] {
		mlopen::deref(dataType) = mlopen::deref(tensorDesc).GetType();
		mlopen::tie_deref(n, c, h, w) = mlopen::tie4(mlopen::deref(tensorDesc).GetLengths());
		mlopen::tie_deref(nStride, cStride, hStride, wStride) = mlopen::tie4(mlopen::deref(tensorDesc).GetStrides());
	});
}

// Internal API
// MD: This should not be reqired to be exported. Temporary hack
MLOPEN_EXPORT mlopenStatus_t mlopenGet4dTensorDescriptorLengths(
		mlopenTensorDescriptor_t tensorDesc,
		int *n,
		int *c,
		int *h,
		int *w) {

	MLOPEN_LOG_FUNCTION(tensorDesc, n, c, h, w);
	return mlopen::try_([&] {
		mlopen::tie_deref(n, c, h, w) = mlopen::tie4(mlopen::deref(tensorDesc).GetLengths());
	});
}


// Internal API
MLOPEN_EXPORT mlopenStatus_t mlopenGet4dTensorDescriptorStrides(
		mlopenTensorDescriptor_t tensorDesc,
		int *nStride,
		int *cStride,
		int *hStride,
		int *wStride) {

	MLOPEN_LOG_FUNCTION(tensorDesc, nStride, cStride, hStride, wStride);
	return mlopen::try_([&] {
		mlopen::tie_deref(nStride, cStride, hStride, wStride) = mlopen::tie4(mlopen::deref(tensorDesc).GetStrides());
	});
}

extern "C"
mlopenStatus_t mlopenSetTensorDescriptor(
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t dataType,
		int nbDims,
		int *dimsA,
		int *stridesA) {

	MLOPEN_LOG_FUNCTION(tensorDesc, dataType, nbDims, dimsA, stridesA);
	return mlopen::try_([&] {
		if (stridesA == nullptr) {
			mlopen::deref(tensorDesc) = mlopen::TensorDescriptor(dataType, dimsA, nbDims);
		} else {
			mlopen::deref(tensorDesc) = mlopen::TensorDescriptor(dataType, dimsA, stridesA, nbDims);
		}
	});
}

// Internal API
int mlopenGetTensorDescriptorElementSize(mlopenTensorDescriptor_t tensorDesc) {
	return mlopen::deref(tensorDesc).GetElementSize();
}

extern "C" 
mlopenStatus_t mlopenGetTensorDescriptorSize(mlopenTensorDescriptor_t tensorDesc, int* size) {
	MLOPEN_LOG_FUNCTION(tensorDesc, size);
	return mlopen::try_([&] {
		mlopen::deref(size) = mlopen::deref(tensorDesc).GetSize();
	});
}

extern "C"
mlopenStatus_t mlopenGetTensorDescriptor(
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t *dataType,
		int *dimsA,
		int *stridesA) {

	MLOPEN_LOG_FUNCTION(tensorDesc, dataType, dimsA, stridesA);
	return mlopen::try_([&] {
		if (dataType != nullptr) {
			*dataType = mlopen::deref(tensorDesc).GetType();
		}
		if (dimsA != nullptr) {
			std::copy(mlopen::deref(tensorDesc).GetLengths().begin(), mlopen::deref(tensorDesc).GetLengths().end(), dimsA);
		}
		if (stridesA != nullptr) {
			std::copy(mlopen::deref(tensorDesc).GetStrides().begin(), mlopen::deref(tensorDesc).GetStrides().end(), stridesA);
		}
	});

}

extern "C"
mlopenStatus_t mlopenDestroyTensorDescriptor(mlopenTensorDescriptor_t tensorDesc) {
	MLOPEN_LOG_FUNCTION(tensorDesc);
	return mlopen::try_([&] {
		mlopen_destroy_object(tensorDesc);
	});
}

extern "C"
mlopenStatus_t mlopenTransformTensor(mlopenHandle_t handle,
		const void						*alpha,
		const mlopenTensorDescriptor_t	xDesc,
		const void						*x,
		const void						*beta,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y) {

	MLOPEN_LOG_FUNCTION(alpha, xDesc, x, beta, yDesc, y);
	return mlopen::try_([&] {
		TransformTensor(mlopen::deref(handle), 
				alpha,
				mlopen::deref(xDesc),
				DataCast(x),
				beta,
				mlopen::deref(yDesc),
				DataCast(y));
	});

}

extern "C"
mlopenStatus_t mlopenAddTensor(mlopenHandle_t handle,
		const void						*alpha,
		const mlopenTensorDescriptor_t	aDesc,
		const void						*A,
		const void						*beta,
		const mlopenTensorDescriptor_t	cDesc,
		void							*C) {

	MLOPEN_LOG_FUNCTION(alpha, aDesc, A, beta, cDesc, C);
	return mlopen::try_([&] {
		AddTensor(mlopen::deref(handle), 
				alpha,
				mlopen::deref(aDesc),
				DataCast(A),
				beta,
				mlopen::deref(cDesc),
				DataCast(C));
	});

}

extern "C"
mlopenStatus_t mlopenOpTensor(mlopenHandle_t handle,
		mlopenTensorOp_t				tensorOp,
		const void						*alpha1,
		const mlopenTensorDescriptor_t	aDesc,
		const void						*A,
		const void						*alpha2,
		const mlopenTensorDescriptor_t	bDesc,
		const void						*B,
		const void						*beta,
		const mlopenTensorDescriptor_t	cDesc,
		void							*C) {

	MLOPEN_LOG_FUNCTION(tensorOp, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
	return mlopen::try_([&] {
		OpTensor(mlopen::deref(handle),
				tensorOp,
				alpha1,
				mlopen::deref(aDesc),
				DataCast(A),
				alpha2,
				mlopen::deref(bDesc),
				DataCast(B),
				beta,
				mlopen::deref(cDesc),
				DataCast(C));
	});


}

extern "C"
mlopenStatus_t mlopenSetTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*valuePtr) {

	MLOPEN_LOG_FUNCTION(yDesc, y, valuePtr);
	return mlopen::try_([&] {
		mlopen::deref(yDesc).SetTensor(mlopen::deref(handle),
				DataCast(y),
				valuePtr);
	});
	

}

extern "C"
mlopenStatus_t mlopenScaleTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*alpha) {

	MLOPEN_LOG_FUNCTION(yDesc, y, alpha);
	return mlopen::try_([&] {
		mlopen::deref(yDesc).ScaleTensor(mlopen::deref(handle),
				DataCast(y),
				alpha);
	});


}
