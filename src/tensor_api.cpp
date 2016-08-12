#include <mlopen/tensor.hpp>
#include <mlopen/errors.hpp>
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreateTensorDescriptor(
		mlopenTensorDescriptor_t *tensorDesc) {

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

	return mlopen::try_([&] {
		return mlopen::deref(yDesc).TransformTensor(mlopen::deref(handle), 
				alpha,
				mlopen::deref(xDesc),
				DataCast(x),
				beta,
				DataCast(y));
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

	return mlopen::try_([&] {
		return mlopen::deref(cDesc).OpTensor(mlopen::deref(handle),
				tensorOp,
				alpha1,
				mlopen::deref(aDesc),
				DataCast(A),
				alpha2,
				mlopen::deref(bDesc),
				DataCast(B),
				beta,
				DataCast(C));
	});


}

extern "C"
mlopenStatus_t mlopenSetTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*valuePtr) {

	return mlopen::try_([&] {
		return mlopen::deref(yDesc).SetTensor(mlopen::deref(handle),
				DataCast(y),
				valuePtr);
	});
	

}

extern "C"
mlopenStatus_t mlopenScaleTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*alpha) {

	return mlopen::try_([&] {
		return mlopen::deref(yDesc).ScaleTensor(mlopen::deref(handle),
				DataCast(y),
				alpha);
	});


}
