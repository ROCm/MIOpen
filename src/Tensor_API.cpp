#include "Tensor.hpp"
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreateTensorDescriptor(
		mlopenTensorDescriptor_t *tensorDesc) {
	
	if(tensorDesc == nullptr) {
		return mlopenStatusBadParm;
	}

	try {
		*tensorDesc = new mlopenTensorDescriptor();
	} catch (mlopenStatus_t status) {
		return status;
	}

	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenSet4dTensorDescriptor(
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t dataType,
		int n,
		int c,
		int h,
		int w) {
	
	try{
		std::initializer_list<int> lens = {n, c, h, w};
		*tensorDesc = mlopenTensorDescriptor(dataType, lens.begin(), 4);

	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;
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
	
	try{
		*dataType = tensorDesc->GetType();
		std::tie(*n, *c, *h, *w) = tie4(tensorDesc->GetLengths());
		std::tie(*nStride, *cStride, *hStride, *wStride) = tie4(tensorDesc->GetStrides());
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenGet4dTensorDescriptorLengths(
		mlopenTensorDescriptor_t tensorDesc,
		int *n,
		int *c,
		int *h,
		int *w) {
	
	try{
		std::tie(*n, *c, *h, *w) = tie4(tensorDesc->GetLengths());
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenGet4dTensorDescriptorStrides(
		mlopenTensorDescriptor_t tensorDesc,
		int *nStride,
		int *cStride,
		int *hStride,
		int *wStride) {
	
	try{
		std::tie(*nStride, *cStride, *hStride, *wStride) = tie4(tensorDesc->GetStrides());
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenSetTensorDescriptor(
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t dataType,
		int nbDims,
		int *dimsA,
		int *stridesA) {

	try{
		if (stridesA == nullptr) {
			*tensorDesc = mlopenTensorDescriptor(dataType, dimsA, nbDims);
		} else {
			*tensorDesc = mlopenTensorDescriptor(dataType, dimsA, stridesA, nbDims);
		}
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;

}

extern "C" 
mlopenStatus_t mlopenGetTensorDescriptorSize(mlopenTensorDescriptor_t tensorDesc, int* size) {
	try {
		*size = tensorDesc->GetSize();
	} catch (mlopenStatus_t success) {
		return success;
	}
	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenGetTensorDescriptor(
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t *dataType,
		int *dimsA,
		int *stridesA) {

	try{
		if (dataType != nullptr) {
			*dataType = tensorDesc->GetType();
		}
		if (dimsA != nullptr) {
			std::copy(tensorDesc->GetLengths().begin(), tensorDesc->GetLengths().end(), dimsA);
		}
		if (stridesA != nullptr) {
			std::copy(tensorDesc->GetStrides().begin(), tensorDesc->GetStrides().end(), stridesA);
		}
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;

}

extern "C"
mlopenStatus_t mlopenDestroyTensorDescriptor(mlopenTensorDescriptor_t tensorDesc) {
	try {
		delete tensorDesc;
	} catch (mlopenStatus_t status) {
		return status;
	}
	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenTransformTensor(mlopenHandle_t handle,
		const void						*alpha,
		const mlopenTensorDescriptor_t	xDesc,
		const void						*x,
		const void						*beta,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y) {

	return yDesc->TransformTensor(handle, 
			alpha,
			xDesc,
			DataCast(x),
			beta,
			DataCast(y));
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

	return cDesc->OpTensor(handle,
			tensorOp,
			alpha1,
			aDesc,
			DataCast(A),
			alpha2,
			bDesc,
			DataCast(B),
			beta,
			DataCast(C));

}

extern "C"
mlopenStatus_t mlopenSetTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*valuePtr) {
	
	return yDesc->SetTensor(handle,
			DataCast(y),
			valuePtr);

}

extern "C"
mlopenStatus_t mlopenScaleTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*alpha) {

	return yDesc->ScaleTensor(handle,
			DataCast(y),
			alpha);

}
