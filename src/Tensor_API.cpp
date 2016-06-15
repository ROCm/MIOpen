#include "Tensor.hpp"

extern "C"
mlopenStatus_t mlopenCreateTensorDescriptor(mlopenHandle_t handle,
		mlopenTensorDescriptor_t *tensorDesc) {
	
	if(tensorDesc == nullptr) {
		return mlopenStatusBadParm;
	}

	try {
		*tensorDesc = new mlopenTensorDescriptor();
	} catch (mlopenStatus_t status) {
		return status;
	}

	(*tensorDesc)->SetTensorHandle(handle);	

	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenInit4dTensorDescriptor(mlopenHandle_t handle,
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t dataType,
		int n,
		int c,
		int h,
		int w,
		int nStride,
		int cStride,
		int hStride,
		int wStride) {
	
	try{
		tensorDesc->Set4Dims(n, c, h, w);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try {
		tensorDesc->Set4Strides(nStride,
			cStride,
			hStride,
			wStride);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->SetDataType(dataType);
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenGet4dTensorDescriptor(mlopenHandle_t handle,
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
		tensorDesc->Get4Dims(n, c, h, w);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try {
		tensorDesc->Get4Strides(nStride,
			cStride,
			hStride,
			wStride);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->GetDataType(*dataType);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->GetTensorHandle(handle);
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenInitNdTensorDescriptor(mlopenHandle_t handle,
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t dataType,
		int nbDims,
		int *dimsA,
		int *stridesA) {

	try{
		tensorDesc->SetDims(nbDims);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->SetNDims(nbDims, dimsA);
	} catch (mlopenStatus_t success) {
		return success;
	}
	
	try{
		tensorDesc->SetNStrides(nbDims, stridesA);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->SetDataType(dataType);
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;

}

extern "C"
mlopenStatus_t mlopenGetNdTensorDescriptor(mlopenHandle_t handle,
		mlopenTensorDescriptor_t tensorDesc,
		mlopenDataType_t *dataType,
		int *nbDims,
		int *dimsA,
		int *stridesA) {

	try{
		tensorDesc->GetDims(*nbDims);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->GetNDims(dimsA);
	} catch (mlopenStatus_t success) {
		return success;
	}
	
	try{
		tensorDesc->GetNStrides(stridesA);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->GetDataType(*dataType);
	} catch (mlopenStatus_t success) {
		return success;
	}

	try{
		tensorDesc->GetTensorHandle(handle);
	} catch (mlopenStatus_t success) {
		return success;
	}

	return mlopenStatusSuccess;

}

extern "C"
mlopenStatus_t mlopenDestroyTensor(mlopenTensorDescriptor_t tensorDesc) {
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
		const mlopenTensorDescriptor_t	 yDesc,
		void							*y) {

	// Calling the transform function on the destination tensor
	return yDesc->TransformTensor(handle, 
			alpha,
			xDesc,
			x,
			beta,
			y);
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

	// Calling the transform function on the destination tensor
	return cDesc->OpTensor(handle,
			tensorOp,
			alpha1,
			aDesc,
			A,
			alpha2,
			bDesc,
			B,
			beta,
			C);
}

extern "C"
mlopenStatus_t mlopenSetTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*valuePtr) {
	
	// Calling the transform function on the destination tensor
	return yDesc->SetTensor(handle,
			y,
			valuePtr);
	
}

extern "C"
mlopenStatus_t mlopenScaleTensor(mlopenHandle_t handle,
		const mlopenTensorDescriptor_t	yDesc,
		void							*y,
		const void						*alpha) {

	// Calling the transform function on the destination tensor
	return yDesc->ScaleTensor(handle,
			y,
			alpha);

}








