#include "Tensor.hpp"

mlopenTensorDescriptor::mlopenTensorDescriptor() : _dims(4) {

	// Setting the default dims to 4
	_dimA = std::vector<int> (4,0);
	_strideA = std::vector<int> (4, 1);
	_tensorHandle = nullptr;
}

mlopenStatus_t mlopenTensorDescriptor::SetTensorHandle(mlopenHandle_t handle) {
	_tensorHandle = handle;
	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::Set4Dims(int n,
		int c,
		int h,
		int w) {
	_dimA[0] = n;
	_dimA[1] = c;
	_dimA[2] = h;
	_dimA[3] = w;

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::Set4Strides(int nStride,
		int cStride,
		int hStride,
		int wStride) {
	_strideA[0] = nStride;
	_strideA[1] = cStride;
	_strideA[2] = hStride;
	_strideA[3] = wStride;

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::SetNDims(int dims, 
		int *dimsA) {

	for(auto i = 0; i < dims; i++) {
		_dimA[i] = dimsA[i];
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::SetNStrides(int dims,
		int *stridesA) {

	for(auto i = 0; i < dims; i++) {
		_strideA[i] = stridesA[i];
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::SetDims(int dims) {
	_dims = dims;

	_dimA.resize(dims);
	_dimA.clear();

	_strideA.resize(dims);
	_strideA.clear();

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::Get4Dims(int *n,
		int *c,
		int *h, 
		int *w) {
	*n = _dimA[0];
	*c = _dimA[1];
	*h = _dimA[2];
	*w = _dimA[3];

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::Get4Strides(int *nStride,
		int *cStride,
		int *hStride, 
		int *wStride) {
	*nStride = _strideA[0];
	*cStride = _strideA[1];
	*hStride = _strideA[2];
	*wStride = _strideA[3];

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::GetNDims(int *dimsA) {
	for(auto i = 0; i <_dims; i++) {
		dimsA[i] = _dimA[i];
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::GetNStrides(int *stridesA) {
	for(auto i = 0; i < _dims; i++) {
		stridesA[i] = _strideA[i];
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::GetDims(int *dims) {
	*dims = _dims;

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::GetDataType(mlopenDataType_t *dataType) {
	*dataType = _dataType;

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::GetTensorHandle(mlopenHandle_t handle) {
	handle = _tensorHandle;
	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::TransformTensor(mlopenHandle_t handle,
		const void *alpha,
		const mlopenTensorDescriptor_t srcTensorDesc,
		const void *srcTensor,
		const void *beta,
		void *dstTensor) {

	printf("To be implemented\n");

	// Check if the source and dest tensors have the same dims .. strides can be different
	// Check that output tensors do not overlap .. output tensors cannot be transformed in place .. no aliasing
	// Implement conversion of unsupported tensor to a supported one
	// Launch kernels using the handle
	// Needs to have a kernel cache?
	// If beta[0] = 0 then just a memcopy with scaled alpha[0]?
}
