#include "Tensor.hpp"
#include <string>
#include <algorithm>

mlopenTensorDescriptor::mlopenTensorDescriptor() : _dims(4) {

	printf("In Tensor Descriptor Ctor\n");
	// Setting the default dims to 4
	_dimA = std::vector<int> (4,0);
	_strideA = std::vector<int> (4, 1);
	_tensorHandle = nullptr;
}

void mlopenTensorDescriptor::CalculateStrides()
{
	_strideA.clear();
	_strideA.resize(_dimA.size(), 0);
	_strideA.back() = 1;
	std::partial_sum(_dimA.rbegin(), _dimA.rend(), _strideA.rbegin()+1, std::multiplies<int>());
}

mlopenStatus_t mlopenTensorDescriptor::SetTensorHandle(mlopenHandle_t handle) {
	_tensorHandle = handle;
	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::Set4Dims(int n,
		int c,
		int h,
		int w) {

	printf("In Set4Dims\n");
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

mlopenStatus_t mlopenTensorDescriptor::SetDataType(mlopenDataType_t dataType) {
	_dataType = dataType;

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

mlopenStatus_t mlopenTensorDescriptor::GetDims(int &dims) {
	dims = _dims;

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::GetDataType(mlopenDataType_t &dataType) {
	dataType = _dataType;

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::GetTensorHandle(mlopenHandle_t handle) {
	handle = _tensorHandle;
	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::_CheckTensorDims(mlopenTensorDescriptor_t srcTensorDesc) {

	if(srcTensorDesc->_dims != this->_dims) {
		return mlopenStatusBadParm;
	}

	std::vector<int> srcDims(srcTensorDesc->_dimA); 
	std::vector<int> dstDims(this->_dimA);

	int dims = srcTensorDesc->_dims;

	for(int i = 0; i < dims; i++) {
		if(srcDims[i] != dstDims[i]) {
			return mlopenStatusBadParm;
		}
	}
	
	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::_CheckTensorDataTypes(mlopenTensorDescriptor_t srcTensorDesc) {
	
	if(srcTensorDesc->_dataType != this->_dataType) {
		return mlopenStatusBadParm;
	}

	return mlopenStatusSuccess;
}

