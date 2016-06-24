#include "Tensor.hpp"
#include <string>

mlopenTensorDescriptor::mlopenTensorDescriptor() : _dims(4) {

	printf("In Tensor Descriptor Ctor\n");
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

mlopenStatus_t mlopenTensorDescriptor::TransformTensor(mlopenHandle_t handle,
		const void *alpha,
		const mlopenTensorDescriptor_t srcTensorDesc,
		const void *srcTensor,
		const void *beta,
		void *dstTensor) {

	printf("To be implemented (TransformTensor) \n");

	if(this->_CheckTensorDims(srcTensorDesc) != mlopenStatusSuccess) {
		return mlopenStatusBadParm;
	}

	// Check that output tensors do not overlap .. output tensors cannot be transformed in place .. no aliasing
	// Implement conversion of unsupported tensor to a supported one
	// Launch kernels using the handle
	// If beta[0] = 0 then just a memcopy with scaled alpha[0]?

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	mlopenStream_t queue;
	handle->GetStream(&queue);

//	OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

	// If beta = 0, y = alpha*x;
	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::OpTensor(mlopenHandle_t handle,
		mlopenTensorOp_t				tensorOp,
		const void						*alpha1,
		const mlopenTensorDescriptor_t	inputTensorDesc1,
		const void						*inputTensor1,
		const void						*alpha2,
		const mlopenTensorDescriptor_t	inputTensorDesc2,
		const void						*inputTensor2,
		const void						*beta,
		void							*dstTensor) {
	
	printf("To be implemented (Op Tensor) \n");

	// inputTensor1 and dstTensor must have same dims
	if(this->_CheckTensorDims(inputTensorDesc1) != mlopenStatusSuccess) {
		return mlopenStatusBadParm;
	}

	// input Tensor2 and dstTensor must have same dims or all the dims of
	// inputTensor2 must be 1
	if(this->_CheckTensorDims(inputTensorDesc2) != mlopenStatusSuccess) {
		std::vector<int> input2Dims(inputTensorDesc2->_dimA);
		for(auto i : input2Dims) {
			if(i != 1) {
				return mlopenStatusBadParm;
			}
		}
	}
	
	if(this->_CheckTensorDataTypes(inputTensorDesc1) != mlopenStatusSuccess && 
			this->_CheckTensorDataTypes(inputTensorDesc2) != mlopenStatusSuccess) {
		return mlopenStatusBadParm;
	}

	// Launch kernels using the handle

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	mlopenStream_t queue;
	handle->GetStream(&queue);

#if MLOpen_BACKEND_OPENCL
	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);
#endif

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenTensorDescriptor::SetTensor(mlopenHandle_t handle,
		void							*dstTensor,
		const void						*valuePtr) {

	printf("To be implemented (SetTensor) \n");
	if(valuePtr == nullptr || dstTensor == nullptr) {
		return mlopenStatusBadParm;
	}

	// Launch kernels using the handle

	// [MD]: Can we just use host enqueue API to set the values in
	// the buffer?

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	mlopenStream_t queue;
	handle->GetStream(&queue);

#if MLOpen_BACKEND_OPENCL
//	OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);
#endif

	return mlopenStatusSuccess;

}

mlopenStatus_t mlopenTensorDescriptor::ScaleTensor(mlopenHandle_t handle,
		void							*dstTensor,
		const void						*valuePtr) {

	printf("To be implemented (ScaleTensor) \n");
	if(dstTensor == nullptr) {
		return mlopenStatusBadParm;
	}


	// [MD]: Can we just use the TransformTensor Kernel with beta = 0 ?

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	mlopenStream_t queue;
	handle->GetStream(&queue);

#if MLOpen_BACKEND_OPENCL
	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);
#endif

	return mlopenStatusSuccess;
}
