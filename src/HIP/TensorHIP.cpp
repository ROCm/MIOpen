#include "Tensor.hpp"

#if MLOpen_BACKEND_HIP
template<>
mlopenStatus_t mlopenTensorDescriptor::TransformTensor<void *>(mlopenHandle_t handle,
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

	// If beta = 0, y = alpha*x;
	return mlopenStatusSuccess;
}

template<>
mlopenStatus_t mlopenTensorDescriptor::OpTensor<void *>(mlopenHandle_t handle,
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

	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

	return mlopenStatusSuccess;

}

template<>
mlopenStatus_t mlopenTensorDescriptor::SetTensor<void *>(mlopenHandle_t handle,
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

//	OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

	return mlopenStatusSuccess;

}

template<>
mlopenStatus_t mlopenTensorDescriptor::ScaleTensor<void *>(mlopenHandle_t handle,
		void							*dstTensor,
		const void						*alpha) {

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

	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

	return mlopenStatusSuccess;

}


#endif
