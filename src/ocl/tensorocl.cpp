#include <mlopen/tensor.hpp>
#include <mlopen/errors.hpp>
#include <algorithm>

namespace mlopen {

void TensorDescriptor::TransformTensor(Handle& /* handle */,
			const void * /*alpha*/,
			const TensorDescriptor& srcTensorDesc,
			const cl_mem  /*srcTensor*/,
			const void * /*beta*/,
			cl_mem  /*dstTensor*/) {

	printf("To be implemented (TransformTensor) \n");

	if(*this == srcTensorDesc) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	// Check that output tensors do not overlap .. output tensors cannot be transformed in place .. no aliasing
	// Implement conversion of unsupported tensor to a supported one
	// Launch kernels using the handle
	// If beta[0] = 0 then just a memcopy with scaled alpha[0]?

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

//	OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

	// If beta = 0, y = alpha*x
}

void TensorDescriptor::OpTensor(Handle& /* handle */,
		mlopenTensorOp_t				 /*tensorOp*/,
		const void						* /*alpha1*/,
		const TensorDescriptor&	inputTensorDesc1,
		const cl_mem					 /*inputTensor1*/,
		const void						* /*alpha2*/,
		const TensorDescriptor&	inputTensorDesc2,
		const cl_mem					 /*inputTensor2*/,
		const void						* /*beta*/,
		cl_mem							 /*dstTensor*/) {

	printf("To be implemented (Op Tensor) \n");

	// inputTensor1 and dstTensor must have same dims
	if(this->lens != inputTensorDesc1.lens) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	// input Tensor2 and dstTensor must have same dims or all the dims of
	// inputTensor2 must be 1
	if(
		this->lens != inputTensorDesc2.lens && 
		! std::all_of(inputTensorDesc2.lens.begin(), inputTensorDesc2.lens.end(), [](int x) { return x == 1; })
	) 
	{
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	
	if(this->type != inputTensorDesc1.type && this->type != inputTensorDesc2.type) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	// Launch kernels using the handle

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

}

void TensorDescriptor::SetTensor(Handle& /* handle */,
		cl_mem							dstTensor,
		const void						*valuePtr) {

	printf("To be implemented (SetTensor) \n");
	if(valuePtr == nullptr || dstTensor == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	// Launch kernels using the handle

	// [MD]: Can we just use host enqueue API to set the values in
	// the buffer?

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

//	OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

}

void TensorDescriptor::ScaleTensor(Handle& /* handle */,
		cl_mem							dstTensor,
		const void						* /*alpha*/) {

	printf("To be implemented (ScaleTensor) \n");
	if(dstTensor == nullptr) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}


	// [MD]: Can we just use the TransformTensor Kernel with beta = 0 ?

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

}

void TensorDescriptor::CopyTensor(Handle &handle, 
		const TensorDescriptor &srcDesc,
		const cl_mem src,
		cl_mem dest) {

	size_t srcSize = srcDesc.GetElementSize();

	cl_int status;
	status = clEnqueueCopyBuffer(handle.GetStream(), src, dest, 0, 0, srcSize*sizeof(srcDesc.GetType()), 0, NULL, NULL);

	if(status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status); }
}

} // namespace mlopen
