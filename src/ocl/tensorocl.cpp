#include <mlopen/tensor.hpp>
#include <mlopen/tensor_ops.hpp>
#include <mlopen/errors.hpp>
#include <algorithm>

namespace mlopen {

void TensorDescriptor::SetTensor(Handle& /* handle */,
		Data_t							dstTensor,
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
		Data_t							dstTensor,
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

// Free Tensor Functions
// 
void AddTensor(Handle&              handle,
			const void              * /*alpha*/,
			const TensorDescriptor& aTensorDesc,
			ConstData_t             ATensor,
			const void              * /*beta*/,
			const TensorDescriptor& cTensorDesc,
			Data_t                  CTensor) {

    if(ATensor == nullptr || CTensor == nullptr) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }

    auto a_lens = aTensorDesc.GetLengths();
    auto c_lens = cTensorDesc.GetLengths();

    if(aTensorDesc.GetSize() != cTensorDesc.GetSize()) {
        MLOPEN_THROW("Number of Tensor dims do not match: " + std::to_string(aTensorDesc.GetSize()) + 
                ", " + std::to_string(cTensorDesc.GetSize()));
    }

    for(auto i = 0; i < a_lens.size(); i++) {
        if(a_lens[i] != 1 && a_lens[i] != c_lens[i]) {
            MLOPEN_THROW("ATensor dim != 1 && ATensor dim != CTensor dim: " + i);
        }
    }

    auto first_n = std::find_if(a_lens.rbegin(), a_lens.rend(), [](int i){ return i != 1; });
    auto d = std::distance(a_lens.begin(), first_n.base());

    int num_wg = *first_n;
    int work_per_thread = std::accumulate(c_lens.begin() + d, c_lens.end(), 1, std::multiplies<int>());

    int n, c, h, w;
    std::tie(n, c, h, w) = tie4(aTensorDesc.GetLengths());

}

void TransformTensor(Handle& /* handle */,
			const void * /*alpha*/,
			const TensorDescriptor& srcTensorDesc,
			ConstData_t  /*srcTensor*/,
			const void * /*beta*/,
			const TensorDescriptor& destTensorDesc,
			Data_t  /*destTensor*/) {

	printf("To be implemented (TransformTensor) \n");

	if(destTensorDesc == srcTensorDesc) {
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

void OpTensor(Handle& /* handle */,
		mlopenTensorOp_t				 /*tensorOp*/,
		const void						* /*alpha1*/,
		const TensorDescriptor&	inputTensorDesc1,
		ConstData_t					 /*inputTensor1*/,
		const void						* /*alpha2*/,
		const TensorDescriptor&	inputTensorDesc2,
		ConstData_t					 /*inputTensor2*/,
		const void						* /*beta*/,
		const TensorDescriptor& destTensorDesc,
		Data_t							 /*destTensor*/) {

	printf("To be implemented (Op Tensor) \n");

	// inputTensor1 and dstTensor must have same dims
	if(destTensorDesc.GetLengths() != inputTensorDesc1.GetLengths()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	// input Tensor2 and dstTensor must have same dims or all the dims of
	// inputTensor2 must be 1
	if(
		destTensorDesc.GetLengths() != inputTensorDesc2.GetLengths() && 
		! std::all_of(inputTensorDesc2.GetLengths().begin(), inputTensorDesc2.GetLengths().end(), [](int x) { return x == 1; })
	) 
	{
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	
	if(destTensorDesc.GetType() != inputTensorDesc1.GetType() && destTensorDesc.GetType() != inputTensorDesc2.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}

	// Launch kernels using the handle

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

}

void CopyTensor(Handle &handle, 
		const TensorDescriptor &srcDesc,
		ConstData_t src,
		const TensorDescriptor &destDesc,
		Data_t dest) {

	if(srcDesc.GetElementSize() != destDesc.GetElementSize() || srcDesc.GetType() != destDesc.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	size_t srcSize = srcDesc.GetElementSize();

	handle.Copy(src, dest, srcSize*sizeof(srcDesc.GetType()));
}

} // namespace mlopen
