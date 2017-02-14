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
mlopenStatus_t AddTensor(Handle&              handle,
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

    if(a_lens.size() != c_lens.size()) {
        MLOPEN_THROW("Number of Tensor dims do not match: " + std::to_string(a_lens.size()) + ", " + std::to_string(c_lens.size()));
    }

    for(auto i = 0; i < a_lens.size(); i++) {
        if(a_lens[i] != 1 && a_lens[i] != c_lens[i]) {
            MLOPEN_THROW("ATensor dim != 1 && ATensor dim != CTensor dim: " + i);
        }
    }

    auto first_not_one = std::find_if(a_lens.rbegin(), a_lens.rend(), [](int i){ return i != 1; });
    auto d = std::distance(a_lens.begin(), first_not_one.base());

    int num_wg = *first_not_one;
    int work_per_wg = std::accumulate(a_lens.begin(), a_lens.begin() + (d-1), 1, std::multiplies<int>()) *
                    std::accumulate(c_lens.begin() + d, c_lens.end(), 1, std::multiplies<int>());

    int n_not_ones = std::count_if(a_lens.begin(), a_lens.end(), [](int i){ return i != 1; });

    int a_n, a_c, a_h, a_w;
    std::tie(a_n, a_c, a_h, a_w) = tie4(aTensorDesc.GetLengths());

    int c_n, c_c, c_h, c_w;
    std::tie(c_n, c_c, c_h, c_w) = tie4(cTensorDesc.GetLengths());

    std::string program_name = "MLOpenTensorKernels.cl";
    std::string kernel_name = "AddTensor";

	const std::vector<size_t> vld(1, 256);
	const std::vector<size_t> vgd(1, num_wg*256);

    std::string parms = " -DFIRST_N=" + std::to_string(d-1)
                    + " -DN_NOT_ONES=" + std::to_string(n_not_ones);

    handle.GetKernel(kernel_name,
            "placeholder",
            program_name,
            kernel_name,
            vld,
            vgd,
            parms) (ATensor, a_n, a_c, a_h, a_w, CTensor, c_n, c_c, c_h, c_w, n_not_ones, work_per_wg);

    return mlopenStatusSuccess;
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

mlopenStatus_t CopyTensor(Handle &handle, 
		const TensorDescriptor &srcDesc,
		ConstData_t src,
		const TensorDescriptor &destDesc,
		Data_t dest) {

	if(srcDesc.GetElementSize() != destDesc.GetElementSize() || srcDesc.GetType() != destDesc.GetType()) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	size_t srcSize = srcDesc.GetElementSize();

	handle.Copy(src, dest, srcSize*sizeof(srcDesc.GetType()));

    return mlopenStatusSuccess;
}

} // namespace mlopen
