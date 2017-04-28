#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/errors.hpp>
#include <algorithm>
#include <numeric>

namespace miopen {

void TensorDescriptor::SetTensor(Handle& /* handle */,
		Data_t							dstTensor,
		const void						*valuePtr) {

	printf("To be implemented (SetTensor) \n");
	if(valuePtr == nullptr || dstTensor == nullptr) {
		MIOPEN_THROW(miopenStatusBadParm);
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
		MIOPEN_THROW(miopenStatusBadParm);
	}


	// [MD]: Can we just use the TransformTensor Kernel with beta = 0 ?

	std::string program_name; // CL kernel filename
	std::string kernel_name; // kernel name
	std::string parms; // kernel parameters

	//OCLKernel kernel = KernelCache::get(queue, program_name, kernel_name, parms);

}

// Free Tensor Functions
//
static void CreateBitmapAndGrid(unsigned int &bitmap, std::vector<int> &a_lens, std::vector<int> &c_lens, int &num_wg, int &work, int d)
{
    for(int i = d; i>= 0; i--) {
        if(a_lens[i] != 1) {
            bitmap |= (1 << (a_lens.size()-(i+1))); // works only 4d tensors in NCHW
            num_wg *= a_lens[i];
        }
        else
            work *= c_lens[i];
    }
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
		MIOPEN_THROW(miopenStatusBadParm);
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

static bool IsBitmapLeadingOnes(unsigned int &bitmap, int n_size, int first_not_one)
{
    bool leading_ones = false;

    for(int i = first_not_one; i >= 0; i--) {
        leading_ones &= bitmap & (1 << (n_size-1-i));
    }
    return leading_ones;
}

void OpTensor(Handle&           handle,
		miopenTensorOp_t		tensorOp,
		const void				* /*alpha1*/,
		const TensorDescriptor&	aTensorDesc,
		ConstData_t				ATensor,
		const void				* /*alpha2*/,
		const TensorDescriptor&	bTensorDesc,
		ConstData_t				BTensor,
		const void				* /*beta*/,
		const TensorDescriptor& cTensorDesc,
		Data_t					CTensor) {

    if(ATensor == nullptr || BTensor == nullptr || CTensor == nullptr) {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(aTensorDesc != cTensorDesc) {
        MIOPEN_THROW("A and C Tensors do not match");
    }
   
    if(bTensorDesc.GetType() != cTensorDesc.GetType()) {
        MIOPEN_THROW("Datatypes for B and C tensors do not match !");
    }

    auto b_lens = bTensorDesc.GetLengths();
    auto c_lens = cTensorDesc.GetLengths();

    if(b_lens.size() != c_lens.size()) {
        MIOPEN_THROW("Number of dims in B and C Tensors do not match: " + std::to_string(b_lens.size()) + ", " + std::to_string(c_lens.size()));
    }

    for(auto i = 0; i < c_lens.size(); i++) {
        if(b_lens[i] != 1 && b_lens[i] != c_lens[i]) {
            MIOPEN_THROW("BTensor dim != 1 && BTensor dim != CTensor dim: " + std::to_string(i));
        }
    }
    auto first_not_one = std::find_if(b_lens.rbegin(), b_lens.rend(), [](int i){ return i != 1; });
    auto d = std::distance(b_lens.begin(), first_not_one.base());

    int num_wg = *first_not_one == 0 ? 1 : *first_not_one;
    int work_per_wg = std::accumulate(c_lens.begin() + d, c_lens.end(), 1, std::multiplies<int>());

    int c_n, c_c, c_h, c_w;
    std::tie(c_n, c_c, c_h, c_w) = tie4(cTensorDesc.GetLengths());

    int b_c, b_h, b_w;
    std::tie(std::ignore, b_c, b_h, b_w) = tie4(bTensorDesc.GetLengths());

    int c_nstride, c_cstride;
    std::tie(c_nstride, c_cstride, std::ignore, std::ignore) = tie4(cTensorDesc.GetStrides());
    
    int b_nstride, b_cstride;
    std::tie(b_nstride, b_cstride, std::ignore, std::ignore) = tie4(bTensorDesc.GetStrides());

    unsigned int bitmap = 0;
    // update bitmap for first_not_one
    bitmap |= (1 << (b_lens.size() - d)); 

    // (d-2) is because distance starts from 1 and 0 
    // also, we need to go past the "first_not_one" as that is already
    // accounted for in the bitmap
    CreateBitmapAndGrid(bitmap, b_lens, c_lens, num_wg, work_per_wg, (d-2));

    // Forward Convolution Bias specialization
    // for fwd-bias, bitmap looks like <0, 1, 0, 0>
    // Is the no. of work-groups and the work for each wg balanced?
    auto fwd_conv_bias = bitmap == (1 << 2) ? 1 : 0;
    auto incr_wg = 0;
    if(fwd_conv_bias == 1
            && num_wg < 640 && work_per_wg > 256) { //640 workgroups of size 256 needed to completely fill the GPU
        work_per_wg /= c_n;
        num_wg *= c_n;
        incr_wg = 1;
    }

    size_t local_threads = 256;

    // Does the bitmap contain leading ones, i.e. 1,1,1,0 or 1,1,0,0
    // or 1,1,1,1 or 1,0,0,0
    bool leading_ones = IsBitmapLeadingOnes(bitmap, 4, (d-2));
    if(leading_ones && work_per_wg < 64) {
        local_threads = 64;    
    }

    std::string parms = " -DFWD_CONV_BIAS=" + std::to_string(fwd_conv_bias) +
                        " -DINCR_WG=" + std::to_string(incr_wg) +
                        " -DLEADING_ONES=" + std::to_string(leading_ones) +
                        " -DMIOPEN_TYPE=" + GetDataType(bTensorDesc.GetType()) +
                        " -DFIRST_NOT_ONE=" + std::to_string(d-1);

    std::string program_name = "MIOpenTensorKernels.cl";
	
    const std::vector<size_t> vld {local_threads, 1, 1};
    size_t global_threads = num_wg*local_threads;
	const std::vector<size_t> vgd {global_threads, 1, 1};

    int op = tensorOp;

    if(fwd_conv_bias) {
        handle.GetKernel("OpTensorFwdBias",
                "",
                program_name,
                "OpTensorFwdBias",
                vld,
                vgd,
                parms) (ATensor, BTensor, b_c, CTensor, c_n, c_nstride, c_cstride, work_per_wg, op);

    }
    else {
        handle.GetKernel("OpTensorGeneric",
                "",
                program_name,
                "OpTensorGeneric",
                vld,
                vgd,
                parms) (ATensor, BTensor, b_c, b_h, b_w, b_nstride, b_cstride, CTensor, c_c, c_h, c_w, c_nstride, c_cstride, bitmap, work_per_wg, op);
    }
}

void CopyTensor(Handle &handle, 
		const TensorDescriptor &srcDesc,
		ConstData_t src,
		const TensorDescriptor &destDesc,
		Data_t dest) {

	if(srcDesc.GetElementSize() != destDesc.GetElementSize() || srcDesc.GetType() != destDesc.GetType()) {
		MIOPEN_THROW(miopenStatusBadParm);
	}
	size_t srcSize = srcDesc.GetElementSize();

	handle.Copy(src, dest, srcSize*sizeof(srcDesc.GetType()));
}

} // namespace miopen
