#include <mlopen/softmax.hpp>
#include <mlopen/mlo_internal.hpp>
#include <mlopen/kernel_cache.hpp>

namespace mlopen {

mlopenStatus_t SoftmaxForward(
		Handle						&handle,
		const void					* /*alpha*/,
		const void					* /*beta*/,
		const TensorDescriptor		&yDesc,
		cl_mem						y) 
{
	int n, c, h, w;
	std::tie(n, c, h, w) = tie4(yDesc.GetLengths());
	
	std::string program_name = "MLOpenSoftmax.cl";
	std::string kernel_name = "SoftmaxForward";
	//TODO: do we need to pass network_config?
	std::string network = "placeholder";

	size_t workgroups = std::min(n*h*w, 64*40*32);

	const std::vector<size_t> vld(1, 64);
	const std::vector<size_t> vgd(1, workgroups*vld[0]);

	cl_long  sz = n*h*w;

	handle.GetKernel("mlopenSoftmaxForward",
			network,
			program_name,
			kernel_name,
			vld,
			vgd,
			"")(y, c, sz);


	return mlopenStatusSuccess;
}

mlopenStatus_t SoftmaxBackward(
		Handle						&/*handle*/,
		const void					* /*alpha*/,
		const TensorDescriptor		&/*yDesc*/,
		const cl_mem				/*y*/,
		const TensorDescriptor		&/*dyDesc*/,
		const cl_mem				/*dy*/,
		const void					* /*beta*/,
		const TensorDescriptor		&/*dxDesc*/,
		cl_mem						/*dx*/) 
{
	printf("in softmax backward\n");
	return mlopenStatusSuccess;
}

} // namespace mlopen
