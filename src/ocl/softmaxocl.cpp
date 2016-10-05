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
	if(h != 1 || w != 1) {
		throw std::invalid_argument("height and width != 1 is not supported\n");
	}
	std::string program_name = "MLOpenSoftmax.cl";
	std::string kernel_name = "SoftmaxForward";
	//TODO: do we need to pass network_config?
	std::string network = "placeholder";

	const std::vector<size_t> vld(1, 256);
	const std::vector<size_t> vgd(1, n*vld[0]);

	handle.GetKernel("mlopenSoftmaxForward",
			network,
			program_name,
			kernel_name,
			vld,
			vgd,
			"")(y, c);


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
