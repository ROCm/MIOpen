#include <mlopen/softmax.hpp>
#include <mlopen/mlo_internal.hpp>
#include <mlopen/kernel_cache.hpp>

namespace mlopen {

mlopenStatus_t SoftmaxForward(
		Handle						&handle,
		const void					* /*alpha*/,
		const TensorDescriptor		&xDesc,
		const cl_mem				x,
		const void					* /*beta*/,
		const TensorDescriptor		&yDesc,
		cl_mem						y) {

	printf("in softmax forward\n");
}

mlopenStatus_t SoftmaxBackward(
		Handle						&handle,
		const void					* /*alpha*/,
		const TensorDescriptor		&yDesc,
		const cl_mem				y,
		const TensorDescriptor		&dyDesc,
		const cl_mem				dy,
		const void					* /*beta*/,
		const TensorDescriptor		&dxDesc,
		cl_mem						dx) {

	printf("in softmax backward\n");
}

} // namespace mlopen
