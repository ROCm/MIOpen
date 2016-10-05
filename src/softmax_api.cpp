#include <mlopen/softmax.hpp>
#include <mlopen/errors.hpp>

extern "C"
mlopenStatus_t mlopenSoftmaxForward(
	mlopenHandle_t						handle,
	const void							*alpha,
	const mlopenTensorDescriptor_t		xDesc,
	const void							*x,
	const void							*beta,
	const mlopenTensorDescriptor_t		yDesc,
	void								*y) {

	// copy tensors
	//
	return mlopen::try_([&] {
		mlopen::SoftmaxForward(mlopen::deref(handle),
			alpha,
			mlopen::deref(xDesc),
			DataCast(x),
			beta,
			mlopen::deref(yDesc),
			DataCast(y));
	});
}

mlopenStatus_t mlopenSoftmaxBackward(
	mlopenHandle_t						handle,
	const void							*alpha,
	const mlopenTensorDescriptor_t		yDesc,
	const void							*y,
	const mlopenTensorDescriptor_t		dyDesc,
	const void							*dy,
	const void							*beta,
	const mlopenTensorDescriptor_t		dxDesc,
	void								*dx) {

	return mlopen::try_([&] {
		mlopen::SoftmaxBackward(mlopen::deref(handle),
			alpha,
			mlopen::deref(yDesc),
			DataCast(y),
			mlopen::deref(dyDesc),
			DataCast(dy),
			beta,
			mlopen::deref(dxDesc),
			DataCast(dx));
	});
}
