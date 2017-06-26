#include <miopen/softmax.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>

extern "C"
miopenStatus_t miopenSoftmaxForward(
	miopenHandle_t						handle,
	const void							*alpha,
	const miopenTensorDescriptor_t		xDesc,
	const void							*x,
	const void							*beta,
	const miopenTensorDescriptor_t		yDesc,
	void								*y) {
	MIOPEN_LOG_FUNCTION(alpha, xDesc, x, beta, yDesc, y);
	return miopen::try_([&] {
		CopyTensor(miopen::deref(handle),
			miopen::deref(xDesc),
			DataCast(x),
			miopen::deref(yDesc),
			DataCast(y));

		miopen::SoftmaxForward(miopen::deref(handle),
			alpha,
			beta,
			miopen::deref(yDesc),
			DataCast(y));
	});
}

miopenStatus_t miopenSoftmaxBackward(
	miopenHandle_t						handle,
	const void							*alpha,
	const miopenTensorDescriptor_t		yDesc,
	const void							*y,
	const miopenTensorDescriptor_t		dyDesc,
	const void							*dy,
	const void							*beta,
	const miopenTensorDescriptor_t		dxDesc,
	void								*dx) {

	MIOPEN_LOG_FUNCTION(alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
	return miopen::try_([&] {
		CopyTensor(miopen::deref(handle),
			miopen::deref(dyDesc),
			DataCast(dy),
			miopen::deref(dxDesc),
			DataCast(dx));

		miopen::SoftmaxBackward(miopen::deref(handle),
			alpha,
			miopen::deref(yDesc),
			DataCast(y),
			beta,
			miopen::deref(dxDesc),
			DataCast(dx));
	});
}
