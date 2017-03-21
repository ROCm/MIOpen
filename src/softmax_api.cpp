#include <mlopen/softmax.hpp>
#include <mlopen/errors.hpp>
#include <mlopen/logger.hpp>

extern "C"
mlopenStatus_t mlopenSoftmaxForward(
	mlopenHandle_t						handle,
	const void							*alpha,
	const mlopenTensorDescriptor_t		xDesc,
	const void							*x,
	const void							*beta,
	const mlopenTensorDescriptor_t		yDesc,
	void								*y) {
	MLOPEN_LOG_FUNCTION(alpha, xDesc, x, beta, yDesc, y);
	return mlopen::try_([&] {
		CopyTensor(mlopen::deref(handle),
			mlopen::deref(xDesc),
			DataCast(x),
			mlopen::deref(yDesc),
			DataCast(y));

		mlopen::SoftmaxForward(mlopen::deref(handle),
			alpha,
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

	MLOPEN_LOG_FUNCTION(alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
	return mlopen::try_([&] {
		CopyTensor(mlopen::deref(handle),
			mlopen::deref(dyDesc),
			DataCast(dy),
			mlopen::deref(dxDesc),
			DataCast(dx));

		mlopen::SoftmaxBackward(mlopen::deref(handle),
			alpha,
			mlopen::deref(yDesc),
			DataCast(y),
			beta,
			mlopen::deref(dxDesc),
			DataCast(dx));
	});
}
