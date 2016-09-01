#include <mlopen/activ.hpp>
#include <mlopen/errors.hpp>
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreateActivationDescriptor(
		mlopenActivationDescriptor_t *activDesc) {

	return mlopen::try_([&] {
		mlopen::deref(activDesc) = new mlopen::ActivationDescriptor();
	});
}

extern "C"
mlopenStatus_t mlopenSetActivationDescriptor(
		mlopenActivationDescriptor_t		activDesc,
		mlopenActivationMode_t				mode,
		double						activAlpha,
		double						activBeta,
		double						activPower) {
		
	return mlopen::try_([&] {
		std::initializer_list<double> parms = {activAlpha, activBeta, activPower};
		mlopen::deref(activDesc) = mlopen::ActivationDescriptor(mode, 
			parms.begin());
	});
}

extern "C"
mlopenStatus_t mlopenGetActivationDescriptor(
		mlopenActivationDescriptor_t	activDesc,
		mlopenActivationMode_t			*mode,
		double							*activAlpha,
		double							*activBeta,
		double							*activPower) {

	return mlopen::try_([&] {
		*mode = mlopen::deref(activDesc).GetMode();
		*activAlpha = mlopen::deref(activDesc).GetAlpha();
		*activBeta = mlopen::deref(activDesc).GetBeta();
		*activPower = mlopen::deref(activDesc).GetPower();
	});
}

extern "C"
mlopenStatus_t mlopenActivationForward(
		mlopenHandle_t						handle,
		mlopenActivationDescriptor_t		activDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		void								*y,
		bool                                do_backward,
		void								*workSpace,
		size_t								*workSpaceSize) {


	return mlopen::try_([&] {
			mlopen::deref(activDesc).Forward(mlopen::deref(handle),
			alpha,
			mlopen::deref(xDesc),
			DataCast(x),
			beta,
			mlopen::deref(yDesc),
			DataCast(y),
			do_backward,
			DataCast(workSpace),
			workSpaceSize);

	});
}

extern "C"
mlopenStatus_t mlopenActivationBackward(
		mlopenHandle_t						handle,
		mlopenActivationDescriptor_t		activDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const void							*y,
		const mlopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		void								*dx,
		const void							*workSpace) {

	return mlopen::try_([&] {
			mlopen::deref(activDesc).Backward(mlopen::deref(handle),
			alpha,
			mlopen::deref(yDesc),
			DataCast(y),
			mlopen::deref(dyDesc),
			DataCast(dy),
			mlopen::deref(xDesc),
			DataCast(x),
			beta,
			mlopen::deref(dxDesc),
			DataCast(dx),
			DataCast(workSpace));
	});
}

extern "C"
mlopenStatus_t mlopenDestroyActivationDescriptor(mlopenActivationDescriptor_t activDesc) {
	return mlopen::try_([&] {
		delete activDesc;
	});
}

