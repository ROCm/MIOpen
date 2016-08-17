#include <lrn.hpp>
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreateLRNDescriptor(
		mlopenLRNDescriptor_t *lrnDesc) {

	return mlopen::try_([&] {
		mlopen::deref(lrnDesc) = new mlopenLRNDescriptor();
	});
}

extern "C"
mlopenStatus_t mlopenSetLRNDescriptor(
		mlopenLRNDescriptor_t		lrnDesc,
		mlopenLRNMode_t				mode,
		unsigned int				lrnN,	
		double						lrnAlpha,
		double						lrnBeta,
		double						lrnK) {
		
	return mlopen::try_([&] {
		std::initializer_list<double> parms = {lrnAlpha, lrnBeta, lrnK};
		mlopen::deref(lrnDesc) = mlopenLRNDescriptor(mode, 
			lrnN,
			parms.begin());
	});
}

extern "C"
mlopenStatus_t mlopenGet2dLRNDescriptor(
		const mlopenLRNDescriptor_t		lrnDesc,
		mlopenLRNMode_t					*mode,
		unsigned int					*lrnN,
		double							*lrnAlpha,
		double							*lrnBeta,
		double							*lrnK) {

	return mlopen::try_([&] {
		mlopen::deref(mode) = lrnDesc->GetMode();
		mlopen::deref(lrnN) = lrnDesc->GetN();
		mlopen::deref(lrnAlpha) = lrnDesc->GetAlpha();
		mlopen::deref(lrnBeta) = lrnDesc->GetBeta();
		mlopen::deref(lrnK) = lrnDesc->GetK();
	});
}

extern "C"
mlopenStatus_t mlopenLRNForward(
		mlopenHandle_t						handle,
		const mlopenLRNDescriptor_t			lrnDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		void								*y,
		bool                                do_backward,
		void								*workSpace,
		size_t								workSpaceSize) {


	return mlopen::try_([&] {
		lrnDesc->Forward(handle,
			alpha,
			xDesc,
			DataCast(x),
			beta,
			yDesc,
			DataCast(y),
			do_backward,
			DataCast(workSpace),
			workSpaceSize);

	});
}

extern "C"
mlopenStatus_t mlopenLRNBackward(
		mlopenHandle_t						handle,
		const mlopenLRNDescriptor_t			lrnDesc,
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
		lrnDesc->Backward(handle,
			alpha,
			yDesc,
			DataCast(y),
			dyDesc,
			DataCast(dy),
			xDesc,
			DataCast(x),
			beta,
			dxDesc,
			DataCast(dx),
			DataCast(workSpace));
	});
}

