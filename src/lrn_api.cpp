#include <mlopen/lrn.hpp>
#include <mlopen/errors.hpp>
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreateLRNDescriptor(
		mlopenLRNDescriptor_t *lrnDesc) {

	return mlopen::try_([&] {
		mlopen::deref(lrnDesc) = new mlopen::LRNDescriptor();
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
		mlopen::deref(lrnDesc) = mlopen::LRNDescriptor(mode, 
			lrnN,
			parms.begin());
	});
}

extern "C"
mlopenStatus_t mlopenGetLRNDescriptor(
		const mlopenLRNDescriptor_t		lrnDesc,
		mlopenLRNMode_t					*mode,
		unsigned int					*lrnN,
		double							*lrnAlpha,
		double							*lrnBeta,
		double							*lrnK) {

	return mlopen::try_([&] {
		*mode = mlopen::deref(lrnDesc).GetMode();
		*lrnN = mlopen::deref(lrnDesc).GetN();
		*lrnAlpha = mlopen::deref(lrnDesc).GetAlpha();
		*lrnBeta = mlopen::deref(lrnDesc).GetBeta();
		*lrnK = mlopen::deref(lrnDesc).GetK();
	});
}

extern "C"
mlopenStatus_t mlopenLRNGetWorkSpaceSize(
		const mlopenTensorDescriptor_t		yDesc,
		size_t								*workSpaceSize) {
	
	// TODO: Supporting size 4 bytes only
	return mlopen::try_([&] {
		mlopen::deref(workSpaceSize) = mlopen::deref(yDesc).GetLengths()[0] * mlopen::deref(yDesc).GetStrides()[0] * sizeof(float); 
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
		void								*workSpace) {

	return mlopen::try_([&] {
			mlopen::deref(lrnDesc).Forward(mlopen::deref(handle),
			alpha,
			mlopen::deref(xDesc),
			DataCast(x),
			beta,
			mlopen::deref(yDesc),
			DataCast(y),
			do_backward,
			DataCast(workSpace));
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
			mlopen::deref(lrnDesc).Backward(mlopen::deref(handle),
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
mlopenStatus_t mlopenDestroyLRNDescriptor(mlopenLRNDescriptor_t lrnDesc) {
	return mlopen::try_([&] {
		delete lrnDesc;
	});
}

