#include <pooling.hpp>
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreatePoolingDescriptor(
		mlopenPoolingDescriptor_t *poolDesc) {

	return mlopen::try_([&] {
		mlopen::deref(poolDesc) = new mlopenPoolingDescriptor();
	});
}

extern "C"
mlopenStatus_t mlopenSet2dPoolingDescriptor(
		mlopenPoolingDescriptor_t			poolDesc,
		mlopenPoolingMode_t					mode,
		int									windowHeight,
		int									windowWidth,
		int									pad_h,
		int									pad_w,
		int									u,
		int									v) {

	return mlopen::try_([&] {
		std::initializer_list<int> lens = {windowHeight, windowWidth};
		std::initializer_list<int> pads = {pad_h, pad_w};
		std::initializer_list<int> strides = {u, v};
		mlopen::deref(poolDesc) = mlopenPoolingDescriptor(mode, 
			lens.begin(),
			pads.begin(),
			strides.begin(), 2);
	});
}

extern "C"
mlopenStatus_t mlopenGet2dPoolingDescriptor(
		const mlopenPoolingDescriptor_t		poolDesc,
		mlopenPoolingMode_t					*mode,
		int									*windowHeight,
		int									*windowWidth,
		int									*pad_h,
		int									*pad_w,
		int									*u,
		int									*v) {

	return mlopen::try_([&] {
		mlopen::deref(mode) = poolDesc->GetMode();
		std::tie(mlopen::deref(windowHeight), mlopen::deref(windowWidth)) = tie2(poolDesc->GetLengths());
		std::tie(mlopen::deref(u), mlopen::deref(v)) = tie2(poolDesc->GetStrides());
		std::tie(mlopen::deref(pad_h), mlopen::deref(pad_w)) = tie2(poolDesc->GetPads());
	});
}

extern "C"
mlopenStatus_t mlopenSetNdPoolingDescriptor(
		mlopenPoolingDescriptor_t			poolDesc,
		mlopenPoolingMode_t					mode,
		int									nbDims,
		int									*windowDimA,
		int									*padA,
		int									*stridesA) {

	return mlopen::try_([&] {
		mlopen::deref(poolDesc) = mlopenPoolingDescriptor(mode, windowDimA, padA, stridesA, nbDims);
	});
}

extern "C"
mlopenStatus_t mlopenGetNdPoolingDescriptor(
		mlopenPoolingDescriptor_t			poolDesc,
		mlopenPoolingMode_t					*mode,
		int									*nbDims,
		int									*windowDimA,
		int									*padA,
		int									*stridesA) {

	return mlopen::try_([&] {
		if (mode != nullptr) {
			*mode = poolDesc->GetMode();
		}
		if (nbDims != nullptr) {
			*nbDims = poolDesc->GetSize();
		}
		if (windowDimA != nullptr) {
			std::copy(poolDesc->GetLengths().begin(), poolDesc->GetLengths().end(), windowDimA);
		}
		if (stridesA != nullptr) {
			std::copy(poolDesc->GetStrides().begin(), poolDesc->GetStrides().end(), stridesA);
		}
		if (padA != nullptr) {
			std::copy(poolDesc->GetPads().begin(), poolDesc->GetPads().end(), padA);
		}

	});
}

extern "C"
mlopenStatus_t mlopenGetPoolingForwardOutputDim(
		const mlopenPoolingDescriptor_t		poolDesc,
		const mlopenTensorDescriptor_t		tensorDesc,
		int									*n,
		int									*c,
		int									*h,
		int									*w) {

	return mlopen::try_([&] {
		poolDesc->GetForwardOutputDim(tensorDesc,
				n, 
				c, 
				h,
				w);
	});

}

extern "C"
mlopenStatus_t mlopenPoolingForward(
		mlopenHandle_t						handle,
		const mlopenPoolingDescriptor_t		poolDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		void								*y) {

	return mlopen::try_([&] {
		poolDesc->Forward(handle,
				alpha,
				xDesc,
				DataCast(x),
				beta,
				yDesc,
				DataCast(y));
	});

}

extern "C"
mlopenStatus_t mlopenPoolingBackward(
		mlopenHandle_t						handle,
		const mlopenPoolingDescriptor_t		poolDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const void							*y,
		const mlopenTensorDescriptor_t		dyDesc,
		const void							*dy,
		const mlopenTensorDescriptor_t		xDesc,
		const void							*x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		void								*dx) {

	return mlopen::try_([&] {
		poolDesc->Backward(handle,
				alpha,
				yDesc,
				DataCast(y),
				dyDesc,
				DataCast(dy),
				xDesc,
				DataCast(x),
				beta,
				dxDesc,
				DataCast(dx));
	});

}

extern "C"
mlopenStatus_t mlopenDestroyiPoolingDescriptor(mlopenPoolingDescriptor_t poolDesc) {
	return mlopen::try_([&] {
		delete poolDesc;
	});
}

