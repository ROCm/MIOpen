#include <mlopen/pooling.hpp>
#include <mlopen/errors.hpp>
#include <initializer_list>
#include <array>

extern "C"
mlopenStatus_t mlopenCreatePoolingDescriptor(
		mlopenPoolingDescriptor_t *poolDesc) {

	return mlopen::try_([&] {
		mlopen::deref(poolDesc) = new mlopen::PoolingDescriptor();
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
		mlopen::deref(poolDesc) = mlopen::PoolingDescriptor(mode, 
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
		mlopen::deref(mode) = mlopen::deref(poolDesc).mode;
		std::tie(mlopen::deref(windowHeight), mlopen::deref(windowWidth)) = mlopen::tie2(mlopen::deref(poolDesc).GetLengths());
		std::tie(mlopen::deref(u), mlopen::deref(v)) = mlopen::tie2(mlopen::deref(poolDesc).GetStrides());
		std::tie(mlopen::deref(pad_h), mlopen::deref(pad_w)) = mlopen::tie2(mlopen::deref(poolDesc).GetPads());
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
		mlopen::deref(poolDesc) = mlopen::PoolingDescriptor(mode, windowDimA, padA, stridesA, nbDims);
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
			*mode = mlopen::deref(poolDesc).mode;
		}
		if (nbDims != nullptr) {
			*nbDims = mlopen::deref(poolDesc).GetSize();
		}
		if (windowDimA != nullptr) {
			std::copy(mlopen::deref(poolDesc).GetLengths().begin(), mlopen::deref(poolDesc).GetLengths().end(), windowDimA);
		}
		if (stridesA != nullptr) {
			std::copy(mlopen::deref(poolDesc).GetStrides().begin(), mlopen::deref(poolDesc).GetStrides().end(), stridesA);
		}
		if (padA != nullptr) {
			std::copy(mlopen::deref(poolDesc).GetPads().begin(), mlopen::deref(poolDesc).GetPads().end(), padA);
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
		mlopen::tie_deref(n, c, h, w) = mlopen::deref(poolDesc).GetForwardOutputDim(mlopen::deref(tensorDesc)); 
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
		void								*y,
		bool                                do_backward,
		void								*workSpace,
		size_t								workSpaceSize) {

	return mlopen::try_([&] {
			mlopen::deref(poolDesc).Forward(mlopen::deref(handle),
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
		void								*dx,
		const void							*workSpace) {

	return mlopen::try_([&] {
			mlopen::deref(poolDesc).Backward(mlopen::deref(handle),
				alpha,
				mlopen::deref(yDesc),
				mlopen::deref(dyDesc),
				DataCast(dy),
				mlopen::deref(xDesc),
				beta,
				mlopen::deref(dxDesc),
				DataCast(dx),
				DataCast(workSpace));
	});

}

extern "C"
mlopenStatus_t mlopenDestroyPoolingDescriptor(mlopenPoolingDescriptor_t poolDesc) {
	return mlopen::try_([&] {
		delete poolDesc;
	});
}

