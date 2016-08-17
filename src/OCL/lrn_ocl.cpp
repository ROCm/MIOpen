#include <lrn.hpp>

mlopenStatus_t mlopenLRNDescriptor::Forward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		Data_t								y,
		bool                                do_backward,
		Data_t								workSpace,
		size_t								workSpaceSize) {
	
	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in lrn forward\n");
	return(status);
}

mlopenStatus_t mlopenLRNDescriptor :: Backward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const Data_t						y,
		const mlopenTensorDescriptor_t		dyDesc,
		const Data_t						dy,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		Data_t								dx,
		const Data_t						workSpace) {

	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in lrn backward\n");
	return(status);
}

