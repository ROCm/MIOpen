#include <lrn.hpp>

mlopenStatus_t Forward(
		mlopenHandle_t						handle,
		const mlopenLRNDescriptor_t			lrnDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		Data_t								y) {
	
	printf("in lrn forward\n");
}
mlopenStatus_t Backward(
		mlopenHandle_t						handle,
		const mlopenLRNDescriptor_t			lrnDesc,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const Data_t						y,
		const mlopenTensorDescriptor_t		dyDesc,
		const Data_t						dy,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		Data_t								dx) {

	printf("in lrn backward\n");
}

