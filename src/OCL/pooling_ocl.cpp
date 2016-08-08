#include <pooling.hpp>
#include "mlo_internal.hpp"

mlopenStatus_t mlopenPoolingDescriptor::Forward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const cl_mem						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		cl_mem								y) {
	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in pooling forward\n");
	return(status);
}

mlopenStatus_t mlopenPoolingDescriptor::Backward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const cl_mem						y,
		const mlopenTensorDescriptor_t		dyDesc,
		const cl_mem						dy,
		const mlopenTensorDescriptor_t		xDesc,
		const cl_mem						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		cl_mem								dx) {

	mlopenStatus_t status = mlopenStatusSuccess;
	printf("in pooling backward\n");
	return(status);
}

