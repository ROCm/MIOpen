#ifndef _MLOPEN_POOLING_HPP_
#define _MLOPEN_POOLING_HPP_

#include <MLOpen.h>
#include <errors.hpp>
#include <Handle.hpp>
#include <Tensor.hpp>
#include "KernelCache.hpp"
#include "Common.hpp"
#include <vector>

struct mlopenPoolingDescriptor {
	mlopenPoolingDescriptor();
	mlopenPoolingDescriptor(mlopenPoolingMode_t m, const int *plens, const int *ppads, const int *pstrides, int size);

	const std::vector<int>& GetLengths() const;
	const std::vector<int>& GetStrides() const;
	const std::vector<int>& GetPads() const;
	int GetSize() const;

	mlopenPoolingMode_t GetMode() const;

	mlopenStatus_t GetForwardOutputDim(
		const mlopenTensorDescriptor_t		tensorDesc,
		int									*n,
		int									*c,
		int									*h,
		int									*w);

	mlopenStatus_t Forward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		Data_t								y);

	mlopenStatus_t Backward(
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
		Data_t								dx);

	private:
	std::vector<int> lens;
	std::vector<int> strides;
	std::vector<int> pads;	

	mlopenPoolingMode_t mode;
};
#endif // _MLOPEN_POOLING_HPP_
