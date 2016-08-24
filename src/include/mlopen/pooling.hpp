#ifndef _MLOPEN_POOLING_HPP_
#define _MLOPEN_POOLING_HPP_

#include <mlopen.h>
#include <mlopen/errors.hpp>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include "mlopen/kernel_cache.hpp"
#include "mlopen/common.hpp"
#include <vector>

namespace mlopen {

struct PoolingDescriptor : mlopenPoolingDescriptor {
	PoolingDescriptor();
	PoolingDescriptor(mlopenPoolingMode_t m, const int *plens, const int *ppads, const int *pstrides, int size);

	const std::vector<int>& GetLengths() const;
	const std::vector<int>& GetStrides() const;
	const std::vector<int>& GetPads() const;
	int GetSize() const;

	mlopenPoolingMode_t GetMode() const;

	mlopenStatus_t GetForwardOutputDim(
		const TensorDescriptor				&tensorDesc,
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
		Data_t								y,
		bool								do_backward,
		Data_t								workSpace,
		size_t								workSpaceSize);

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
		Data_t								dx,
		const Data_t                        workSpace);

	private:
	std::vector<int> lens;
	std::vector<int> strides;
	std::vector<int> pads;	

	mlopenPoolingMode_t mode;
};
}
MLOPEN_DEFINE_OBJECT(mlopenPoolingDescriptor, mlopen::PoolingDescriptor);
#endif // _MLOPEN_POOLING_HPP_
