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

	std::tuple<int, int, int, int> GetForwardOutputDim(const TensorDescriptor& inputTensorDesc) const;

	mlopenStatus_t Forward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&xDesc,
		const Data_t				x,
		const void					*beta,
		const TensorDescriptor		&yDesc,
		Data_t						y,
		bool						do_backward,
		Data_t						workSpace,
		size_t						workSpaceSize);

	mlopenStatus_t Backward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&yDesc,
		const Data_t				y,
		const TensorDescriptor		&dyDesc,
		const Data_t				dy,
		const TensorDescriptor		&xDesc,
		const Data_t				x,
		const void					*beta,
		const TensorDescriptor		&dxDesc,
		Data_t						dx,
		const Data_t                workSpace);

	std::vector<int> lens;
	std::vector<int> strides;
	std::vector<int> pads;	

	mlopenPoolingMode_t mode;
};
}
MLOPEN_DEFINE_OBJECT(mlopenPoolingDescriptor, mlopen::PoolingDescriptor);
#endif // _MLOPEN_POOLING_HPP_
