#ifndef _MIOPEN_POOLING_HPP_
#define _MIOPEN_POOLING_HPP_

#include <miopen.h>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include "miopen/common.hpp"
#include <vector>

namespace miopen {

struct PoolingDescriptor : miopenPoolingDescriptor {
	PoolingDescriptor();
	PoolingDescriptor(miopenPoolingMode_t m, std::initializer_list<int> plens, std::initializer_list<int> pstrides, std::initializer_list<int> ppads);
	PoolingDescriptor(miopenPoolingMode_t m, const int *plens, const int *ppads, const int *pstrides, int size);

	miopenPoolingMode_t GetMode() const;
	const std::vector<int>& GetLengths() const;
	const std::vector<int>& GetStrides() const;
	const std::vector<int>& GetPads() const;
	miopenPoolingMode_t GetMode();
	int GetSize() const;

	std::tuple<int, int, int, int> GetForwardOutputDim(const TensorDescriptor& tensorDesc) const;
	TensorDescriptor GetForwardOutputTensor(const TensorDescriptor& tensorDesc) const;

	std::size_t GetWorkSpaceSize(const TensorDescriptor& tensorDesc) const;

	miopenStatus_t Forward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&xDesc,
		ConstData_t				x,
		const void					*beta,
		const TensorDescriptor		&yDesc,
		Data_t						y,
		bool						do_backward,
		Data_t						workSpace,
		size_t						workSpaceSize) const;

	miopenStatus_t Backward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&yDesc,
		ConstData_t				y,
		const TensorDescriptor		&dyDesc,
		ConstData_t				dy,
		const TensorDescriptor		&xDesc,
		ConstData_t				x,
		const void					*beta,
		const TensorDescriptor		&dxDesc,
		Data_t						dx,
		ConstData_t                workSpace) const;

	friend std::ostream& operator<< (std::ostream& stream, const PoolingDescriptor& x);

	std::vector<int> lens;
	std::vector<int> strides;
	std::vector<int> pads;	

	miopenPoolingMode_t mode;
};
}  // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenPoolingDescriptor, miopen::PoolingDescriptor);
#endif // _MIOPEN_POOLING_HPP_
