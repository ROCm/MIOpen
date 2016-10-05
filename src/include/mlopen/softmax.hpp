#ifndef _MLOPEN_SOFTMAX_HPP_
#define _MLOPEN_SOFTMAX_HPP_

#include <mlopen.h>
#include <mlopen/errors.hpp>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include "mlopen/common.hpp"

namespace mlopen {

	mlopenStatus_t SoftmaxForward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&xDesc,
		ConstData_t					x,
		const void					*beta,
		const TensorDescriptor		&yDesc,
		Data_t						y);

	mlopenStatus_t SoftmaxBackward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&yDesc,
		ConstData_t					y,
		const TensorDescriptor		&dyDesc,
		ConstData_t					dy,
		const void					*beta,
		const TensorDescriptor		&dxDesc,
		Data_t						dx);

} // namespace mlopen
#endif // _MLOPEN_SOFTMAX_HPP_
