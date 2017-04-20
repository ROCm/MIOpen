#ifndef MIOPEN_SOFTMAX_HPP_
#define MIOPEN_SOFTMAX_HPP_

#include <miopen.h>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include "miopen/common.hpp"

namespace miopen {

miopenStatus_t SoftmaxForward(
	Handle						&handle,
	const void					*alpha,
	const void					*beta,
	const TensorDescriptor		&yDesc,
	Data_t						y);

miopenStatus_t SoftmaxBackward(
	Handle						&handle,
	const void					*alpha,
	const TensorDescriptor		&yDesc,
	ConstData_t					y,
	const void					*beta,
	const TensorDescriptor		&dxDesc,
	Data_t						dx);

} // namespace miopen
#endif // _MIOPEN_SOFTMAX_HPP_
