#ifndef _MLOPEN_UTIL_HPP_
#define _MLOPEN_UTIL_HPP_

#include <mlopen.h>
#include <mlopen/errors.hpp>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/tensor_ops.hpp>
#include "mlopen/common.hpp"

namespace mlopen {

mlopenStatus_t Im2ColGPU(
	Handle					&handle,
	const TensorDescriptor&	imDesc,
	ConstData_t				im,
	const TensorDescriptor&	wDesc,
	const int				pad_h,
	const int				pad_w,
	const int				stride_h,
	const int				stride_w,
	Data_t					col); 

} // namespace mlopen
#endif // _MLOPEN_UTIL_HPP_
