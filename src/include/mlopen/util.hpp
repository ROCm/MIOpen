#ifndef _MLOPEN_UTIL_HPP_
#define _MLOPEN_UTIL_HPP_

#include <mlopen.h>
#include <mlopen/errors.hpp>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/tensor_ops.hpp>
#include "mlopen/common.hpp"

namespace mlopen {

float Im2ColGPU(
	Handle	&handle,
	ConstData_t im, size_t im_offset,
	const int c, const int h, const int w,
	const int wei_h, const int wei_w,
	const int out_h, const int out_w,
	const int pad_h, const int	pad_w,
	const int stride_h, const int stride_w,
	Data_t					col);

} // namespace mlopen
#endif // _MLOPEN_UTIL_HPP_
