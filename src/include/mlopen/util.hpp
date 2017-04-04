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
	int data_size, ConstData_t im, size_t im_offset,
	int c, int h, int w,
	int wei_h, int wei_w,
	int out_h, int out_w,
	int pad_h, int	pad_w,
	int stride_h, int stride_w,
	Data_t					col);

} // namespace mlopen
#endif // _MLOPEN_UTIL_HPP_
