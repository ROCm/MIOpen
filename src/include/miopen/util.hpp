#ifndef MIOPEN_UTIL_HPP_
#define MIOPEN_UTIL_HPP_

#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include "miopen/common.hpp"

namespace miopen {

float Im2ColGPU(
	Handle	&handle,
	int data_size, ConstData_t im, size_t im_offset,
	int c, int h, int w,
	int wei_h, int wei_w,
	int out_h, int out_w,
	int pad_h, int	pad_w,
	int stride_h, int stride_w,
	Data_t					col);

} // namespace miopen
#endif // _MIOPEN_UTIL_HPP_
