#ifndef GUARD_MLOPEN_TENSOR_OPPS_HPP_
#define GUARD_MLOPEN_TENSOR_OPPS_HPP_

#include <mlopen/handle.hpp>
#include <mlopen/object.hpp>
#include <mlopen.h>
#include <mlopen/common.hpp>
#include <vector>

namespace mlopen {

void TransformTensor(Handle& handle,
		const void *alpha,
		const TensorDescriptor& srcTensorDesc,
		ConstData_t srcTensor,
		const void *beta,
		const TensorDescriptor& dstTensorDesc,
		Data_t dstTensor);

void OpTensor(Handle& handle,
		mlopenTensorOp_t				tensorOp,
		const void						*alpha1,
		const TensorDescriptor&	inputTensorDesc1,
		ConstData_t					A,
		const void						*alpha2,
		const TensorDescriptor&	inputTensorDesc2,
		ConstData_t					B,
		const void						*beta,
		const TensorDescriptor& CTensorDesc,
		Data_t							C);

void CopyTensor(Handle &handle,
		const TensorDescriptor &srcDesc,
		ConstData_t src,
		const TensorDescriptor &destDesc,
		Data_t dest);

} // namespace mlopen
#endif //GUARD_MLOPEN_TENSOR_OPPS_HPP_
