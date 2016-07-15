#ifndef _MLOPEN_TENSOR_HPP_
#define _MLOPEN_TENSOR_HPP_

#include "Handle.hpp"
#include "MLOpen.h"
#include "KernelCache.hpp"
#include "Common.hpp"
#include <vector>
// TODO: remove this include later
#include <cstdio>

struct mlopenTensorDescriptor {
	mlopenTensorDescriptor();

	void CalculateStrides();

	// Set functions
	mlopenStatus_t SetTensorHandle(mlopenHandle_t handle);
	mlopenStatus_t Set4Dims(int n,
			int c,
			int h, 
			int w);
	mlopenStatus_t Set4Strides(int nStride,
			int cStride,
			int hStride,
			int wStride);
 	mlopenStatus_t SetNDims(int dims,
			int *dimsA);
 	mlopenStatus_t SetNStrides(int dims,
  			int *StridesA);
	mlopenStatus_t SetDataType(mlopenDataType_t dataType);
	mlopenStatus_t SetDims(int dims);

	// Get functions
	mlopenStatus_t GetTensorHandle(mlopenHandle_t handle);
	mlopenStatus_t Get4Dims(int *n,
			int *c,
			int *h,
			int *w);
	mlopenStatus_t Get4Strides(int *nStride,
			int *cStride,
			int *hStride,
			int *wStride);
	mlopenStatus_t GetNDims(int *dimsA);
	mlopenStatus_t GetNStrides(int *stridesA);
	mlopenStatus_t GetDataType(mlopenDataType_t &dataType);
	mlopenStatus_t GetDims(int &dims);

	mlopenStatus_t TransformTensor(mlopenHandle_t handle,
			const void *alpha,
			const mlopenTensorDescriptor_t srcTensorDesc,
			const Data_t srcTensor,
			const void *beta,
			Data_t dstTensor);

	mlopenStatus_t OpTensor(mlopenHandle_t handle,
			mlopenTensorOp_t				tensorOp,
			const void						*alpha1,
			const mlopenTensorDescriptor_t	aDesc,
			const Data_t					A,
			const void						*alpha2,
			const mlopenTensorDescriptor_t	bDesc,
			const Data_t					B,
			const void						*beta,
			Data_t							C);

	mlopenStatus_t SetTensor(mlopenHandle_t handle,
			Data_t							dstTensor,
			const void						*valuePtr);

	mlopenStatus_t ScaleTensor(mlopenHandle_t handle,
			Data_t							y,
			const void						*alpha);

	// Internal
	mlopenStatus_t _CheckTensorDims(mlopenTensorDescriptor_t srcTensorDesc);
	mlopenStatus_t _CheckTensorDataTypes(mlopenTensorDescriptor_t srcTensorDesc);

	int _dims;
	std::vector<int> _dimA;
	std::vector<int> _strideA;

	mlopenDataType_t _dataType;
	mlopenHandle_t _tensorHandle;
};

#endif // _MLOPEN_TENSOR_HPP_
