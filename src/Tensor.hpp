#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include "MLOpen.h"
#include <vector>
// TODO: remove this include later
#include <cstdio>

struct mlopenTensorDescriptor {
	mlopenTensorDescriptor();

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
	mlopenStatus_t GetDataType(mlopenDataType_t *dataType);
	mlopenStatus_t GetDims(int *dims);

	// Transform functions
	mlopenStatus_t TransformTensor(mlopenHandle_t handle,
			const void *alpha,
			const mlopenTensorDescriptor_t srcTensorDesc,
			const void *srcTensor,
			const void *beta,
			void *dstTensor);
	
	int _dims;
	std::vector<int> _dimA;
	std::vector<int> _strideA;

	mlopenDataType_t _dataType;
	mlopenHandle_t _tensorHandle;
};

#endif // _TENSOR_HPP_
