#ifndef _MLOPEN_TENSOR_HPP_
#define _MLOPEN_TENSOR_HPP_

#include "Handle.hpp"
#include "MLOpen.h"
#include "KernelCache.hpp"
#include "Common.hpp"
#include <vector>
// TODO: remove this include later
#include <cstdio>

template<class T>
auto tie4(T&& x) -> decltype(std::tie(x[0], x[1], x[2], x[3]))
{
	return std::tie(x[0], x[1], x[2], x[3]);
}

struct mlopenTensorDescriptor {
	mlopenTensorDescriptor();
	mlopenTensorDescriptor(mlopenDataType_t t, const int* plens, int size);
	mlopenTensorDescriptor(mlopenDataType_t t, const int* plens, const int* pstrides, int size);

	void CalculateStrides();

	const std::vector<int>& GetLengths() const;
	const std::vector<int>& GetStrides() const;
	int GetSize() const;

	mlopenDataType_t GetType() const;

	int GetElementSize() const;

	int GetIndex(std::initializer_list<int> l) const;

	template<class... Ts>
	int GetIndex(Ts... is) const
	{
		return this->GetIndex({is...});
	}

	bool operator==(const mlopenTensorDescriptor& rhs) const;
	bool operator!=(const mlopenTensorDescriptor& rhs) const;


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

private:
	std::vector<int> lens;
	std::vector<int> strides;

	mlopenDataType_t type;
};

#endif // _MLOPEN_TENSOR_HPP_
