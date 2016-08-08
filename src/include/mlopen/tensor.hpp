#ifndef GUARD_MLOPEN_TENSOR_HPP_
#define GUARD_MLOPEN_TENSOR_HPP_

#include <mlopen/context.hpp>
#include <mlopen/object.hpp>
#include <mlopen.h>
#include <mlopen/kernel_cache.hpp>
#include <mlopen/common.hpp>
#include <vector>
// TODO: remove this include later
#include <cstdio>

namespace mlopen {

template<class T>
auto tie4(T&& x) -> decltype(std::tie(x[0], x[1], x[2], x[3]))
{
	return std::tie(x[0], x[1], x[2], x[3]);
}

struct TensorDescriptor : mlopenTensorDescriptor {
	TensorDescriptor();
	TensorDescriptor(mlopenDataType_t t, const int* plens, int size);
	TensorDescriptor(mlopenDataType_t t, const int* plens, const int* pstrides, int size);

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

	bool operator==(const TensorDescriptor& rhs) const;
	bool operator!=(const TensorDescriptor& rhs) const;


	mlopenStatus_t TransformTensor(mlopen::Context& handle,
			const void *alpha,
			const TensorDescriptor& srcTensorDesc,
			const Data_t srcTensor,
			const void *beta,
			Data_t dstTensor);

	mlopenStatus_t OpTensor(mlopen::Context& handle,
			mlopenTensorOp_t				tensorOp,
			const void						*alpha1,
			const TensorDescriptor&	aDesc,
			const Data_t					A,
			const void						*alpha2,
			const TensorDescriptor&	bDesc,
			const Data_t					B,
			const void						*beta,
			Data_t							C);

	mlopenStatus_t SetTensor(mlopen::Context& handle,
			Data_t							dstTensor,
			const void						*valuePtr);

	mlopenStatus_t ScaleTensor(mlopen::Context& handle,
			Data_t							y,
			const void						*alpha);

private:
	std::vector<int> lens;
	std::vector<int> strides;

	mlopenDataType_t type;
};
}

MLOPEN_DEFINE_OBJECT(mlopenTensorDescriptor, mlopen::TensorDescriptor)


#endif // GUARD_MLOPEN_TENSOR_HPP_
