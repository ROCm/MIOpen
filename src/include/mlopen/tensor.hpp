#ifndef GUARD_MLOPEN_TENSOR_HPP_
#define GUARD_MLOPEN_TENSOR_HPP_

#include <mlopen/handle.hpp>
#include <mlopen/object.hpp>
#include <mlopen.h>
#include <mlopen/common.hpp>
#include <vector>
// TODO(paul): remove this include later
#include <cstdio>

namespace mlopen {

template<class T>
auto tie4(T&& x) -> decltype(std::tie(x[0], x[1], x[2], x[3]))
{
	return std::tie(x[0], x[1], x[2], x[3]);
}

template<class T>
auto tie2(T&& x) -> decltype(std::tie(x[0], x[1]))
{
	return std::tie(x[0], x[1]);
}


struct TensorDescriptor : mlopenTensorDescriptor {
	TensorDescriptor();
	TensorDescriptor(mlopenDataType_t t, std::initializer_list<int> plens);
	TensorDescriptor(mlopenDataType_t t, std::initializer_list<int> plens, std::initializer_list<int> pstrides);
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

	std::string ToString() const;

	void TransformTensor(Handle& handle,
			const void *alpha,
			const TensorDescriptor& srcTensorDesc,
			ConstData_t srcTensor,
			const void *beta,
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
			Data_t							C);

	void SetTensor(Handle& handle,
			Data_t							dstTensor,
			const void						*valuePtr);

	void ScaleTensor(Handle& handle,
			Data_t							dstTensor,
			const void						*alpha);

	void CopyTensor(Handle &handle,
			const TensorDescriptor &srcDesc,
			ConstData_t src,
			Data_t dest);

private:
	std::vector<int> lens;
	std::vector<int> strides;

	mlopenDataType_t type;
};
}  // namespace mlopen

MLOPEN_DEFINE_OBJECT(mlopenTensorDescriptor, mlopen::TensorDescriptor)


#endif // GUARD_MLOPEN_TENSOR_HPP_
