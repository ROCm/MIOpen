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
	assert(x.size() == 4);
	return std::tie(x[0], x[1], x[2], x[3]);
}

template<class T>
auto tie2(T&& x) -> decltype(std::tie(x[0], x[1]))
{
	assert(x.size() == 2);
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

	void SetTensor(Handle& handle,
			Data_t							dstTensor,
			const void						*valuePtr);

	void ScaleTensor(Handle& handle,
			Data_t							dstTensor,
			const void						*alpha);

private:
	std::vector<int> lens;
	std::vector<int> strides;

	mlopenDataType_t type;
};
}  // namespace mlopen

MLOPEN_DEFINE_OBJECT(mlopenTensorDescriptor, mlopen::TensorDescriptor)


#endif // GUARD_MLOPEN_TENSOR_HPP_
