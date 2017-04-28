#ifndef GUARD_MIOPEN_TENSOR_HPP_
#define GUARD_MIOPEN_TENSOR_HPP_

#include <miopen/handle.hpp>
#include <miopen/object.hpp>
#include <miopen.h>
#include <miopen/common.hpp>
#include <vector>
#include <iostream>
#include <cassert>
// TODO(paul): remove this include later
#include <cstdio>

namespace miopen {

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


struct TensorDescriptor : miopenTensorDescriptor {
	TensorDescriptor();
	TensorDescriptor(miopenDataType_t t, std::initializer_list<int> plens);
	TensorDescriptor(miopenDataType_t t, std::initializer_list<int> plens, std::initializer_list<int> pstrides);
	TensorDescriptor(miopenDataType_t t, const int* plens, int size);
	TensorDescriptor(miopenDataType_t t, const int* plens, const int* pstrides, int size);

	void CalculateStrides();

	const std::vector<int>& GetLengths() const;
	const std::vector<int>& GetStrides() const;
	int GetSize() const;

	miopenDataType_t GetType() const;

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

	friend std::ostream& operator<< (std::ostream& stream, const TensorDescriptor& t);
private:
	std::vector<int> lens;
	std::vector<int> strides;

	miopenDataType_t type;
};


}  // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenTensorDescriptor, miopen::TensorDescriptor)


#endif // GUARD_MIOPEN_TENSOR_HPP_
