#include "Tensor.hpp"
#include <string>
#include <algorithm>
#include <cassert>

mlopenTensorDescriptor::mlopenTensorDescriptor() {}


mlopenTensorDescriptor::mlopenTensorDescriptor(mlopenDataType_t t, const int* plens, int size)
: type(t), lens(plens, plens+size)
{
	this->CalculateStrides();
}
mlopenTensorDescriptor::mlopenTensorDescriptor(mlopenDataType_t t, const int* plens, const int* pstrides, int size)
: type(t), lens(plens, plens+size), strides(pstrides, pstrides+size)
{}

void mlopenTensorDescriptor::CalculateStrides()
{
	strides.clear();
	strides.resize(lens.size(), 0);
	strides.back() = 1;
	std::partial_sum(lens.rbegin(), lens.rend()-1, strides.rbegin()+1, std::multiplies<int>());
}

const std::vector<int> mlopenTensorDescriptor::GetLengths() const
{
	return lens;
}
const std::vector<int> mlopenTensorDescriptor::GetStrides() const
{
	return strides;
}
int mlopenTensorDescriptor::GetSize() const
{
	assert(lens.size() == strides.size());
	return lens.size();
}
mlopenDataType_t mlopenTensorDescriptor::GetType() const
{
	return this->type;
}
bool mlopenTensorDescriptor::operator==(const mlopenTensorDescriptor& rhs) const
{
	assert(this->lens.size() == rhs.strides.size());
	return this->type == rhs.type and this->lens == rhs.lens and this->strides == rhs.strides;
}

bool mlopenTensorDescriptor::operator!=(const mlopenTensorDescriptor& rhs) const
{
	return not (*this == rhs);
}
