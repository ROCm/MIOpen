#include <mlopen/tensor.hpp>
#include <mlopen/errors.hpp>
#include <string>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace mlopen {

TensorDescriptor::TensorDescriptor() {}

TensorDescriptor::TensorDescriptor(mlopenDataType_t t, std::initializer_list<int> plens)
: lens(plens), type(t)
{
	this->CalculateStrides();
}
	
TensorDescriptor::TensorDescriptor(mlopenDataType_t t, std::initializer_list<int> plens, std::initializer_list<int> pstrides)
: lens(plens), strides(pstrides), type(t)
{}

TensorDescriptor::TensorDescriptor(mlopenDataType_t t, const int* plens, int size)
: lens(plens, plens+size), type(t)
{
	this->CalculateStrides();
}
TensorDescriptor::TensorDescriptor(mlopenDataType_t t, const int* plens, const int* pstrides, int size)
: lens(plens, plens+size), strides(pstrides, pstrides+size), type(t)
{}

void TensorDescriptor::CalculateStrides()
{
	strides.clear();
	strides.resize(lens.size(), 0);
	strides.back() = 1;
	std::partial_sum(lens.rbegin(), lens.rend()-1, strides.rbegin()+1, std::multiplies<int>());
}

const std::vector<int>& TensorDescriptor::GetLengths() const
{
	return lens;
}
const std::vector<int>& TensorDescriptor::GetStrides() const
{
	return strides;
}
int TensorDescriptor::GetSize() const
{
	assert(lens.size() == strides.size());
	return lens.size();
}
int TensorDescriptor::GetElementSize() const
{
	assert(lens.size() == strides.size());
	return std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<int>());
}
mlopenDataType_t TensorDescriptor::GetType() const
{
	return this->type;
}

int TensorDescriptor::GetIndex(std::initializer_list<int> l) const
{
	assert(l.size() <= this->GetSize());
	return std::inner_product(l.begin(), l.end(), strides.begin(), 0);
}

bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
{
	assert(this->lens.size() == rhs.strides.size());
	return this->type == rhs.type && this->lens == rhs.lens && this->strides == rhs.strides;
}

bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const
{
	return ! (*this == rhs);
}

std::string TensorDescriptor::ToString() const
{
	std::string result;
	for(auto i:this->lens)
	{
		result += std::to_string(i) + ", ";
	}
	return result.substr(0, result.length()-2);
}

} // namespace mlopen

// TODO(paul): Remove
int mlopenGetTensorIndex(mlopenTensorDescriptor_t tensorDesc, std::initializer_list<int> indices)
{
	return mlopen::deref(tensorDesc).GetIndex(indices);
}
