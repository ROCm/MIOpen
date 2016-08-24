#include <mlopen/pooling.hpp>
#include <cassert>

namespace mlopen {

PoolingDescriptor::PoolingDescriptor() {}

PoolingDescriptor::PoolingDescriptor(mlopenPoolingMode_t m,
		const int *plens,
		const int *ppads,
		const int *pstrides,
		int			size) : mode(m), lens(plens, plens+size), pads(ppads, ppads+size), strides(pstrides, pstrides+size) {}

const std::vector<int>& mlopenPoolingDescriptor::GetLengths() const
{
	return lens;
}
const std::vector<int>& mlopenPoolingDescriptor::GetStrides() const
{
	return strides;
}

const std::vector<int>& mlopenPoolingDescriptor::GetPads() const
{
	return pads;
}
int mlopenPoolingDescriptor::GetSize() const
{
	assert(lens.size() == strides.size() && lens.size() == pads.size());
	return lens.size();
}

mlopenPoolingMode_t mlopenPoolingDescriptor::GetMode() const
{
	return this->mode;
}

mlopenStatus_t mlopenPoolingDescriptor::GetForwardOutputDim(
		const TensorDescriptor				&tensorDesc,
		int									*n,
		int									*c,
		int									*h,
		int									*w) {

	if(tensorDesc == nullptr) {
		return mlopenStatusBadParm;
	}

	int input_n, input_c, input_h, input_w;
	std::tie(input_n, input_c, input_h, input_w) = mlopen::tie4(tensorDesc.GetLengths());

	*n = input_n;
	*c = input_c;
	
	int u, v, pad_h, pad_w, window_h, window_w;
	std::tie(u, v) = mlopen::tie2(GetStrides());
	std::tie(pad_h, pad_w) = mlopen::tie2(GetPads());
	std::tie(window_h, window_w) = mlopen::tie2(GetLengths());

	*h = (input_h - window_h + 2*pad_h) / u + 1;
	*w = (input_w - window_w + 2*pad_w) / v + 1;

	return mlopenStatusSuccess;
}
}
