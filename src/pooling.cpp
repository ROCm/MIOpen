#include <mlopen/pooling.hpp>
#include <cassert>
#include <cmath>

namespace mlopen {

PoolingDescriptor::PoolingDescriptor() {}

PoolingDescriptor::PoolingDescriptor(mlopenPoolingMode_t m,
		const int *plens,
		const int *ppads,
		const int *pstrides,
		int			size) : lens(plens, plens+size), strides(pstrides, pstrides+size), pads(ppads, ppads+size), mode(m) {}

mlopenPoolingMode_t PoolingDescriptor::GetMode() const
{
	return(mode);
}

const std::vector<int>& PoolingDescriptor::GetLengths() const
{
	return lens;
}
const std::vector<int>& PoolingDescriptor::GetStrides() const
{
	return strides;
}

const std::vector<int>& PoolingDescriptor::GetPads() const
{
	return pads;
}

mlopenPoolingMode_t PoolingDescriptor::GetMode()
{
	return mode;
}

int PoolingDescriptor::GetSize() const
{
	assert(lens.size() == strides.size() && lens.size() == pads.size());
	return lens.size();
}

std::tuple<int, int, int, int> PoolingDescriptor::GetForwardOutputDim(
		const TensorDescriptor				&tensorDesc) const {

	assert(tensorDesc.GetLengths().size() == 4);

	int input_n;
	int input_c;
	int input_h;
	int input_w;

	std::tie(input_n, input_c, input_h, input_w) = mlopen::tie4(tensorDesc.GetLengths());

	int u, v, pad_h, pad_w, window_h, window_w;
	std::tie(u, v) = mlopen::tie2(GetStrides());
	std::tie(pad_h, pad_w) = mlopen::tie2(GetPads());
	std::tie(window_h, window_w) = mlopen::tie2(GetLengths());

	return std::make_tuple(input_n, input_c, 
	ceil((input_h - window_h + 2*pad_h) / float(u)) + 1,
	ceil((input_w - window_w + 2*pad_w) / float(v)) + 1);

}

} // namespace mlopen
