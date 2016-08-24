#ifndef GUARD_MLOPEN_TENSOR_DRIVER_HPP
#define GUARD_MLOPEN_TENSOR_DRIVER_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include "mlopenTensor.hpp"
#include <MLOpen.h>

std::vector<int> GetTensorLengths(mlopenTensorDescriptor_t &tensor){
	int n;	
	int c;	
	int h;	
	int w;	

	 mlopenGet4dTensorDescriptorLengths(tensor, 
			 &n, &c, &h, &w);

	 return std::vector<int> ({n, c, h, w});
}

std::vector<int> GetTensorStrides(mlopenTensorDescriptor_t &tensor){
	int nstride;	
	int cstride;	
	int hstride;	
	int wstride;	

	 mlopenGet4dTensorDescriptorStrides(tensor, 
			 &nstride,
			 &cstride,
			 &hstride,
			 &wstride);

	 return std::vector<int> ({nstride, cstride, hstride, wstride});
}

int SetTensor4d(mlopenTensorDescriptor_t t, std::vector<int> &len) {
	return mlopenSet4dTensorDescriptor(
			t,
			mlopenFloat,
			UNPACK_VEC4(len));
}

size_t GetTensorSize(mlopenTensorDescriptor_t &tensor) {
	std::vector<int> len = GetTensorLengths(tensor);
	size_t sz = std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());

	return sz;
}
#endif // GUARD_MLOPEN_TENSOR_DRIVER_HPP
