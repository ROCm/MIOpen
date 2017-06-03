#ifndef GUARD_MIOPEN_TENSOR_DRIVER_HPP
#define GUARD_MIOPEN_TENSOR_DRIVER_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <miopen/tensor.hpp>
#include <miopen/tensor_extra.hpp>
#include <miopen/miopen.h>

std::vector<int> GetTensorLengths(miopenTensorDescriptor_t &tensor){
	int n;	
	int c;	
	int h;	
	int w;	

	 miopenGet4dTensorDescriptorLengths(tensor, 
			 &n, &c, &h, &w);

	 return std::vector<int> ({n, c, h, w});
}

std::vector<int> GetTensorStrides(miopenTensorDescriptor_t &tensor){
	int nstride;	
	int cstride;	
	int hstride;	
	int wstride;	

	 miopenGet4dTensorDescriptorStrides(tensor, 
			 &nstride,
			 &cstride,
			 &hstride,
			 &wstride);

	 return std::vector<int> ({nstride, cstride, hstride, wstride});
}

int SetTensor4d(miopenTensorDescriptor_t t, std::vector<int> &len) {
	return miopenSet4dTensorDescriptor(
			t,
			miopenFloat,
			UNPACK_VEC4(len));
}

size_t GetTensorSize(miopenTensorDescriptor_t &tensor) {
	std::vector<int> len = GetTensorLengths(tensor);
	size_t sz = std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());

	return sz;
}
#endif // GUARD_MIOPEN_TENSOR_DRIVER_HPP
