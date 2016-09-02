#ifndef _MLOPEN_LRN_HPP_
#define _MLOPEN_LRN_HPP_

#include <mlopen.h>
#include <mlopen/errors.hpp>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/common.hpp>
#include <vector>

namespace mlopen {

struct LRNDescriptor : mlopenLRNDescriptor{
	LRNDescriptor();
	LRNDescriptor(mlopenLRNMode_t m, unsigned int pn, const double *pparms);

	mlopenLRNMode_t GetMode() const;
	unsigned int GetN() const;
	double GetAlpha() const;
	double GetBeta() const;
	double GetK() const;
	
	mlopenStatus_t Forward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&xDesc,
		ConstData_t				x,
		const void					*beta,
		const TensorDescriptor		&yDesc,
		Data_t						y,
		bool                        do_backward,
		Data_t						workSpace,
		size_t						*workSpaceSize);

	mlopenStatus_t Backward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&yDesc,
		ConstData_t				y,
		const TensorDescriptor		&dyDesc,
		ConstData_t				dy,
		const TensorDescriptor		&xDesc,
		ConstData_t				x,
		const void					*beta,
		const TensorDescriptor		&dxDesc,
		Data_t						dx,
		ConstData_t				workSpace);

	private:
	unsigned int lrnN = 0;
	std::vector<double> parms;

	mlopenLRNMode_t mode;
};

} // namespace mlopen
MLOPEN_DEFINE_OBJECT(mlopenLRNDescriptor, mlopen::LRNDescriptor);
#endif // _MLOPEN_LRN_HPP_
