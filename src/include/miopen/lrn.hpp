#ifndef MIOPEN_LRN_HPP_
#define MIOPEN_LRN_HPP_

#include <miopen.h>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <vector>

namespace miopen {

struct LRNDescriptor : miopenLRNDescriptor{
	LRNDescriptor();
	LRNDescriptor(miopenLRNMode_t m, unsigned int pn, const double *pparms);

	miopenLRNMode_t GetMode() const;
	unsigned int GetN() const;
	double GetAlpha() const;
	double GetBeta() const;
	double GetK() const;
	
	miopenStatus_t Forward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&xDesc,
		ConstData_t				x,
		const void					*beta,
		const TensorDescriptor		&yDesc,
		Data_t						y,
		bool                        do_backward,
		Data_t						workSpace);

	miopenStatus_t Backward(
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

	friend std::ostream& operator<< (std::ostream& stream, const LRNDescriptor& x);

private:
	unsigned int lrnN = 0;
	std::vector<double> parms;

	miopenLRNMode_t mode;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenLRNDescriptor, miopen::LRNDescriptor);
#endif // _MIOPEN_LRN_HPP_
