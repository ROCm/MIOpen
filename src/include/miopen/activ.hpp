#ifndef MIOPEN_ACTIV_HPP_
#define MIOPEN_ACTIV_HPP_

#include <miopen.h>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <vector>

namespace miopen {

struct ActivationDescriptor : miopenActivationDescriptor{
	ActivationDescriptor();
	ActivationDescriptor(miopenActivationMode_t m, const double *pparms);

	miopenActivationMode_t GetMode() const;
	double GetAlpha() const;
	double GetBeta() const;
	double GetPower() const;
	
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

	friend std::ostream& operator<< (std::ostream& stream, const ActivationDescriptor& x);

private:
	std::vector<double> parms;

	miopenActivationMode_t mode;
};

}  // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenActivationDescriptor, miopen::ActivationDescriptor);
#endif // _MIOPEN_ACTIV_HPP_
