#ifndef _MLOPEN_ACTIV_HPP_
#define _MLOPEN_ACTIV_HPP_

#include <mlopen.h>
#include <mlopen/errors.hpp>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/common.hpp>
#include <vector>

namespace mlopen {

struct ActivationDescriptor : mlopenActivationDescriptor{
	ActivationDescriptor();
	ActivationDescriptor(mlopenActivationMode_t m, const double *pparms);

	mlopenActivationMode_t GetMode() const;
	double GetAlpha() const;
	double GetBeta() const;
	double GetPower() const;
	
	mlopenStatus_t Forward(
		Handle						&handle,
		const void					*alpha,
		const TensorDescriptor		&xDesc,
		ConstData_t				x,
		const void					*beta,
		const TensorDescriptor		&yDesc,
		Data_t						y,
		bool                        do_backward,
		Data_t						workSpace);

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

	friend std::ostream& operator<< (std::ostream& stream, const ActivationDescriptor& x);

private:
	std::vector<double> parms;

	mlopenActivationMode_t mode;
};

}  // namespace mlopen
MLOPEN_DEFINE_OBJECT(mlopenActivationDescriptor, mlopen::ActivationDescriptor);
#endif // _MLOPEN_ACTIV_HPP_
