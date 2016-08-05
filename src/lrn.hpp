#ifndef _MLOPEN_LRN_HPP_
#define _MLOPEN_LRN_HPP_

#include <MLOpen.h>
#include <errors.hpp>
#include <Handle.hpp>
#include <Tensor.hpp>
#include "KernelCache.hpp"
#include "Common.hpp"
#include <vector>

struct mlopenLRNDescriptor {
	mlopenLRNDescriptor();
	mlopenLRNDescriptor(mlopenLRNMode_t m, const unsigned int pn, const double *pparms);

	mlopenLRNMode_t GetMode() const;
	unsigned int GetN() const;
	double GetAlpha() const;
	double GetBeta() const;
	double GetK() const;
	
	mlopenStatus_t Forward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		yDesc,
		Data_t								y);

	mlopenStatus_t Backward(
		mlopenHandle_t						handle,
		const void							*alpha,
		const mlopenTensorDescriptor_t		yDesc,
		const Data_t						y,
		const mlopenTensorDescriptor_t		dyDesc,
		const Data_t						dy,
		const mlopenTensorDescriptor_t		xDesc,
		const Data_t						x,
		const void							*beta,
		const mlopenTensorDescriptor_t		dxDesc,
		Data_t								dx);

	private:
	unsigned int lrnN;
	std::vector<double> parms;

	mlopenLRNMode_t mode;
};

#endif // _MLOPEN_LRN_HPP_
