#ifndef GUARD_MIOPEN_CONVOLUTION_HPP_
#define GUARD_MIOPEN_CONVOLUTION_HPP_

#include <miopen.h>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/mlo_internal.hpp>
#include <functional>

namespace miopen {

using WinogradKernelParams = std::tuple<int, int, int, int, int, int>;

struct PerfField
{
    std::string name;
    float time;
    std::size_t workspace;

    bool operator < (const PerfField &p) const 
    {
        return (time < p.time);
    }
};

struct ConvolutionDescriptor : miopenConvolutionDescriptor {
	
	ConvolutionDescriptor(int p_pad_h = 0, int p_pad_w = 0, int p_u = 1, int p_v = 1, int p_upscalex = 1, int p_upscaley = 1);
	ConvolutionDescriptor(miopenConvolutionMode_t p_mode, int p_pad_h = 0, int p_pad_w = 0, int p_u = 1, int p_v = 1, int p_upscalex = 1, int p_upscaley = 1);

	std::tuple<int, int, int, int> GetForwardOutputDim(const TensorDescriptor& inputTensorDesc,
										const TensorDescriptor& filterDesc) const;
	TensorDescriptor GetForwardOutputTensor(const TensorDescriptor& inputTensorDesc,
										const TensorDescriptor& filterDesc) const;

	std::tuple<int, int, int, int> GetBackwardsWeightsDim(const TensorDescriptor& inputTensorDesc, 
										const TensorDescriptor& outputTensorDesc) const;
	TensorDescriptor GetBackwardWeightsTensor(const TensorDescriptor& inputTensorDesc,
										const TensorDescriptor& outputTensorDesc) const;

	std::tuple<int, int, int, int> GetBackwardOutputDim(const TensorDescriptor& outputTensorDesc,
										const TensorDescriptor& filterDesc) const;
	TensorDescriptor GetBackwardOutputTensor(const TensorDescriptor& outputTensorDesc,
										const TensorDescriptor& filterDesc) const;

	size_t ForwardGetWorkSpaceSizeGEMM(
        Handle&                     handle,
		const TensorDescriptor&		wDesc,
		const TensorDescriptor&		yDesc) const;

	size_t ForwardGetWorkSpaceSizeFFT(
		const TensorDescriptor&		wDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		yDesc) const;

	size_t ForwardGetWorkSpaceSize(
        Handle&                     handle,
		const TensorDescriptor&		wDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		yDesc) const;

	void FindConvFwdAlgorithm(Handle& handle,
		const TensorDescriptor&			xDesc,
		ConstData_t						x,
		const TensorDescriptor&			wDesc,
		ConstData_t						w,
		const TensorDescriptor&			yDesc,
		ConstData_t						y,
		int						requestAlgoCount,
		int								*returnedAlgoCount,
		miopenConvAlgoPerf_t			*perfResults,
		Data_t							workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch) const;

    int FindWinogradKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		const TensorDescriptor&			wDesc,
		const TensorDescriptor&			yDesc,
        WinogradKernelParams&           k_p,
        KernelInvoke&                   kernel,
        int                             direction) const;

    int FindFwdFFTKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		const TensorDescriptor&			wDesc,
		const TensorDescriptor&			yDesc,
		size_t							workSpaceSize,
        std::vector<KernelInvoke>&      kernels) const;

    float ExecuteFwdFFTKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		ConstData_t						x,
		const TensorDescriptor&			wDesc,
		ConstData_t						w,
		const TensorDescriptor&			yDesc,
		Data_t							y,
		Data_t							workSpace,
		size_t							workSpaceSize,
		bool							timed = false) const;

    int FindDirectKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		const TensorDescriptor&			wDesc,
		const TensorDescriptor&			yDesc,
        std::vector<KernelInvoke>&      kernels,
        bool                            exhaustiveSearch,
        int                             direction) const;

	void ConvolutionForward(Handle& handle,
		const void						*alpha,
		const TensorDescriptor&			xDesc,
		ConstData_t						x,
		const TensorDescriptor&			wDesc,
		ConstData_t						w,
		miopenConvFwdAlgorithm_t		algo,
		const void						*beta,
		const TensorDescriptor&			yDesc,
		Data_t							y,
		Data_t							workSpace,
		size_t							workSpaceSize) const;

	void FindConvBwdDataAlgorithm(Handle& handle,
		const TensorDescriptor&			dyDesc,
		ConstData_t						dy,
		const TensorDescriptor&			wDesc,
		ConstData_t						w,
		const TensorDescriptor&			dxDesc,
		ConstData_t						dx,
		int						requestAlgoCount,
		int								*returnedAlgoCount,
		miopenConvAlgoPerf_t			*perfResults,
		void							*workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch) const;

	void ConvolutionBackwardData(Handle& handle,
		const void						*alpha,
		const TensorDescriptor&			dyDesc,
		ConstData_t						dy,
		const TensorDescriptor&			wDesc,
		ConstData_t						w,
		miopenConvBwdDataAlgorithm_t	algo,
		const void						*beta,
		const TensorDescriptor&			dxDesc,
		Data_t							dx,
		void							*workSpace,
		size_t							workSpaceSize) const;

	size_t ConvolutionBackwardWeightsGetWorkSpaceSize(
        Handle&                     handle,
		const TensorDescriptor&		dyDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		dwDesc) const;

	size_t BackwardWeightsGetWorkSpaceSizeGEMM(
        Handle&                     handle,
		const TensorDescriptor&		dyDesc,
		const TensorDescriptor&		dwDesc) const;

	size_t BackwardWeightsGetWorkSpaceSizeDirect(
		Handle&						handle,
		const TensorDescriptor&		dyDesc,
		const TensorDescriptor&		xDesc,
		const TensorDescriptor&		dwDesc) const;

	void FindConvBwdWeightsAlgorithm(Handle& handle,
		const TensorDescriptor&			dyDesc,
		ConstData_t						dy,
		const TensorDescriptor&			xDesc,
		ConstData_t						x,
		const TensorDescriptor&			dwDesc,
		ConstData_t						dw,
		int						requestAlgoCount,
		int								*returnedAlgoCount,
		miopenConvAlgoPerf_t			*perfResults,
		Data_t							workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch) const;

	void ConvolutionBackwardWeights(Handle& handle,
		const void						*alpha,
		const TensorDescriptor&			dyDesc,
		ConstData_t						dy,
		const TensorDescriptor&			xDesc,
		ConstData_t						x,
		miopenConvBwdWeightsAlgorithm_t	algo,
		const void						*beta,
		const TensorDescriptor&			dwDesc,
		Data_t							dw,
		Data_t							workSpace,
		size_t							workSpaceSize) const;

	miopenConvolutionMode_t mode;
	int pad_h;
	int pad_w;
	int u;
	int v;
	int upscalex;
	int upscaley;
};

void ConvolutionBackwardBias(Handle& handle,
                             const void              *alpha,
                             const TensorDescriptor& dyDesc,
                             ConstData_t             dy,
                             const void              *beta,
                             const TensorDescriptor& dbDesc,
                             Data_t                  db);

std::ostream& operator<< (std::ostream& stream, const ConvolutionDescriptor& c);

}  // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenConvolutionDescriptor, miopen::ConvolutionDescriptor);

#endif // GUARD_MIOPEN_CONVOLUTION_HPP_
