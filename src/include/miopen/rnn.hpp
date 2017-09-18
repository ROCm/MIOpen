/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_MIOPEN_RNN_HPP_
#define GUARD_MIOPEN_RNN_HPP_

#include <functional>
#include <miopen/common.hpp>
//#include <miopen/conv_algo_name.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

/*
using WinogradKernelParams = std::tuple<int, int, int, int, int, int, int, int, bool>;

struct PerfField
{
std::string name;
float time;
std::size_t workspace;

bool operator<(const PerfField& p) const { return (time < p.time); }
};
*/

struct RNNDescriptor : miopenRNNDescriptor
{

    RNNDescriptor(int seqLength = 1, int layer = 1, int bidir = 0, int bias = 0);
    RNNDescriptor(miopenRNNMode_t p_mode, int seqLength = 1, int layer = 1, int bidir = 0, int bias = 0);

    /*
std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
GetForwardOutputDim(const TensorDescriptor& inputTensorDesc,
                    const TensorDescriptor& filterDesc) const;
TensorDescriptor GetForwardOutputTensor(const TensorDescriptor& inputTensorDesc,
                                        const TensorDescriptor& filterDesc) const;

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
GetBackwardsWeightsDim(const TensorDescriptor& inputTensorDesc,
                       const TensorDescriptor& outputTensorDesc) const;
TensorDescriptor GetBackwardWeightsTensor(const TensorDescriptor& inputTensorDesc,
                                          const TensorDescriptor& outputTensorDesc) const;

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
GetBackwardOutputDim(const TensorDescriptor& outputTensorDesc,
                     const TensorDescriptor& filterDesc) const;
TensorDescriptor GetBackwardOutputTensor(const TensorDescriptor& outputTensorDesc,
                                         const TensorDescriptor& filterDesc) const;

size_t ForwardGetWorkSpaceSizeGEMM(Handle& handle,
                                   const TensorDescriptor& wDesc,
                                   const TensorDescriptor& yDesc) const;

size_t ForwardGetWorkSpaceSizeFFT(const TensorDescriptor& wDesc,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& yDesc) const;

bool IsWinograd3x3Supported(Handle& handle,
                            bool direction,
                            const TensorDescriptor& wDesc,
                            const TensorDescriptor& xDesc) const;

bool IsBwdWeightsDirectSupported(const TensorDescriptor& wDesc) const;

bool IsDirectSupported(const TensorDescriptor& wDesc) const;

size_t ForwardGetWorkSpaceSize(Handle& handle,
                               const TensorDescriptor& wDesc,
                               const TensorDescriptor& xDesc,
                               const TensorDescriptor& yDesc) const;

void FindConvFwdAlgorithm(Handle& handle,
                          const TensorDescriptor& xDesc,
                          ConstData_t x,
                          const TensorDescriptor& wDesc,
                          ConstData_t w,
                          const TensorDescriptor& yDesc,
                          ConstData_t y,
                          int requestAlgoCount,
                          int* returnedAlgoCount,
                          miopenConvAlgoPerf_t* perfResults,
                          Data_t workSpace,
                          size_t workSpaceSize,
                          bool exhaustiveSearch) const;

int FindWinogradKernel(Handle& handle,
                       const TensorDescriptor& xDesc,
                       const TensorDescriptor& wDesc,
                       const TensorDescriptor& yDesc,
                       WinogradKernelParams& k_p,
                       KernelInvoke& kernel,
                       int direction) const;

int FindFwdFFTKernel(Handle& handle,
                     const TensorDescriptor& xDesc,
                     const TensorDescriptor& wDesc,
                     const TensorDescriptor& yDesc,
                     size_t workSpaceSize,
                     std::vector<KernelInvoke>& kernels) const;

float ExecuteFwdFFTKernel(Handle& handle,
                          const TensorDescriptor& xDesc,
                          ConstData_t x,
                          const TensorDescriptor& wDesc,
                          ConstData_t w,
                          const TensorDescriptor& yDesc,
                          Data_t y,
                          Data_t workSpace,
                          size_t workSpaceSize,
                          bool timed = false) const;

int FindBwdFFTKernel(Handle& handle,
                     const TensorDescriptor& dyDesc,
                     const TensorDescriptor& wDesc,
                     const TensorDescriptor& dxDesc,
                     size_t workSpaceSize,
                     std::vector<KernelInvoke>& kernels) const;

float ExecuteBwdFFTKernel(Handle& handle,
                          const TensorDescriptor& dyDesc,
                          ConstData_t dy,
                          const TensorDescriptor& wDesc,
                          ConstData_t w,
                          const TensorDescriptor& dxDesc,
                          Data_t dx,
                          Data_t workSpace,
                          size_t workSpaceSize,
                          bool timed = false) const;

int FindDirectKernel(Handle& handle,
                     const TensorDescriptor& xDesc,
                     const TensorDescriptor& wDesc,
                     const TensorDescriptor& yDesc,
                     std::vector<KernelInvoke>& kernels,
                     bool exhaustiveSearch,
                     int direction) const;
					 */

void RNNForwardTraining(Handle& handle,
	const int seqLen,
	const TensorDescriptor& xDesc,
	ConstData_t x,
	const TensorDescriptor& hxDesc,
	ConstData_t hx,
	const TensorDescriptor& cxDesc,
	ConstData_t cx,
	const TensorDescriptor& wDesc,
	ConstData_t w,
	const TensorDescriptor& yDesc,
	Data_t y,
	const TensorDescriptor& hyDesc,
	Data_t hy,
	const TensorDescriptor& cyDesc,
	Data_t cy,
	Data_t workSpace,
	size_t workSpaceSize,
	Data_t reserveSpace,
	size_t reserveSpaceSize, 
	const int *in_n,
	const int in_h,
	const int out_h,
	const int hy_d,
	const int hy_n,
	const int hy_h) const;

/*
size_t BackwardDataGetWorkSpaceSizeGEMM(Handle& handle,
                                        const TensorDescriptor& wDesc,
                                        const TensorDescriptor& dyDesc) const;

size_t BackwardGetWorkSpaceSizeFFT(const TensorDescriptor& wDesc,
                                   const TensorDescriptor& dyDesc,
                                   const TensorDescriptor& dxDesc) const;

size_t BackwardDataGetWorkSpaceSize(Handle& handle,
                                    const TensorDescriptor& wDesc,
                                    const TensorDescriptor& dyDesc,
                                    const TensorDescriptor& dxDesc) const;

void FindConvBwdDataAlgorithm(Handle& handle,
                              const TensorDescriptor& dyDesc,
                              ConstData_t dy,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& dxDesc,
                              ConstData_t dx,
                              int requestAlgoCount,
                              int* returnedAlgoCount,
                              miopenConvAlgoPerf_t* perfResults,
                              Data_t workSpace,
                              size_t workSpaceSize,
                              bool exhaustiveSearch) const;
*/

void RNNBackwardData(Handle& handle,
	const int seqLen,
	const TensorDescriptor& yDesc,
	ConstData_t y,
	const TensorDescriptor& dyDesc,
	ConstData_t dy,
	const TensorDescriptor& dhyDesc,
	ConstData_t dhy,
	const TensorDescriptor& dcyDesc,
	ConstData_t dcy,
	const TensorDescriptor& wDesc,
	ConstData_t w,
	const TensorDescriptor& hxDesc,
	ConstData_t hx,
	const TensorDescriptor& cxDesc,
	ConstData_t cx,
	const TensorDescriptor& dxDesc,
	Data_t dx,
	const TensorDescriptor& dhxDesc,
	Data_t dhx,
	const TensorDescriptor& dcxDesc,
	Data_t dcx,
	Data_t workSpace,
	size_t workSpaceSize,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize) const;

/*
size_t ConvolutionBackwardWeightsGetWorkSpaceSize(Handle& handle,
                                                  const TensorDescriptor& dyDesc,
                                                  const TensorDescriptor& xDesc,
                                                  const TensorDescriptor& dwDesc) const;

size_t BackwardWeightsGetWorkSpaceSizeGEMM(Handle& handle,
                                           const TensorDescriptor& dyDesc,
                                           const TensorDescriptor& dwDesc) const;

size_t BackwardWeightsGetWorkSpaceSizeDirect(Handle& handle,
                                             const TensorDescriptor& dyDesc,
                                             const TensorDescriptor& xDesc,
                                             const TensorDescriptor& dwDesc) const;

void FindConvBwdWeightsAlgorithm(Handle& handle,
                                 const TensorDescriptor& dyDesc,
                                 ConstData_t dy,
                                 const TensorDescriptor& xDesc,
                                 ConstData_t x,
                                 const TensorDescriptor& dwDesc,
                                 ConstData_t dw,
                                 int requestAlgoCount,
                                 int* returnedAlgoCount,
                                 miopenConvAlgoPerf_t* perfResults,
                                 Data_t workSpace,
                                 size_t workSpaceSize,
                                 bool exhaustiveSearch) const;
								 */

void RNNBackwardWeights(Handle& handle,
	const int seqLen,
	const TensorDescriptor& xDesc,
	ConstData_t x,
	const TensorDescriptor& hxDesc,
	ConstData_t hx,
	const TensorDescriptor& dyDesc,
	ConstData_t dy,
	ConstData_t workSpace,
	size_t workSpaceSize,
	const TensorDescriptor& dwDesc,
	Data_t dw,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize) const;
                                                                    
    miopenRNNMode_t mode;
    int seqLength;
    int layer;
    int bidir;
	int bias;
};
/*
void ConvolutionBackwardBias(Handle& handle,
                             const void* alpha,
                             const TensorDescriptor& dyDesc,
                             ConstData_t dy,
                             const void* beta,
                             const TensorDescriptor& dbDesc,
                             Data_t db);

std::ostream& operator<<(std::ostream& stream, const RNNDescriptor& c);

*/

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenRNNDescriptor, miopen::RNNDescriptor);

#endif // GUARD_MIOPEN_RNN_HPP_