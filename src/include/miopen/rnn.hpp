//<<<<<<< rnngpu
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
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor.hpp>
#include <vector>
namespace miopen {
	
struct RNNDescriptor : miopenRNNDescriptor
{

    RNNDescriptor(int seqLength = 1, int layer = 1, int bidir = 0, int bias = 0);
    RNNDescriptor(miopenRNNMode_t p_mode, int seqLength = 1, int layer = 1, int bidir = 0, int bias = 0);

void RNNForwardTraining(Handle& handle,
	const int seqLen,
//	const TensorDescriptor& xDesc,
	ConstData_t x,
//	const TensorDescriptor& hxDesc,
	ConstData_t hx,
//	const TensorDescriptor& cxDesc,
	ConstData_t cx,
//	const TensorDescriptor& wDesc,
	ConstData_t w,
//	const TensorDescriptor& yDesc,
	Data_t y,
//	const TensorDescriptor& hyDesc,
	Data_t hy,
//	const TensorDescriptor& cyDesc,
	Data_t cy,
	Data_t workSpace,
	size_t workSpaceSize,
	Data_t reserveSpace,
	size_t reserveSpaceSize,
	const std::vector<int> &in_n,
	const int in_h,
	const int hy_d,
	const int hy_n,
	const int hy_h,
	const int out_h) const;

void RNNBackwardData(Handle& handle,
	const int seqLen,
//	const TensorDescriptor& yDesc,
	ConstData_t y,
//	const TensorDescriptor& dyDesc,
	ConstData_t dy,
//	const TensorDescriptor& dhyDesc,
	ConstData_t dhy,
//	const TensorDescriptor& dcyDesc,
	ConstData_t dcy,
//	const TensorDescriptor& wDesc,
	ConstData_t w,
//	const TensorDescriptor& hxDesc,
	ConstData_t hx,
//	const TensorDescriptor& cxDesc,
	ConstData_t cx,
//	const TensorDescriptor& dxDesc,
	Data_t dx,
//	const TensorDescriptor& dhxDesc,
	Data_t dhx,
//	const TensorDescriptor& dcxDesc,
	Data_t dcx,
	Data_t workSpace,
	size_t workSpaceSize,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize,
	const std::vector<int> &in_n,
	const int in_h,
	const int hy_d,
	const int hy_n,
	const int hy_h,
	const int out_h) const;

void RNNBackwardWeights(Handle& handle,
	const int seqLen,
//	const TensorDescriptor& xDesc,
	ConstData_t x,
//	const TensorDescriptor& hxDesc,
	ConstData_t hx,
//	const TensorDescriptor& dyDesc,
	ConstData_t dy,
	ConstData_t workSpace,
	size_t workSpaceSize,
//	const TensorDescriptor& dwDesc,
	Data_t dw,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize,
	const std::vector<int> &in_n,
	const int in_h,
	const int hy_d,
	const int hy_n,
	const int hy_h,
	const int out_h) const;
                                                                    
    miopenRNNMode_t mode;
    int seqLength;
    int layer;
    int bidir;
	int bias;
};
} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenRNNDescriptor, miopen::RNNDescriptor);

#endif // GUARD_MIOPEN_RNN_HPP_
//=======
#ifndef GUARD_MIOPEN_RNN_HPP_
#define GUARD_MIOPEN_RNN_HPP_

#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <miopen/mlo_internal.hpp>
#include <functional>

namespace miopen {

struct PerfField
{
    std::string name;
    float time;
    std::size_t workspace;

    bool operator<(const PerfField& p) const { return (time < p.time); }
};

struct RNNDescriptor : miopenRNNDescriptor
{

    RNNDescriptor();
    RNNDescriptor(int sLength, int layer, miopenRNNDirectionMode_t bidir = miopenRNNunidirection);
    RNNDescriptor(miopenRNNMode_t p_mode, int sLength, int layer, miopenRNNDirectionMode_t bidir);
    RNNDescriptor(int hsz,
                  int layers,
                  miopenRNNInputMode_t inMode,
                  miopenRNNDirectionMode_t bidir,
                  miopenRNNMode_t rmode,
                  miopenRNNAlgo_t amode,
                  miopenDataType_t dType = miopenFloat);

    int hsize;
    int seqLength;
    int nlayers;

    miopenRNNMode_t rnnMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNAlgo_t algoMode;
    miopenRNNInputMode_t inputMode;
    miopenDataType_t dataType;
    
    size_t GetWorkspaceSize(Handle& handle,
                                const int seqLength,
                                TensorDescriptor** xDesc) const;
    
    size_t GetReserveSize(Handle& handle,
                                const int seqLength,
                                TensorDescriptor** xDesc) const;
    
    size_t
    GetParamsSize(Handle& handle, const TensorDescriptor& xDesc, miopenDataType_t dtype) const;

    void GetLayerParam(Handle& handle,
                       const TensorDescriptor& xDesc,
                       const TensorDescriptor& wDesc,
                       ConstData_t w,
                       const int layerID,
                       const TensorDescriptor& paramDesc,
                       Data_t** layerParam) const;

    void GetLayerBias(Handle& handle,
                      const TensorDescriptor& xDesc,
                      const TensorDescriptor& wDesc,
                      ConstData_t w,
                      const int layerID,
                      const TensorDescriptor& biasDesc,
                      Data_t** layerBias) const;

    
    void ForwardRNNTrain(Handle& handle,
                             const TensorDescriptor& xDesc,
                             ConstData_t x,
                             const TensorDescriptor& hxDesc,
                             ConstData_t hx,
                             const TensorDescriptor& cxDesc,
                             ConstData_t cx,
                             const TensorDescriptor& wDesc,
                             ConstData_t w,
                             const TensorDescriptor& yDesc,
                             ConstData_t y,
                             const TensorDescriptor& hyDesc,
                             Data_t hy,
                             const TensorDescriptor& cyDesc,
                             ConstData_t cy,
                             Data_t workSpace,
                             size_t workSpaceSize,
                             Data_t reserveSpace,
                             size_t reserveSpaceSize) const;

    void ForwardRNNInference(Handle& handle,
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
                             size_t workSpaceSize) const;
    
    void ForwardRNNInferCell(Handle& handle,
                             const TensorDescriptor& xDesc,
                             ConstData_t x,
                             const TensorDescriptor& hxDesc,
                             ConstData_t hx,
                             const TensorDescriptor& wDesc,
                             ConstData_t w,
                             const TensorDescriptor& yDesc,
                             Data_t y,
                             const TensorDescriptor& hyDesc,
                             Data_t hy,
                             Data_t workSpace,
                             size_t workSpaceSize) const;

    void ForwardRNNTrainCell(Handle& handle,
                             const TensorDescriptor& xDesc,
                             ConstData_t x,
                             const TensorDescriptor& hxDesc,
                             ConstData_t hx,
                             const TensorDescriptor& wDesc,
                             ConstData_t w,
                             const TensorDescriptor& yDesc,
                             Data_t y,
                             const TensorDescriptor& hyDesc,
                             Data_t hy,
                             Data_t workSpace,
                             size_t workSpaceSize,
                             Data_t reserveSpace,
                             size_t reserveSpaceSize) const;

    void BackwardRNNDataCell(Handle& handle,
                             const TensorDescriptor& yDesc,
                             ConstData_t y,
                             const TensorDescriptor& dyDesc,
                             ConstData_t dy,
                             const TensorDescriptor& dhyDesc,
                             ConstData_t dhy,
                             const TensorDescriptor& wDesc,
                             ConstData_t w,
                             const TensorDescriptor& hxDesc,
                             ConstData_t hx,
                             const TensorDescriptor& dxDesc,
                             Data_t dx,
                             const TensorDescriptor& dhxDesc,
                             Data_t dhx,
                             Data_t workSpace,
                             size_t workSpaceSize,
                             ConstData_t reserveSpace,
                             size_t reserveSpaceSize) const;

    void BackwardRNNWeightsCell(Handle& handle,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& hxDesc,
                                ConstData_t hx,
                                const TensorDescriptor& yDesc,
                                ConstData_t y,
                                const TensorDescriptor& dwDesc,
                                Data_t dw,
                                ConstData_t workSpace,
                                size_t workSpaceSize,
                                ConstData_t reserveSpace,
                                size_t reserveSpaceSize) const;
};

std::ostream& operator<<(std::ostream& stream, const RNNDescriptor& c);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenRNNDescriptor, miopen::RNNDescriptor);

#endif // GUARD_MIOPEN_RNN_HPP_
//>>>>>>> rnn
