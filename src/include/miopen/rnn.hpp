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

#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <miopen/mlo_internal.hpp>
#include <functional>
#include <numeric>

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
    RNNDescriptor(int hsz,
                  int layers = 1,
                  miopenRNNMode_t rmode = miopenRNNRELU,
                  miopenRNNInputMode_t inMode = miopenRNNskip,
                  miopenRNNDirectionMode_t bidir = miopenRNNunidirection,
                  miopenRNNBiasMode_t bias = miopenRNNwithBias,
                  miopenRNNAlgo_t amode = miopenRNNdefault,
                  miopenDataType_t dType = miopenFloat);


    size_t hsize; // DLOWELL: is this uniform over all layers?
    size_t nLayers;
    size_t nHiddenTensorsPerLayer; // TODO dlowell: set via constructor, or "set" functions
    size_t workspaceScale;
    
    size_t inputBatchLenSum;
    
    miopenRNNMode_t rnnMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNAlgo_t algoMode;
    miopenRNNInputMode_t inputMode;
    miopenRNNBiasMode_t biasMode;
    miopenDataType_t dataType;
    
    size_t GetWorkspaceSize(Handle& handle,
                                const int seqLength,
                                TensorDescriptor* xDesc) ;
    
    size_t GetReserveSize(Handle& handle,
                                const int seqLength,
                                TensorDescriptor* xDesc) ;
    
    size_t
    GetParamsSize(Handle& handle, const TensorDescriptor& xDesc, miopenDataType_t dtype) const;

    void GetLayerParam(Handle& handle,
                       const TensorDescriptor& xDesc,
                       const TensorDescriptor& wDesc,
                       ConstData_t w,
                       const int layerID,
                       const TensorDescriptor& paramDesc,
                       size_t paramOffset) const;

    void GetLayerBias(Handle& handle,
                      const TensorDescriptor& xDesc,
                      const TensorDescriptor& wDesc,
                      ConstData_t w,
                      const int layerID,
                      const TensorDescriptor& biasDesc,
                      size_t biasOffset) const;

    
    void RNNForwardTraining(Handle& handle,
                        const int seqLen,
                    	TensorDescriptor* xDesc,
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
                        size_t reserveSpaceSize) const;

    
    
    void RNNForwardInference(Handle& handle,
                             const int seqLen,
                             TensorDescriptor* xDesc,
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
    
    
    
    
    void RNNBackwardData(Handle& handle,
                            const int seqLen,
                        	TensorDescriptor* yDesc,
                            ConstData_t y,
                        	TensorDescriptor* dyDesc,
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
                        	TensorDescriptor* dxDesc,
                            Data_t dx,
                        	const TensorDescriptor& dhxDesc,
                            Data_t dhx,
                        	const TensorDescriptor& dcxDesc,
                            Data_t dcx,
                            Data_t workSpace,
                            size_t workSpaceSize,
                            ConstData_t reserveSpace,
                            size_t reserveSpaceSize) const;
    
    
    void RNNBackwardWeights(Handle& handle,
                            const int seqLen,
                        	TensorDescriptor* xDesc,
                            ConstData_t x,
                        	const TensorDescriptor& hxDesc,
                            ConstData_t hx,
                        	TensorDescriptor* yDesc,
                            ConstData_t y,
                        	const TensorDescriptor& dwDesc,
                            Data_t dw,
                            ConstData_t workSpace,
                            size_t workSpaceSize,
                            ConstData_t reserveSpace,
                            size_t reserveSpaceSize) const;
    
    
    // DLOWELL : These will be implemented once all the other elements are in place
    
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

