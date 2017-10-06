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
                  miopenRNNAlgo_t amode = miopenRNNdefault,
                  miopenDataType_t dType = miopenFloat);

    size_t hsize; // DLOWELL: is this uniform over all layers?
    size_t seqLength; //DLOWELL: remove?
    size_t nLayers;
    size_t nHiddenTensorsPerLayer; // TODO dlowell: set via constructor, or "set" functions
    size_t workspaceScale;
    
    size_t inputBatchLenSum;
    
    miopenRNNMode_t rnnMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNAlgo_t algoMode;
    miopenRNNInputMode_t inputMode;
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
                        size_t reserveSpaceSize/*,
                        const std::vector<int> &in_n,
                        const int in_h,
                        const int hy_d,
                        const int hy_n,
                        const int hy_h, // DLOWELL: These should all be internal
                        const int out_h*/) const;

    
    
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
                            size_t reserveSpaceSize/*,
                            const std::vector<int> &in_n,
                            const int in_h,
                            const int hy_d,
                            const int hy_n,
                            const int hy_h,
                            const int out_h*/) const;
    
    
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
                            size_t reserveSpaceSize/*,
                            const std::vector<int> &in_n,
                            const int in_h,
                            const int hy_d,
                            const int hy_n,
                            const int hy_h,
                            const int out_h*/) const;
    
    
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
