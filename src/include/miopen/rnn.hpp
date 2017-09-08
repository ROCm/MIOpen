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

    bool operator < (const PerfField &p) const 
    {
        return (time < p.time);
    }
};

struct RNNDescriptor : miopenRNNDescriptor {
	
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
    
    miopenRNNMode_t          rnnMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNAlgo_t          algoMode;
    miopenRNNInputMode_t     inputMode;
    miopenDataType_t         dataType;
    /*
        //TODO (dlowell)  this requires array_view for the array of tensor descriptors
     
    size_t GetWorkspaceSize(Handle& handle,
                                const int seqLength,
                                const TensorDescriptor& *xDesc) const;
    size_t GetReserveSize(Handle& handle,
                                const int seqLength,
                                const TensorDescriptor *xDesc) const;  
     */
    size_t GetParamsSize(Handle& handle,
                        const TensorDescriptor& xDesc,
                        miopenDataType_t dtype) const;
   
    void GetLayerParam(Handle& handle,
                        const TensorDescriptor& xDesc,
                        const TensorDescriptor& wDesc,
                        ConstData_t w,
                        const int layerID,
                        const TensorDescriptor& paramDesc,
                        Data_t **layerParam) const;
    
    void GetLayerBias(Handle& handle,
                        const TensorDescriptor& xDesc,
                        const TensorDescriptor& wDesc,
                        ConstData_t w,
                        const int layerID,
                        const TensorDescriptor& biasDesc,
                        Data_t **layerBias) const;
    
    void ForwardInferRNNCell(Handle& handle,
                            const TensorDescriptor& xDesc,
                            ConstData_t x,
                            const TensorDescriptor& hxDesc,
                            ConstData_t hx,
                            const TensorDescriptor& wDesc,
                            ConstData_t w,
                            const TensorDescriptor& hyDesc,
                            ConstData_t hy,
                            const TensorDescriptor& yDesc,
                            Data_t y,
                            Data_t workSpace,
                            size_t workSpaceSize) const;
    
    void ForwardTrainRNNCell(Handle& handle,
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
    
    void BackwardDataRNNCell(Handle& handle,
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
    
    void BackwardWeightsRNNCell(Handle& handle,
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


std::ostream& operator<< (std::ostream& stream, const RNNDescriptor& c);

}  // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenRNNDescriptor, miopen::RNNDescriptor);

#endif // GUARD_MIOPEN_RNN_HPP_
