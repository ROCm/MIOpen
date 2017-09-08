#include <miopen/rnn.hpp>
#include <miopen/errors.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)


// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

namespace miopen {

RNNDescriptor::RNNDescriptor(){
    seqLength = 1;
    nlayers   = 1;
    rnnMode   = miopenRNNRELU;
    dirMode   = miopenRNNunidirection;
    algoMode  = miopenRNNdefault;
    inputMode = miopenRNNlinear;    
}
    
RNNDescriptor::RNNDescriptor(int sLength, int layer, miopenRNNDirectionMode_t bidir)
: seqLength(sLength), nlayers(layer), dirMode(bidir)
{
    rnnMode   = miopenRNNRELU;
    algoMode  = miopenRNNdefault;
    inputMode = miopenRNNlinear; 
}

RNNDescriptor::RNNDescriptor(miopenRNNMode_t p_mode, int sLength, int layer, miopenRNNDirectionMode_t bidir)
: seqLength(sLength), nlayers(layer), rnnMode(p_mode), dirMode(bidir)
{
    algoMode  = miopenRNNdefault;
    inputMode = miopenRNNlinear; 	
}

RNNDescriptor::RNNDescriptor(int hsz,
                    int layers,
                    miopenRNNInputMode_t inMode, 
                    miopenRNNDirectionMode_t bidir, 
                    miopenRNNMode_t rmode,
                    miopenRNNAlgo_t amode,
                    miopenDataType_t dType)
{
    hsize = seqLength = hsz;
    nlayers = layers;
    
    inputMode = inMode;
    dirMode   = bidir;
    rnnMode   = rmode;
    algoMode  = amode;
    dataType  = dType;	
}


/*
 //TODO (dlowell) these require an array_view for the array of tensor descriptors
size_t RNNDescriptor::GetWorkspaceSize(Handle& handle,
                                const int seqLength,
                                const TensorDescriptor& xDesc){
    size_t x = 0;
    return x;
}
        
        
        
size_t RNNDescriptor::GetReserveSize(Handle& handle,
                                const int seqLength,
                                const TensorDescriptor& *Desc){
    size_t x = 0;
    return x;    
}
*/


size_t RNNDescriptor::GetParamsSize(Handle& handle,
                        const TensorDescriptor& xDesc,
                        miopenDataType_t dtype) const
{
    size_t x = 0;
    return x;    
}



void RNNDescriptor::GetLayerParam(Handle& handle,
                        const TensorDescriptor& xDesc,
                        const TensorDescriptor& wDesc,
                        ConstData_t w,
                        const int layerID,
                        const TensorDescriptor& paramDesc,
                        Data_t **layerParam) const
{
    
}



void RNNDescriptor::GetLayerBias(Handle& handle,
                        const TensorDescriptor& xDesc,
                        const TensorDescriptor& wDesc,
                        ConstData_t w,
                        const int layerID,
                        const TensorDescriptor& biasDesc,
                        Data_t **layerBias) const
{
    
}

        
void RNNDescriptor::ForwardInferRNNCell(Handle& handle,
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
                            size_t workSpaceSize) const
{
    
}



void RNNDescriptor::ForwardTrainRNNCell(Handle& handle,
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
                            size_t reserveSpaceSize) const
{
}



void RNNDescriptor::BackwardDataRNNCell(Handle& handle,
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
                            size_t reserveSpaceSize) const
{
    
}



void RNNDescriptor::BackwardWeightsRNNCell(Handle& handle,
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
                            size_t reserveSpaceSize) const
{
    
}
        
} // namespace miopen

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#endif
