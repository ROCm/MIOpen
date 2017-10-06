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

RNNDescriptor::RNNDescriptor()
{
    seqLength     = 0;
    nLayers       = 1;
    hsize         = 0;
    inputBatchLenSum = 0;
    nHiddenTensorsPerLayer = 0;
    rnnMode   = miopenRNNRELU;
    dirMode   = miopenRNNunidirection;
    algoMode  = miopenRNNdefault;
    inputMode = miopenRNNskip;
}

RNNDescriptor::RNNDescriptor(int hsz,
                             int layers,
                             miopenRNNMode_t rmode,
                             miopenRNNInputMode_t inMode,
                             miopenRNNDirectionMode_t bidir,
                             miopenRNNAlgo_t amode,
                             miopenDataType_t dType)
{
    hsize     = hsz;
    nLayers   = layers;
    inputMode = inMode;
    dirMode   = bidir;
    rnnMode   = rmode;
    algoMode  = amode;
    dataType  = dType;
    if(dType != miopenFloat)
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Only float datatype is supported");
    }
    assert(rmode < 4);
    switch (rmode)
    {
        case 0: 
        case 1: 
            nHiddenTensorsPerLayer = 1;
            workspaceScale = 1;
            break;
        case 2:
            nHiddenTensorsPerLayer = 4;
            workspaceScale = 6;
            break;
        case 3:
            nHiddenTensorsPerLayer = 3;
            workspaceScale = 4;            
            break;
    }
    
    if(bidir == miopenRNNbidirection)
    {
        hsize *= 2;
    }
    inputBatchLenSum = 0; //init 
    
}



size_t RNNDescriptor::GetWorkspaceSize(Handle& handle,
                                const int sLen,
                                TensorDescriptor* xDesc)  
{
    // NOTE dlowell: this calculation WILL change during development.
    // currently this is calculated the same as Workspace size
    // x = maxSequenceLen * batchSize * vector_size * numLayers * bytesForDataType * numberOfHiddenMatricesPerCell + Extra 
    // GetElemSize will get vector len * batch_size
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    
    if(!inputBatchLenSum){
        for(int i = 0 ; i < sLen ; i++)
        {
            inputBatchLenSum += xDesc[i].GetLengths()[0];
        }
    }
    auto x = workspaceScale*sLen*inputBatchLenSum*nLayers*sizeof(xDesc[0].GetType())*nHiddenTensorsPerLayer;
    return size_t(x);
}



size_t RNNDescriptor::GetReserveSize(Handle& handle,
                                const int sLen,
                                TensorDescriptor* xDesc) 
{
    // NOTE dlowell: this calculation WILL change during development.
    // x = maxSequenceLen * batchSize * vector_size * numLayers * bytesForDataType * numberOfHiddenMatricesPerCell + Extra 
    // GetElemSize will get vector len * batch_size
    if(xDesc[0].GetType() != dataType)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Data type mismatch between descriptors");
    }
    if(!inputBatchLenSum){
        for(int i = 0 ; i < sLen ; i++)
        {
            inputBatchLenSum += xDesc[i].GetLengths()[0];
        }
    }
    auto x = workspaceScale*sLen*inputBatchLenSum*nLayers*sizeof(xDesc[0].GetType())*nHiddenTensorsPerLayer;
    return size_t(x);
}


size_t RNNDescriptor::GetParamsSize(Handle& handle,
                                    const TensorDescriptor& xDesc,
                                    miopenDataType_t dtype) const
{
    // DLOWELL : The factor of 4 counts the input matrix, hidden matrix, input bias, hidden bias 
    // to each of the activated section of the RNN cell.
    // h_t = sigma(Wx_t + Rh_t-1 + bw + br)
    // for one layer: wDesc <-- (v_hidden x v_input) + (v_hidden x v_hidden) + 2*(1 x v_hidden)
    assert(xDesc.GetLengths().size() > 1);
    auto inputVecSize =  xDesc.GetLengths()[1];
    auto x = nLayers * nHiddenTensorsPerLayer * ((hsize * inputVecSize) + (hsize*hsize) + 2*hsize);
    return size_t(x);
}

void RNNDescriptor::GetLayerParam(Handle& handle,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& wDesc,
                                  ConstData_t w,
                                  const int layerID,
                                  const TensorDescriptor& paramDesc,
                                  size_t paramOffset) const
{
    
    /*If mode in rnnDesc was set to CUDNN_RNN_RELU or
CUDNN_RNN_TANH a value of 0 references the matrix
multiplication applied to the input from the previous layer, a
value of 1 references the matrix multiplication applied to the
recurrent input.*/
    // 0 --> Wx_t
    // 1 --> Rh_t-1
    
    
    
    
    
    
}

void RNNDescriptor::GetLayerBias(Handle& handle,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& wDesc,
                                 ConstData_t w,
                                 const int layerID,
                                 const TensorDescriptor& biasDesc,
                                 size_t biasOffset) const
{
    
    
    
}







void RNNDescriptor::ForwardRNNInferCell(Handle& handle,
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
                                        size_t workSpaceSize) const
{
}

void RNNDescriptor::ForwardRNNTrainCell(Handle& handle,
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

void RNNDescriptor::BackwardRNNDataCell(Handle& handle,
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

void RNNDescriptor::BackwardRNNWeightsCell(Handle& handle,
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
