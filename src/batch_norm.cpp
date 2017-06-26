#include <miopen/batch_norm.hpp>
#include <miopen/errors.hpp>


namespace miopen {

    void DeriveBNTensorDescriptor(TensorDescriptor& derivedBnDesc, const TensorDescriptor& xDesc, miopenBatchNormMode_t bn_mode){
        
        std::vector<int> lengths = xDesc.GetLengths();
        std::vector<int> newlens(lengths.size());
        newlens[1] = lengths[1];
        if(bn_mode==miopenBNSpatial){
            newlens[0] = newlens[2] = newlens[3] = 1;//TODO: support 5D
        }else{
            newlens[0] = 1;
            newlens[2] = lengths[2];
            newlens[3] = lengths[3];;//TODO: support 5D          
        }
        derivedBnDesc = TensorDescriptor(xDesc.GetType(),newlens.data(), xDesc.GetSize());
    }
    
}  // namespace miopen
