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
    RNNDescriptor(int seqLength = 1, int layer = 1, miopenRNNDirectionMode_t bidir = 0);
    RNNDescriptor(miopenRNNMode_t p_mode, int seqLength = 1, int layer = 1, miopenRNNDirectionMode_t bidir = 0);
    RNNDescriptor(int hsize,
                    int nlayers = 1,
                    miopenRNNInputMode_t inputMode, 
                    miopenRNNDirectionMode_t dirMode, 
                    miopenRNNMode_t rnnMode, 
                    miopenRNNAlgo_t algoMode,
                    miopenDataType_t dataType);
    
    
    

    int seqLength;
    int nlayers;
    
    miopenRNNMode_t          rnnMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNAlgo_t          algoMode;
    miopenRNNInputMode_t     inputMode;
    
    void GetWorkspaceSize() const;
    void GetReserveSize() const;  
    void GetParamsSize() const;
    void GetLayerParam() const;
    void GetLayerBias() const;
    void ForwardInferRNNCell() const;
    void ForwardTrainRNNCell() const;
    void BackwardDataRNNCell() const;
    void BackwardWeightsRNNCell() const;
	
};


std::ostream& operator<< (std::ostream& stream, const RNNDescriptor& c);

}  // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenRNNDescriptor, miopen::RNNDescriptor);

#endif // GUARD_MIOPEN_RNN_HPP_
