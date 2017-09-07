#include <miopen/rnn.hpp>
#include <miopen/errors.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)

namespace miopen {

RNNDescriptor::RNNDescriptor(){
    seqLength = 1;
    nlayers   = 1;
    rnnMode   = miopenRNNRELU;
    dirMode   = miopenRNNunidirection;
    algoMode  = miopenRNNdefault;
    inputMode = miopenRNNlinear;    
}
    
RNNDescriptor::RNNDescriptor(int seqLength = 1, int layer = 1, miopenRNNDirectionMode_t bidir = miopenRNNunidirection)
: seqLength(seqLength), nlayers(layer)
{
    rnnMode   = miopenRNNRELU;
    dirMode   = miopenRNNunidirection;
    algoMode  = miopenRNNdefault;
    inputMode = miopenRNNlinear; 
}

RNNDescriptor::RNNDescriptor(miopenRNNMode_t p_mode, int seqLength = 1, int layer = 1, miopenRNNDirectionMode_t bidir = miopenRNNunidirection)
: rnnMode(p_mode), seqLength(seqLength), nlayers(layer), dirMode(bidir)
{
    algoMode  = miopenRNNdefault;
    inputMode = miopenRNNlinear; 	
}


} // namespace miopen
