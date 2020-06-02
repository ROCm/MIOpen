#ifndef _REDUCTION_KERNEL_SIMPLE_CONFIGURATOR_HPP_
#define _REDUCTION_KERNEL_SIMPLE_CONFIGURATOR_HPP_ 1

#include "config.hpp"
#include "number.hpp"
#include "reduction_common.hpp"

namespace ck {

// The simple configurator does not consider the "Reduce_MultiBlock" method, since it is usually
// called to do the second reduction after the first calling of a "Reduce_MultiBlock" reduction.
template <int BlockSize, int warpSize>
struct reduce_kernel_simple_configurator
{
    static constexpr int numWarpsPerBlock = BlockSize / warpSize;

    template <index_t invariantLength, index_t toReduceLength>
    __device__ static constexpr int getGridSize(Number<invariantLength>, Number<toReduceLength>)
    {
        if(toReduceLength < warpSize / 4) // let one thread to do each reduction
            return ((invariantLength + BlockSize - 1) / BlockSize);
        else if(toReduceLength < BlockSize) // let one warp to do each reduction
            return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
        else
            return (invariantLength); // let one block to do each reduction
    };

    template <index_t invariantLength, index_t toReduceLength>
    __device__ static constexpr ckReductionMethod_t getReductionMethod(Number<invariantLength>,
                                                                       Number<toReduceLength>)
    {
        if(toReduceLength < warpSize / 4) // let one thread to do each reduction
            return (CK_Reduce_DirectThreadWise);
        else if(toReduceLength < BlockSize) // let one warp to do each reduction
            return (CK_Reduce_DirectWarpWise);
        else
            return (CK_Reduce_BlockWise);
    };
};

}; // namespace ck

#endif
