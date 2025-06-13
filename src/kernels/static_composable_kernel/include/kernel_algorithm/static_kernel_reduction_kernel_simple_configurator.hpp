#ifndef REDUCTION_KERNEL_SIMPLE_CONFIGURATOR_HPP_
#define REDUCTION_KERNEL_SIMPLE_CONFIGURATOR_HPP_ 1

#include "static_kernel_number.hpp"
#include "static_kernel_reduction_common.hpp"

#include "miopen_warp_size.hpp"

namespace ck {

// The simple configurator does not consider the "Reduce_MultiBlock" method, since it is usually
// called to do the second reduction after the first calling of a "Reduce_MultiBlock" reduction.
template <index_t BlockSize>
struct ReduceKernelSimpleConfigurator
{
    static constexpr index_t numWarpsPerBlock = BlockSize / MIOPEN_WARP_SIZE;

    template <index_t invariantLength, index_t toReduceLength>
    __device__ static constrexpr index_t GetGridSize(Number<invariantLength>,
                                                     Number<toReduceLength>)
    {
        if(toReduceLength <= MIOPEN_WARP_SIZE / 4) // let one thread to do each reduction
            return ((invariantLength + BlockSize - 1) / BlockSize);
        else if(toReduceLength <= BlockSize) // let one warp to do each reduction
            return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
        else
            return (invariantLength); // let one block to do each reduction
    };

    template <index_t invariantLength, index_t toReduceLength>
    __device__ static constexpr ReductionMethod_t GetReductionMethod(Number<invariantLength>,
                                                                     Number<toReduceLength>)
    {
        if(toReduceLength <= MIOPEN_WARP_SIZE / 4) // let one thread to do each reduction
            return (ReductionMethod_t::DirectThreadWise);
        else if(toReduceLength <= BlockSize) // let one warp to do each reduction
            return (ReductionMethod_t::DirectWarpWise);
        else
            return (ReductionMethod_t::BlockWise);
    };
};

}; // namespace ck

#endif
