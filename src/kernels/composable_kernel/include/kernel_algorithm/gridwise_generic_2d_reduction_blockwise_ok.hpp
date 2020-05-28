#ifndef _CK_GRIDWISE_GENERIC_2D_REDUCTION_BLOCKWISE_HPP_
#define _CK_GRIDWISE_GENERIC_2D_REDUCTION_BLOCKWISE_HPP_

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "blockwise_generic_reduction.hpp"
#include "ConstantMatrixDescriptor.hpp"

namespace ck {

template <index_t BlockSize,
          typename DataType,
          typename opDataType,
          typename src2dDesc,
          typename dst1dDesc,
          typename opReduce,
          ReductionMethod reduceImpl,
          index_t GredPerThreadBufferLength,
	  index_t GredAccessesPerThreadInBlock>
struct Gridwise_generic_reduction_xy_to_x;

template <index_t BlockSize,
          typename DataType,
          typename opDataType,
          typename src2dDesc,
          typename dst1dDesc,
          typename opReduce,
          index_t GredAccessesPerThreadInBlock>
struct Gridwise_generic_reduction_xy_to_x<BlockSize, DataType, opDataType, src2dDesc, dst1dDesc, opReduce, Reduce_BlockWise, 0, GredAccessesPerThreadInBlock>
{
    __device__ void Run(opDataType alpha, const DataType* const __restrict__ p_src_global, opDataType beta, DataType* const __restrict__ p_dst_global) const
    {
          // LDS
	  __shared__ DataType p_in_block_buffer[BlockSize*GredAccessesPerThreadInBlock];

          // VGPR, only useful for thread 0
          opDataType accuValue;
          opDataType oAccuValue = opReduce::zeroVal;

          const index_t thread_local_id = get_thread_local_1d_id();
          const index_t block_global_1d_id = get_block_1d_id();

          constexpr auto block_buff_mtx_desc = make_ConstantMatrixDescriptor_packed(Number<GredAccessesPerThreadInBlock>{}, Number<BlockSize>{}); 

          using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_mtx_desc), true, opReduce>;  

          const index_t toReduceBlocks = (src2dDesc::GetLengths()[1] + BlockSize-1) / BlockSize; 

          set_block_buffer_value<BlockSize, DataType>(p_in_block_buffer, BlockSize*GredAccessesPerThreadInBlock, static_cast<DataType>(opReduce::zeroVal));  

          for (index_t reducedBlocks=0; reducedBlocks < toReduceBlocks; reducedBlocks += GredAccessesPerThreadInBlock) {

               for (index_t ind0=0; ind0 < GredAccessesPerThreadInBlock; ind0++) {	
                    index_t ind1 = thread_local_id; 	

                    index_t dimOffset = (reducedBlocks+ind0)*BlockSize + ind1; 
                    if ( dimOffset < src2dDesc::GetLengths()[1] ) {
                         index_t srcOffset = src2dDesc::CalculateOffset( {block_global_1d_id, dimOffset} ); 
	                 index_t dstOffset = block_buff_mtx_desc.CalculateOffset(ind0, ind1); 

                         p_in_block_buffer[dstOffset] = p_src_global[srcOffset]; 
	            }; 	 

                    __syncthreads(); 
               }; 

               index_t oneReduceBlocks = (reducedBlocks < toReduceBlocks-GredAccessesPerThreadInBlock)? GredAccessesPerThreadInBlock : toReduceBlocks-reducedBlocks;
	       blockwise_reduce::Run(p_in_block_buffer, oneReduceBlocks, accuValue);

               if ( thread_local_id == 0 ) 
		    oAccuValue = opReduce{}(oAccuValue, accuValue); 

               set_block_buffer_value<BlockSize, DataType>(p_in_block_buffer, BlockSize*GredAccessesPerThreadInBlock, static_cast<DataType>(opReduce::zeroVal));  
          }; 	

          using ReducedDataLengths = Sequence<1>;
          constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

          // The first thread in the block stores the reduced result to the global location representing the block
          if ( thread_local_id == 0 ) {
               if ( alpha != static_cast<opDataType>( 1.0 ) )
                    oAccuValue *= alpha;

               if ( beta != static_cast<DataType>(0.0) ) {
                    auto threadwise_dst_load = ThreadwiseGenericTensorSliceCopy_v4r2<dst1dDesc,
                                                                               decltype(ReducedDataDesc),
                                                                               ReducedDataLengths,
                                                                               Sequence<0>,
                                                                               0,
                                                                               1,
                                                                               1,
                                                                               AddressSpace::Global,
                                                                               AddressSpace::Vgpr,
                                                                               InMemoryDataOperation::Set>( {block_global_1d_id}, {0} );
                    DataType priorDstValue;

                    threadwise_dst_load.Run(p_dst_global, &priorDstValue);

                    oAccuValue += static_cast<opDataType>(priorDstValue) * beta;
               };


               auto threadwise_dst_store = ThreadwiseGenericTensorSliceCopy_v4r2<decltype(ReducedDataDesc),
                                                                           dst1dDesc,
                                                                           ReducedDataLengths,
                                                                           Sequence<0>,
                                                                           0,
                                                                           1,
                                                                           1,
                                                                           AddressSpace::Vgpr,
                                                                           AddressSpace::Global,
                                                                           InMemoryDataOperation::Set>( {0}, {block_global_1d_id} );
               threadwise_dst_store.Run(&oAccuValue, p_dst_global);
	  }; 
    }
};

} // namespace ck
#endif

