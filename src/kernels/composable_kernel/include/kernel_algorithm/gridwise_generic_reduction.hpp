/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef _CK_GRIDWISE_GENERIC_REDUCTION_HPP_
#define _CK_GRIDWISE_GENERIC_REDUCTION_HPP_

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_kernel_simple_configurator.hpp"

#include "tuple_ext.hpp"

#include "gridwise_generic_2d_reduction_direct_threadwise.hpp"
#include "gridwise_generic_2d_reduction_direct_warpwise.hpp"
#include "gridwise_generic_2d_reduction_blockwise.hpp"
#include "gridwise_generic_2d_reduction_multiblock.hpp"

namespace ck {

template <index_t BlkGroupSize,
	  index_t BlockSize,
          typename srcDataType,         // the type with which the data of the source tensor are stored 
          typename dstDataType,         // the type with which the data of the destintion tensor are stored
          typename compType,            // the type used by the reduce binary operator 
          typename srcDesc,             // the descriptor representing the source tensor to be reduced
          typename toReduceDims,        // the Sequence<...> consists of the indexes of toReduce dimensions in the source tensor descriptor
	  typename invariantDims,       // the Sequence<...> consists of the indexes of invariant dimensions in the source tensor descriptor (can be empty)  
          typename dstDesc,             // the descriptor representing the destination tensor where the reduced tensor data are saved/added 
          int op_I,                     // the enumerate value representing the operation used in Reduction 
          int reduceImpl_I,             // the enumerate value representing the ReductionMethod
          int nanPropaOpt_I,            // the enumerate value representing the NanPropagation Option 
          int reduceIndicesOpt_I,       // the enumerate value representing the Reduce Indices Option
	  index_t GredThreadBufferLength,
	  index_t GredAccessesPerThreadInBlock,
	  index_t GredAccessesPerThreadInWarp>
struct Gridwise_generic_reduction
{
     static constexpr auto reduceImpl = static_cast<ckReductionMethod_t>(reduceImpl_I); 
     static constexpr bool is_method_multiblock = (reduceImpl == CK_Reduce_MultiBlock)? true: false; 
     static constexpr auto op = static_cast<ckReduceTensorOp_t>(op_I); 
     static constexpr auto nanPropaOpt = static_cast<ckNanPropagation_t>(nanPropaOpt_I); 
     static constexpr auto reduceIndicesOpt = static_cast<ckReduceTensorIndices_t>(reduceIndicesOpt_I); 

     template <ckReductionMethod_t impl, int callId>
     struct Gridwise_generic_2d_reduction_wrapper; 

     // wrapper for switching to the Reduce_DirectThreadWise method
     template <int callId>
     struct Gridwise_generic_2d_reduction_wrapper<CK_Reduce_DirectThreadWise,callId>
     {
         template <typename src2dDesc, typename dst1dDesc> 
         __device__ static void Run(src2dDesc, dst1dDesc, srcDataType alpha, const srcDataType* const __restrict__ p_src_global, srcDataType beta, dstDataType* const __restrict__ p_dst_global, 
			                srcDataType* const __restrict__ ws_buf1_global, int* const __restrict__ ws_buf2_global, int* const __restrict__ indices_global) 
         {
              using gridwise_reduce = Gridwise_generic_reduction_xy_to_x_direct_threadwise<BlockSize, srcDataType, dstDataType, src2dDesc, dst1dDesc, compType, op, nanPropaOpt, reduceIndicesOpt, 
                                                                  callId, GredThreadBufferLength>;             // the callId indicates the first or second-time reduction
	      gridwise_reduce{}.Run(alpha, p_src_global, beta, p_dst_global, ws_buf2_global, indices_global);  // ws_buf2_global will be read at the second-time
         }; 
     }; 

     // wrapper for switching to the Reduce_DirectWarpdWise method
     template <int callId>
     struct Gridwise_generic_2d_reduction_wrapper<CK_Reduce_DirectWarpWise,callId>
     {
         template <typename src2dDesc, typename dst1dDesc> 
         __device__ static void Run(src2dDesc, dst1dDesc, srcDataType alpha, const srcDataType* const __restrict__ p_src_global, srcDataType beta, dstDataType* const __restrict__ p_dst_global, 
			                srcDataType* const __restrict__ ws_buf1_global, int* const __restrict__ ws_buf2_global, int* const __restrict__ indices_global) 
         {    
              using gridwise_reduce = Gridwise_generic_reduction_xy_to_x_direct_warpwise<BlockSize, srcDataType, dstDataType, src2dDesc, dst1dDesc, compType, op, nanPropaOpt, reduceIndicesOpt,
                                                                  callId, GredAccessesPerThreadInWarp>;        // the callId indicates the first or second-time reduction
	      gridwise_reduce{}.Run(alpha, p_src_global, beta, p_dst_global, ws_buf2_global, indices_global);  // ws_buf2_global will be read at the second-time
         };
     };

     // wrapper for switching to the Reduce_BlockWise method
     template <int callId>
     struct Gridwise_generic_2d_reduction_wrapper<CK_Reduce_BlockWise,callId>
     {
         template <typename src2dDesc, typename dst1dDesc> 
         __device__ static void Run(src2dDesc, dst1dDesc, srcDataType alpha, const srcDataType* const __restrict__ p_src_global, srcDataType beta, dstDataType* const __restrict__ p_dst_global, 
			                srcDataType* const __restrict__ ws_buf1_global, int* const __restrict__ ws_buf2_global, int* const __restrict__ indices_global) 
         {    
              using gridwise_reduce = Gridwise_generic_reduction_xy_to_x_blockwise<BlockSize, srcDataType, dstDataType, src2dDesc, dst1dDesc, compType, op, nanPropaOpt, reduceIndicesOpt,
                                                                  callId, GredAccessesPerThreadInBlock>;       // the callId indicates the first or second-time reduction
	      gridwise_reduce{}.Run(alpha, p_src_global, beta, p_dst_global, ws_buf2_global, indices_global);  // ws_buf2_global will be read at the second-time 
         };
     };
     
     // wrapper for switching to the Reduce_MultiBlock method
     template <int callId>
     struct Gridwise_generic_2d_reduction_wrapper<CK_Reduce_MultiBlock,callId>
     {
         template <typename src2dDesc, typename dst1dDesc> 
         __device__ static void Run(src2dDesc, dst1dDesc, srcDataType alpha, const srcDataType* const __restrict__ p_src_global, srcDataType beta, dstDataType* const __restrict__ p_dst_global, 
			                srcDataType* const __restrict__ ws_buf1_global, int* const __restrict__ ws_buf2_global, int* const __restrict__ indices_global) 
         {    
              using gridwise_reduce = Gridwise_generic_reduction_xy_to_x_multiblock<BlockSize, srcDataType, dstDataType, src2dDesc, dst1dDesc, compType, op, nanPropaOpt, reduceIndicesOpt,
                                                                  BlkGroupSize, GredAccessesPerThreadInBlock>;   // MultiBlock case is not used by second-time reduction
	      gridwise_reduce{}.Run(alpha, p_src_global, beta, ws_buf1_global, ws_buf2_global);  // ws_buf1_global instead of p_dst_global, ws_buf2_global instead of indices_global 
         };
     };

     __device__ void Run(srcDataType alpha, const srcDataType* const __restrict__ p_src_global, srcDataType beta, dstDataType* const __restrict__ p_dst_global,
		                            void* const __restrict__ ws_buf1_global, void* const __restrict__ ws_buf2_global, void* const __restrict__ indices_global) const
     {
         using srcLengths = decltype(srcDesc::GetLengths());
         using dstLengths = decltype(dstDesc::GetLengths()); 

         using specDims = typename sequence_merge<invariantDims, toReduceDims>::type; 
         static_assert(is_valid_sequence_map<specDims>::value && specDims::Size() == srcLengths::Size(),  
			 "Wrong invariant and/or toReduce dimensions!");
      

         static_assert( toReduceDims::Size() >= 1,  "Wrong specification of source mode, We should at least to have one dimension to be reduced !!"); 

         // The number of invariant dimensions can be zero if all dimension are to be reduced 
         static_assert( invariantDims::Size() > 0 || ( dstLengths::Size() == 1 && dstLengths{}[0] == 1 ), 
         	           "If all source dimensions are reduced, the dest should have only one dimension !!"); 

         constexpr bool reduceAllDims = (invariantDims::Size() == 0) ? true : false;

         static_if<!reduceAllDims>{}( [&](auto) {   // not all dimensions are to be reduced
             using toReduceDimLengths = decltype( srcLengths::Extract(toReduceDims{}) );
             using invariantDimLengths = decltype( srcLengths::Extract(invariantDims{}) );

             // for re-ordering the tensor dimensions
             using lowDimSeq = typename sequence_merge<invariantDims, toReduceDims>::type;
             using highDimSeq = typename arithmetic_sequence_gen<0, srcLengths::Size(), 1>::type;

             // construct the reordered tensor descriptor according to the srcMode and dstMode mapping
             constexpr auto reordered_srcDesc = transform_tensor_descriptor(srcDesc{},
                                                              make_passthrough_tuple( srcLengths::Extract(lowDimSeq{}) ),
                                                              make_dimensions_tuple( lowDimSeq{} ),
                                                              make_dimensions_tuple( highDimSeq{} ) );
             constexpr auto two_dim_srcDesc = transform_tensor_descriptor(reordered_srcDesc,
                                                              make_2d_merge_transform_tuple( invariantDimLengths{}, toReduceDimLengths{} ),
                                                              make_tuple( typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{},
                                                                          typename arithmetic_sequence_gen<dstLengths::Size(), srcLengths::Size(), 1>::type{} ),
                                                              make_tuple( Sequence<0>{}, Sequence<1>{} ) );
								
             constexpr auto one_dim_dstDesc = transform_tensor_descriptor(dstDesc{}, 
                                                              make_1d_merge_transform_tuple( dstLengths{} ),
							      make_tuple( typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{} ), 
						              make_tuple( Sequence<0>{} ) );  

             using gridwise_2d_reduce = Gridwise_generic_2d_reduction_wrapper<reduceImpl,0>;  

	     gridwise_2d_reduce{}.Run(two_dim_srcDesc, one_dim_dstDesc, alpha, p_src_global, beta, p_dst_global, 
			        static_cast<srcDataType* const __restrict__>(ws_buf1_global), static_cast<int* const __restrict__>(ws_buf2_global), 
				static_cast<int * const __restrict__>(indices_global) );   
         }).Else([&](auto) {                       // All dimensions are to be reduced
             constexpr auto one_dim_srcDesc = transform_tensor_descriptor(srcDesc{},
                                                              make_1d_merge_transform_tuple( srcLengths{} ),
                                                              make_tuple( typename arithmetic_sequence_gen<0, srcLengths::Size(), 1>::type{} ),
                                                              make_tuple( Sequence<0>{}) );

             constexpr auto dim_length = one_dim_srcDesc.GetLengths()[0]; 

             constexpr auto two_dim_srcDesc =  transform_tensor_descriptor(one_dim_srcDesc,  
                	                                      make_tuple(UnMerge<Sequence<1, dim_length>>{}),
                                                              make_tuple(Sequence<0>{}),
                                                              make_tuple(Sequence<0, 1>{}) ); 

             constexpr auto one_dim_dstDesc = transform_tensor_descriptor(dstDesc{},
                                                              make_1d_merge_transform_tuple( dstLengths{} ),
                                                              make_tuple( typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{} ),
                                                              make_tuple( Sequence<0>{} ) );

             using gridwise_2d_reduce = Gridwise_generic_2d_reduction_wrapper<reduceImpl,0>;  

	     gridwise_2d_reduce{}.Run(two_dim_srcDesc, one_dim_dstDesc, alpha, p_src_global, beta, p_dst_global, 
			        static_cast<srcDataType* const __restrict__>(ws_buf1_global), static_cast<int* const __restrict__>(ws_buf2_global), 
				static_cast<int * const __restrict__>(indices_global) );   
         }); 
     };

     __device__ void Run_2(srcDataType alpha, const srcDataType* const __restrict__ p_src_global, srcDataType beta, dstDataType* const __restrict__ p_dst_global,
		                              void* const __restrict__ ws_buf1_global, void* const __restrict__ ws_buf2_global, void* const __restrict__ indices_global) const
     {
         using dstLengths = decltype(dstDesc::GetLengths());

         constexpr auto one_dim_dstDesc = transform_tensor_descriptor(dstDesc{},
                                                          make_1d_merge_transform_tuple( dstLengths{} ),
                                                          make_tuple( typename arithmetic_sequence_gen<0, dstLengths::Size(), 1>::type{} ),
                                                          make_tuple( Sequence<0>{} ) );
         constexpr index_t invariantLength = one_dim_dstDesc.GetLengths()[0]; 
	 constexpr index_t toReduceLength = BlkGroupSize; 

         constexpr auto workspace_2d_desc = make_native_tensor_descriptor_packed( Sequence<invariantLength, toReduceLength>{} );

         static_if<is_method_multiblock>{}([&] (auto)  {
              constexpr ckReductionMethod_t reduceImpl2 = reduce_kernel_simple_configurator<BlockSize, warpSize>::getReductionMethod(Number<invariantLength>{},
			                                                                                          Number<toReduceLength>{}); 

              using gridwise_2d_reduce = Gridwise_generic_2d_reduction_wrapper<reduceImpl2,1>;

	      gridwise_2d_reduce{}.Run(workspace_2d_desc, one_dim_dstDesc, alpha, const_cast<const srcDataType* const __restrict__>(static_cast<srcDataType*>(ws_buf1_global)), 
			                                              beta, p_dst_global, const_cast<dstDataType* const __restrict__>(static_cast<dstDataType*>(nullptr)),
					                              static_cast<int* const __restrict__>(ws_buf2_global), static_cast<int* const __restrict__>(indices_global));
         }).Else([&] (auto) {
         }); 
     };
};

} // namespace ck
#endif
