#ifndef CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

// This blockwise copy allow vector access of src and dst.
// It allows the vector size to be different on src and dst.
// The dimension of vector access can be different for src and dst.
// The dimension access order can be different for src and dst.
// Will do valid mapping check on src data: Read 0 if src data has a invalid mapping
// Will do valid mapping check on dst data: No write if dst data has a invalid mapping
// BlockSize can be equal or larger than ThreadCluster size, which means some threads may not do
// threadwise copy
template <index_t BlockSize,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectoReadDim,
          index_t DstVectorWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace          = AddressSpace::Generic,
          AddressSpace ThreadBufferAddressSpace = AddressSpace::Generic,
          AddressSpace DstAddressSpace          = AddressSpace::Generic,
          InMemoryDataOperation DstInMemOp      = InMemoryDataOperation::Set,
          index_t SrcDataStride                 = 1,
          index_t DstDataStride                 = 1>
struct BlockwiseGenericTensorSliceCopy_v4
{
    static constexpr index_t nDim = BlockSrcDesc::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v4(const Index& src_block_slice_origin,
                                                            const Index& dst_block_slice_origin)
    {
        static_assert(nDim == BlockSrcDesc::GetNumOfDimension() &&
                          nDim == BlockDstDesc::GetNumOfDimension() &&
                          nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= mThreadClusterDesc.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            const auto thread_cluster_id =
                mThreadClusterDesc.CalculateClusterIndex(get_thread_local_1d_id());

            const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

            mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
            mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());

            mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
            mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
        }
    }

    __device__ static constexpr index_t GetThreadBufferSize()
    {
        return ThreadBufferDesc::GetElementSpace();
    }

    template <typename BlockSrcData, typename ThreadBufferData>
    __device__ void RunLoadThreadBuffer(const BlockSrcData* p_block_src,
                                        ThreadBufferData* p_thread_buffer) const
    {
        constexpr bool has_optimized_address_calculation =
            decltype(mThreadwiseStore)::HasWorkingOptimizedAddressCalculation();

        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            // TODO: threadwise copy is still being tweaked
            if(has_optimized_address_calculation)
            {
                mThreadwiseLoad.Run_optimized_src_address_calculation(p_block_src, p_thread_buffer);
            }
            else
            {
                mThreadwiseLoad.Run(p_block_src, p_thread_buffer);
            }
        }
    }

    template <typename ThreadBufferData, typename BlockDstData>
    __device__ void RunStoreThreadBuffer(const ThreadBufferData* p_thread_buffer,
                                         BlockDstData* p_block_dst) const
    {
        constexpr bool has_optimized_address_calculation =
            decltype(mThreadwiseStore)::HasWorkingOptimizedAddressCalculation();

        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            // TODO: threadwise copy is still being tweaked
            if(has_optimized_address_calculation)
            {
                mThreadwiseStore.Run_optimized_dst_address_calculation(p_thread_buffer,
                                                                       p_block_dst);
            }
            else
            {
                mThreadwiseStore.Run(p_thread_buffer, p_block_dst);
            }
        }
    }

    template <typename BlockSrcData, typename BlockDstData>
    __device__ void Run(const BlockSrcData* p_block_src, BlockDstData* p_block_dst) const
    {
        static_assert(ThreadBufferAddressSpace == AddressSpace::Vgpr,
                      "wrong! This function use vgpr as its thread "
                      "buffer. However, you have set RunLoadThreadBuffer and RunStoreThreadBuffer "
                      "to use ThreadBufferAddressSpace as their thread buffer, which is not vgpr. "
                      "Behavior may be different");

        BlockSrcData p_thread_buffer[GetThreadBufferSize()];

        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            RunLoadThreadBuffer(p_block_src, p_thread_buffer);

            // if there is type conversion, it's done during store
            RunStoreThreadBuffer(p_thread_buffer, p_block_dst);
        }
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            mThreadwiseLoad.MoveSrcSliceWindow(step_sizes, positive_direction);
        }
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            mThreadwiseStore.MoveDstSliceWindow(step_sizes, positive_direction);
        }
    }

    private:
    using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));

    using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<BlockSrcDesc,
                                                                 ThreadBufferDesc,
                                                                 ThreadSliceLengths,
                                                                 SrcDimAccessOrder,
                                                                 SrcVectoReadDim,
                                                                 SrcDataPerRead,
                                                                 1,
                                                                 SrcAddressSpace,
                                                                 ThreadBufferAddressSpace,
                                                                 InMemoryDataOperation::Set,
                                                                 SrcDataStride,
                                                                 1>;

    using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<ThreadBufferDesc,
                                                                  BlockDstDesc,
                                                                  ThreadSliceLengths,
                                                                  DstDimAccessOrder,
                                                                  DstVectorWriteDim,
                                                                  1,
                                                                  DstDataPerWrite,
                                                                  ThreadBufferAddressSpace,
                                                                  DstAddressSpace,
                                                                  DstInMemOp,
                                                                  1,
                                                                  DstDataStride>;

    static constexpr auto mThreadClusterDesc =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;
};

} // namespace ck

#endif
