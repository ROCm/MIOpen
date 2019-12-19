#ifndef CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

template <index_t BlockSize,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
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

        // map threads to cluster
        constexpr auto thread_cluster_desc =
            make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

        static_assert(BlockSize == thread_cluster_desc.GetElementSize(),
                      "wrong! BlockSize not consistent with ThreadClusterLengths");

        const auto thread_cluster_id =
            thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());

        const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

        mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
        mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());

        mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
        mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
    }

    __device__ static constexpr index_t GetThreadBufferSize()
    {
        return ThreadBufferDesc::GetElementSpace();
    }

    template <typename BlockSrcData,
              typename ThreadBufferData,
              AddressSpace BlockSrcAddressSpace,
              AddressSpace ThreadBufferAddressSpace>
    __device__ void
    RunLoadThreadBuffer(const BlockSrcData* p_block_src,
                        ThreadBufferData* p_thread_buffer,
                        integral_constant<AddressSpace, BlockSrcAddressSpace>,
                        integral_constant<AddressSpace, ThreadBufferAddressSpace>) const
    {
        constexpr auto block_src_address_space =
            integral_constant<AddressSpace, BlockSrcAddressSpace>{};
        constexpr auto thread_buffer_address_space =
            integral_constant<AddressSpace, ThreadBufferAddressSpace>{};

        constexpr bool has_optimized_address_calculation =
            decltype(mThreadwiseStore)::HasWorkingOptimizedAddressCalculation();

        // TODO: threadwise copy is still being tweaked
        if(has_optimized_address_calculation)
        {
            mThreadwiseLoad.Run_optimized_src_address_calculation(
                p_block_src, p_thread_buffer, block_src_address_space, thread_buffer_address_space);
        }
        else
        {
            mThreadwiseLoad.Run(
                p_block_src, p_thread_buffer, block_src_address_space, thread_buffer_address_space);
        }
    }

    template <typename BlockSrcData, typename ThreadBufferData>
    __device__ void RunLoadThreadBuffer(const BlockSrcData* p_block_src,
                                        ThreadBufferData* p_thread_buffer) const
    {
        constexpr auto generic_address_space =
            integral_constant<AddressSpace, AddressSpace::generic>{};

        RunLoadThreadBuffer(
            p_block_src, p_thread_buffer, generic_address_space, generic_address_space);
    }

    template <typename ThreadBufferData,
              typename BlockDstData,
              AddressSpace ThreadBufferAddressSpace,
              AddressSpace BlockDstAddressSpace>
    __device__ void
    RunStoreThreadBuffer(const ThreadBufferData* p_thread_buffer,
                         BlockDstData* p_block_dst,
                         integral_constant<AddressSpace, ThreadBufferAddressSpace>,
                         integral_constant<AddressSpace, BlockDstAddressSpace>) const
    {
        constexpr auto thread_buffer_address_space =
            integral_constant<AddressSpace, ThreadBufferAddressSpace>{};
        constexpr auto block_dst_address_space =
            integral_constant<AddressSpace, BlockDstAddressSpace>{};

        constexpr bool has_optimized_address_calculation =
            decltype(mThreadwiseStore)::HasWorkingOptimizedAddressCalculation();

        // TODO: threadwise copy is still being tweaked
        if(has_optimized_address_calculation)
        {
            mThreadwiseStore.Run_optimized_dst_address_calculation(
                p_thread_buffer, p_block_dst, thread_buffer_address_space, block_dst_address_space);
        }
        else
        {
            mThreadwiseStore.Run(
                p_thread_buffer, p_block_dst, thread_buffer_address_space, block_dst_address_space);
        }
    }

    template <typename ThreadBufferData, typename BlockDstData>
    __device__ void RunStoreThreadBuffer(const ThreadBufferData* p_thread_buffer,
                                         BlockDstData* p_block_dst) const
    {
        constexpr auto generic_address_space =
            integral_constant<AddressSpace, AddressSpace::generic>{};

        RunStoreThreadBuffer(
            p_thread_buffer, p_block_dst, generic_address_space, generic_address_space);
    }

    template <typename BlockSrcData,
              typename BlockDstData,
              AddressSpace BlockSrcAddressSpace,
              AddressSpace BlockDstAddressSpace>
    __device__ void
    Run(const BlockSrcData* p_block_src,
        BlockDstData* p_block_dst,
        integral_constant<AddressSpace, BlockSrcAddressSpace> block_src_address_space,
        integral_constant<AddressSpace, BlockDstAddressSpace> block_dst_address_space) const
    {
        BlockSrcData p_thread_buffer[GetThreadBufferSize()];

        constexpr auto generic_address_space =
            integral_constant<AddressSpace, AddressSpace::generic>{};

        RunLoadThreadBuffer(
            p_block_src, p_thread_buffer, block_src_address_space, generic_address_space);

        // if there is type conversion, it's done during store
        RunStoreThreadBuffer(
            p_thread_buffer, p_block_dst, generic_address_space, block_dst_address_space);
    }

    template <typename BlockSrcData, typename BlockDstData>
    __device__ void Run(const BlockSrcData* p_block_src, BlockDstData* p_block_dst) const
    {
        constexpr auto generic_address_space =
            integral_constant<AddressSpace, AddressSpace::generic>{};

        Run(p_block_src, p_block_dst, generic_address_space, generic_address_space);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseLoad.MoveSrcSliceWindow(step_sizes, positive_direction);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseStore.MoveDstSliceWindow(step_sizes, positive_direction);
    }

    private:
    using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));

    using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<BlockSrcDesc,
                                                                 ThreadBufferDesc,
                                                                 ThreadSliceLengths,
                                                                 SrcDimAccessOrder,
                                                                 SrcVectorAccessDim,
                                                                 SrcDataPerAccess,
                                                                 1>;

    using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<ThreadBufferDesc,
                                                                  BlockDstDesc,
                                                                  ThreadSliceLengths,
                                                                  DstDimAccessOrder,
                                                                  DstVectorAccessDim,
                                                                  1,
                                                                  DstDataPerAccess>;

    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;
};

} // namespace ck

#endif
