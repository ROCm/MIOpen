#ifndef CK_BLOCKWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_V2_HPP
#define CK_BLOCKWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_V2_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "cluster_descriptor.hpp"
#include "threadwise_dynamic_tensor_slice_transfer_v2.hpp"

namespace ck {

// this version does following things to avoid scratch memory issue
// 1. Use StaticallyIndexedArray instead of C array for thread buffer
// 2. ThreadwiseDynamicTensorSliceTransfer_v3 does not keep reference to tensor descriptor
// 3. ThreadwiseDynamicTensorSliceTransfer_v3::Run() does not construct new tensor coordinate
template <index_t BlockSize,
          InMemoryDataOperation DstInMemOp,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          typename SrcVectorTensorLengths,
          typename DstVectorTensorLengths,
          typename SrcVectorTensorContiguousDimOrder,
          typename DstVectorTensorContiguousDimOrder,
          bool ThreadTransferSrcResetCoordinateAfterRun,
          bool ThreadTransferDstResetCoordinateAfterRun>
struct BlockwiseDynamicTensorSliceTransfer_v4r1
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseDynamicTensorSliceTransfer_v4r1(
        const SrcDesc& src_desc,
        const Index& src_block_slice_origin,
        const DstDesc& dst_desc,
        const Index& dst_block_slice_origin)
        : threadwise_transfer_(
              src_desc, make_zero_multi_index<nDim>(), dst_desc, make_zero_multi_index<nDim>())

    {
        static_assert(nDim == remove_reference_t<remove_cv_t<SrcDesc>>::GetNumOfDimension() &&
                          nDim == remove_reference_t<remove_cv_t<DstDesc>>::GetNumOfDimension() &&
                          nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= thread_cluster_desc_.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(get_thread_local_1d_id()));

            const auto thread_data_idx_begin = thread_cluster_idx * ThreadSliceLengths{};

            threadwise_transfer_.SetSrcSliceOrigin(src_desc,
                                                   src_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetDstSliceOrigin(dst_desc,
                                                   dst_block_slice_origin + thread_data_idx_begin);
        }
    }

    template <typename SrcBuffer, typename SrcIteratorHacks>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcBuffer& src_buf,
                            const SrcIteratorHacks& src_iterator_hacks)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunRead(src_desc, src_buf, src_iterator_hacks);
        }
    }

    template <typename DstBuffer>
    __device__ void RunWrite(const DstDesc& dst_desc, DstBuffer& dst_buf)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunWrite(dst_desc, dst_buf);
        }
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(src_desc, step);
        }
    }

    // SrcMoveSliceWindowIteratorHack to control index calculation move slice window
    template <typename SrcMoveSliceWindowIteratorHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& step,
                       const SrcMoveSliceWindowIteratorHack& src_move_slice_window_iterator_hack)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(
                src_desc, step, src_move_slice_window_iterator_hack);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_desc, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor_v2(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseDynamicTensorSliceTransfer_v3r1<ThreadSliceLengths,
                                                  DstInMemOp,
                                                  SrcData,
                                                  DstData,
                                                  SrcDesc,
                                                  DstDesc,
                                                  SrcDimAccessOrder,
                                                  DstDimAccessOrder,
                                                  SrcVectorTensorLengths,
                                                  DstVectorTensorLengths,
                                                  SrcVectorTensorContiguousDimOrder,
                                                  DstVectorTensorContiguousDimOrder,
                                                  ThreadTransferSrcResetCoordinateAfterRun,
                                                  ThreadTransferDstResetCoordinateAfterRun>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
#endif
