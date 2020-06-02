#include "common_header.hpp"
#include "reduction_common.hpp"
#include "gridwise_generic_reduction.hpp"

using namespace ck;

using srcDataType =
    typename get_type_from_type_enum<static_cast<ckDataType_t>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType =
    typename get_type_from_type_enum<static_cast<ckDataType_t>(CK_PARAM_DST_DATATYPE)>::type;
using compType =
    typename get_type_from_type_enum<static_cast<ckDataType_t>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t blockSize = CK_PARAM_BLOCKSIZE; // tunable
constexpr index_t blkGroupSize =
    CK_PARAM_BLKGROUPSIZE; // determined by the problem and the selected BlockSize

using srcLengths = Sequence<CK_PARAM_SRC_DESC_LENGTHS>;
using srcStrides = Sequence<CK_PARAM_SRC_DESC_STRIDES>;
using dstLengths = Sequence<CK_PARAM_DST_DESC_LENGTHS>;
using dstStrides = Sequence<CK_PARAM_DST_DESC_STRIDES>;

using toReduceDims  = Sequence<CK_PARAM_TOREDUCE_DIMS>;
using invariantDims = Sequence<CK_PARAM_INVARIANT_DIMS>;

constexpr ckReduceTensorOp_t op          = static_cast<ckReduceTensorOp_t>(CK_PARAM_REDUCE_OP);
constexpr ckReductionMethod_t reduceImpl = static_cast<ckReductionMethod_t>(CK_PARAM_REDUCE_IMPL);
constexpr ckNanPropagation_t nanPropaOpt = static_cast<ckNanPropagation_t>(CK_PARAM_NAN_PROPAGATE);
constexpr ckReduceTensorIndices_t reduceIndicesOpt =
    static_cast<ckReduceTensorIndices_t>(CK_PARAM_REDUCE_INDICES);

constexpr index_t GredThreadBufferLength       = CK_PARAM_THREAD_BUFFER_LENGTH;        // tunable
constexpr index_t GredAccessesPerThreadInBlock = CK_PARAM_ACCESSES_PER_THREAD_INBLOCK; // tunable
constexpr index_t GredAccessesPerThreadInWarp  = CK_PARAM_ACCESSES_PER_THREAD_INWARP;  // tunable

extern "C" __global__ void gridwise_generic_reduce_1(srcDataType alpha,
                                                     const void* p_src_global,
                                                     srcDataType beta,
                                                     void* p_dst_global,
                                                     void* ws_buf1_global,
                                                     void* ws_buf2_global,
                                                     void* indices_global)
{
    static_assert(srcLengths::Size() > 0 && srcLengths::Size() == srcStrides::Size(),
                  "The source desc specification is invalid!");
    static_assert(dstLengths::Size() > 0 && dstLengths::Size() == dstStrides::Size(),
                  "The destination desc specification is invalid!");
    static_assert(dstLengths::Size() <= srcLengths::Size(),
                  "The destination lengths should be less than source lengths!");

    constexpr auto srcDesc = make_native_tensor_descriptor(srcLengths{}, srcStrides{});
    constexpr auto dstDesc = make_native_tensor_descriptor(dstLengths{}, dstStrides{});

    constexpr auto gridwise_reduce = Gridwise_generic_reduction<blkGroupSize,
                                                                blockSize,
                                                                srcDataType,
                                                                dstDataType,
                                                                compType,
                                                                decltype(srcDesc),
                                                                toReduceDims,
                                                                invariantDims,
                                                                decltype(dstDesc),
                                                                static_cast<int>(op),
                                                                static_cast<int>(reduceImpl),
                                                                static_cast<int>(nanPropaOpt),
                                                                static_cast<int>(reduceIndicesOpt),
                                                                GredThreadBufferLength,
                                                                GredAccessesPerThreadInBlock,
                                                                GredAccessesPerThreadInWarp>{};

    gridwise_reduce.Run(
        alpha,
        const_cast<const srcDataType* const __restrict__>(
            static_cast<const srcDataType*>(p_src_global)),
        beta,
        const_cast<dstDataType* const __restrict__>(static_cast<dstDataType*>(p_dst_global)),
        const_cast<void* const __restrict__>(ws_buf1_global),
        const_cast<void* const __restrict__>(ws_buf2_global),
        const_cast<void* const __restrict__>(indices_global));
};

extern "C" __global__ void gridwise_generic_reduce_2(srcDataType alpha,
                                                     const void* p_src_global,
                                                     srcDataType beta,
                                                     void* p_dst_global,
                                                     void* ws_buf1_global,
                                                     void* ws_buf2_global,
                                                     void* indices_global)
{
    static_assert(srcLengths::Size() > 0 && srcLengths::Size() == srcStrides::Size(),
                  "The source desc specification is invalid!");
    static_assert(dstLengths::Size() > 0 && dstLengths::Size() == dstStrides::Size(),
                  "The destination desc specification is invalid!");
    static_assert(dstLengths::Size() <= srcLengths::Size(),
                  "The destination lengths should be less than source lengths!");

    constexpr auto srcDesc = make_native_tensor_descriptor(srcLengths{}, srcStrides{});
    constexpr auto dstDesc = make_native_tensor_descriptor(dstLengths{}, dstStrides{});

    constexpr auto gridwise_reduce = Gridwise_generic_reduction<blkGroupSize,
                                                                blockSize,
                                                                srcDataType,
                                                                dstDataType,
                                                                compType,
                                                                decltype(srcDesc),
                                                                toReduceDims,
                                                                invariantDims,
                                                                decltype(dstDesc),
                                                                static_cast<int>(op),
                                                                static_cast<int>(reduceImpl),
                                                                static_cast<int>(nanPropaOpt),
                                                                static_cast<int>(reduceIndicesOpt),
                                                                GredThreadBufferLength,
                                                                GredAccessesPerThreadInBlock,
                                                                GredAccessesPerThreadInWarp>{};

    gridwise_reduce.Run_2(
        alpha,
        const_cast<const srcDataType* const __restrict__>(
            static_cast<const srcDataType*>(p_src_global)),
        beta,
        const_cast<dstDataType* const __restrict__>(static_cast<dstDataType*>(p_dst_global)),
        const_cast<void* const __restrict__>(ws_buf1_global),
        const_cast<void* const __restrict__>(ws_buf2_global),
        const_cast<void* const __restrict__>(indices_global));
};
