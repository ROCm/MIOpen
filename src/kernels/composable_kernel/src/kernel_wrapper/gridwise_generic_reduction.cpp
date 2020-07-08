#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"
#include "gridwise_generic_reduction.hpp"

using namespace ck;

template <char tid>
struct get_type_from_type_id
{
    using type = float;
};

template <>
struct get_type_from_type_id<'H'>
{
    using type = half;
};

template <>
struct get_type_from_type_id<'F'>
{
    using type = float;
};

template <>
struct get_type_from_type_id<'D'>
{
    using type = double;
};

template <int persistentID>
struct get_reduce_op // any other ID
{
    static constexpr ckReduceTensorOp_t op = CK_REDUCE_TENSOR_ADD;
};

template <>
struct get_reduce_op<656868> // 'A' * 10000 + 'D' * 100 + 'D'
{
    static constexpr ckReduceTensorOp_t op = CK_REDUCE_TENSOR_ADD;
};

template <>
struct get_reduce_op<778576> // 'M' * 10000 + 'U' * 100 + 'L'
{
    static constexpr ckReduceTensorOp_t op = CK_REDUCE_TENSOR_MUL;
};

template <>
struct get_reduce_op<777378> // 'M' * 10000 + 'I' * 100 + 'N'
{
    static constexpr ckReduceTensorOp_t op = CK_REDUCE_TENSOR_MIN;
};

template <>
struct get_reduce_op<776588> // 'M' * 10000 + 'A' * 100 + 'X'
{
    static constexpr ckReduceTensorOp_t op = CK_REDUCE_TENSOR_MAX;
};

using srcDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_DST_DATATYPE)>::type;
using compType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr int blockSize = CK_PARAM_BLOCKSIZE; // tunable
constexpr int blkGroupSize =
    CK_PARAM_BLKGROUPSIZE; // determined by the problem and the selected BlockSize

using srcLengths = Sequence<CK_PARAM_SRC_DESC_LENGTHS>;
using srcStrides = Sequence<CK_PARAM_SRC_DESC_STRIDES>;
using dstLengths = Sequence<CK_PARAM_DST_DESC_LENGTHS>;
using dstStrides = Sequence<CK_PARAM_DST_DESC_STRIDES>;

using toReduceDims  = Sequence<CK_PARAM_TOREDUCE_DIMS>;
using invariantDims = Sequence<CK_PARAM_INVARIANT_DIMS>;

constexpr ckReduceTensorOp_t op          = get_reduce_op<CK_PARAM_REDUCE_OP>::op;
constexpr ckReductionMethod_t reduceImpl = static_cast<ckReductionMethod_t>(CK_PARAM_REDUCE_IMPL);
constexpr ckNanPropagation_t nanPropaOpt =
    CK_PARAM_NAN_PROPAGATE == 0 ? CK_NOT_PROPAGATE_NAN : CK_PROPAGATE_NAN;
constexpr ckReduceTensorIndices_t reduceIndicesOpt =
    CK_PARAM_REDUCE_INDICES == 0 ? CK_REDUCE_TENSOR_NO_INDICES : CK_REDUCE_TENSOR_FLATTENED_INDICES;

constexpr int GredThreadBufferLength       = CK_PARAM_THREAD_BUFFER_LENGTH;        // tunable
constexpr int GredAccessesPerThreadInBlock = CK_PARAM_ACCESSES_PER_THREAD_INBLOCK; // tunable
constexpr int GredAccessesPerThreadInWarp  = CK_PARAM_ACCESSES_PER_THREAD_INWARP;  // tunable

extern "C" __global__ void gridwise_generic_reduce_1(float alpha,
                                                     const void* p_src_global,
                                                     float beta,
                                                     void* p_dst_global,
                                                     void* ws_buf1_global,
                                                     long ws_buf2_bytes_offset,
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

    gridwise_reduce.Run(alpha,
                        const_cast<const void* const __restrict__>(p_src_global),
                        beta,
                        const_cast<void* const __restrict__>(p_dst_global),
                        const_cast<void* const __restrict__>(ws_buf1_global),
                        ws_buf2_bytes_offset,
                        const_cast<void* const __restrict__>(indices_global));
};

extern "C" __global__ void gridwise_generic_reduce_2(float alpha,
                                                     const void* p_src_global,
                                                     float beta,
                                                     void* p_dst_global,
                                                     void* ws_buf1_global,
                                                     long ws_buf2_bytes_offset,
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

    gridwise_reduce.Run_2(alpha,
                          const_cast<const void* const __restrict__>(p_src_global),
                          beta,
                          const_cast<void* const __restrict__>(p_dst_global),
                          const_cast<void* const __restrict__>(ws_buf1_global),
                          ws_buf2_bytes_offset,
                          const_cast<void* const __restrict__>(indices_global));
};
