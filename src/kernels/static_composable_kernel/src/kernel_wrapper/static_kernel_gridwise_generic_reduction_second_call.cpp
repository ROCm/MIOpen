#include "static_kernel_config.hpp"
#include "static_kernel_number.hpp"
#include "static_kernel_sequence.hpp"
#include "static_kernel_tensor_descriptor_helper.hpp"
#include "static_kernel_reduction_common.hpp"
#include "static_kernel_gridwise_generic_reduction.hpp"

using namespace ck;

template <char tid>
struct get_type_from_type_id
{
    using type = float;
};

template <>
struct get_type_from_type_id<'H'>
{
    using type = half_t;
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

template <index_t persistentID>
struct get_reduce_op // any other ID
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<656868> // 'A' * 10000 + 'D' * 100 + 'D'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<778576> // 'M' * 10000 + 'U' * 100 + 'L'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MUL;
};

template <>
struct get_reduce_op<777378> // 'M' * 10000 + 'I' * 100 + 'N'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MIN;
};

template <>
struct get_reduce_op<776588> // 'M' * 10000 + 'A' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MAX;
};

template <>
struct get_reduce_op<657788> // 'A' * 10000 + 'M' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AMAX;
};

template <>
struct get_reduce_op<658671> // 'A' * 10000 + 'V' * 100 + 'G'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AVG;
};

template <>
struct get_reduce_op<788201> // 'N' * 10000 + 'R' * 100 + '1'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM1;
};

template <>
struct get_reduce_op<788202> // 'N' * 10000 + 'R' * 100 + '2'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM2;
};

using srcDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_DST_DATATYPE)>::type;
using compType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t gridSize =
    CK_PARAM_GRIDSIZE; // determined by the invariant length and the reduction method
constexpr index_t blockSize = CK_PARAM_BLOCKSIZE; // tunable
constexpr index_t blkGroupSize =
    CK_PARAM_BLKGROUPSIZE; // determined by the problem and the selected BlockSize

using srcLengths = Sequence<CK_PARAM_SRC_DESC_LENGTHS>;
using srcStrides = Sequence<CK_PARAM_SRC_DESC_STRIDES>;
using dstLengths = Sequence<CK_PARAM_DST_DESC_LENGTHS>;
using dstStrides = Sequence<CK_PARAM_DST_DESC_STRIDES>;

using toReduceDims  = Sequence<CK_PARAM_TOREDUCE_DIMS>;
using invariantDims = Sequence<CK_PARAM_INVARIANT_DIMS>;

constexpr ReduceTensorOp_t op          = get_reduce_op<CK_PARAM_REDUCE_OP>::op;
constexpr ReductionMethod_t reduceImpl = static_cast<ReductionMethod_t>(CK_PARAM_REDUCE_IMPL);
constexpr NanPropagation_t nanPropaOpt = CK_PARAM_NAN_PROPAGATE == 0
                                             ? NanPropagation_t::NOT_PROPAGATE_NAN
                                             : NanPropagation_t::PROPAGATE_NAN;
constexpr ReduceTensorIndices_t reduceIndicesOpt = CK_PARAM_REDUCE_INDICES == 0
                                                       ? ReduceTensorIndices_t::NO_INDICES
                                                       : ReduceTensorIndices_t::FLATTENED_INDICES;

constexpr index_t GredThreadBufferLength       = CK_PARAM_THREAD_BUFFER_LENGTH;        // tunable
constexpr index_t GredAccessesPerThreadInBlock = CK_PARAM_ACCESSES_PER_THREAD_INBLOCK; // tunable
constexpr index_t GredAccessesPerThreadInWarp  = CK_PARAM_ACCESSES_PER_THREAD_INWARP;  // tunable

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

    constexpr auto gridwise_reduce = GridwiseReduction<blkGroupSize,
                                                       gridSize,
                                                       blockSize,
                                                       srcDataType,
                                                       dstDataType,
                                                       compType,
                                                       decltype(srcDesc),
                                                       toReduceDims,
                                                       invariantDims,
                                                       decltype(dstDesc),
                                                       static_cast<index_t>(op),
                                                       static_cast<index_t>(reduceImpl),
                                                       static_cast<index_t>(nanPropaOpt),
                                                       static_cast<index_t>(reduceIndicesOpt),
                                                       GredThreadBufferLength,
                                                       GredAccessesPerThreadInBlock,
                                                       GredAccessesPerThreadInWarp>{};

    gridwise_reduce.Run<3>(alpha,
                           const_cast<const void* const __restrict__>(p_src_global),
                           beta,
                           const_cast<void* const __restrict__>(p_dst_global),
                           const_cast<void* const __restrict__>(ws_buf1_global),
                           ws_buf2_bytes_offset,
                           const_cast<void* const __restrict__>(indices_global));
};
