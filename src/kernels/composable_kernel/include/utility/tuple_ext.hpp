#ifndef _CK_TUPLE_EXT_HPP_
#define _CK_TUPLE_EXT_HPP_

#include "sequence.hpp"
#include "tuple.hpp"
#include "multi_index_transform.hpp"

namespace ck {

template <index_t... Ns>
__host__ __device__ constexpr auto make_passthrough_tuple(Sequence<Ns...>)
{
    return make_tuple( PassThrough<Ns>{}... );
};

template <index_t... Ids>
__host__ __device__ constexpr auto make_dimensions_tuple(Sequence<Ids...>)
{
    return make_tuple( Sequence<Ids>{}... );
};

template <typename Seq1, typename Seq2> 
__host__ __device__ constexpr auto make_2d_merge_transform_tuple(Seq1 X1, Seq2 X2)
{
    return make_tuple( Merge<Seq1>{}, Merge<Seq2>{} ); 
}; 

template <typename Seq> 
__host__ __device__ constexpr auto make_1d_merge_transform_tuple(Seq X)
{
    return make_tuple( Merge<Seq>{} ); 
}; 

}; // end of namespace ck

#endif


