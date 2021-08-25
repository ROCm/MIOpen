#ifndef CK_TENSOR_COORDINATE_HELPER_HPP
#define CK_TENSOR_COORDINATE_HELPER_HPP

#include "tensor_coordiante_hpp"

namespace ck {

template <typename TensorDesc>
__host__ __device__ constexpr auto
make_tensor_coordinate(TensorDesc, MultiIndex<TensorDesc::GetNumOfDimension()> idx)
{
    return typename TensorCoordinate<TensorDesc>::type(idx);
}

} // namespace ck
#endif
