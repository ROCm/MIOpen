#ifndef CK_DIMENSION_HPP
#define CK_DIMENSION_HPP

#include "common_header.hpp"

namespace ck {

template <index_t Length, index_t Stride>
struct NativeDimension
{
    __host__ __device__ static constexpr auto GetLength() { return Number<Length>{}; }

    __host__ __device__ static constexpr auto GetStride() { return Number<Stride>{}; }
};

} // namespace ck
#endif
