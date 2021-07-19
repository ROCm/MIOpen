#ifndef CK_DATA_TYPE_HPP
#define CK_DATA_TYPE_HPP

namespace ck {

template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<int32_t>
{
    __host__ __device__ static constexpr int32_t Min()
    {
        return std::numeric_limits<int32_t>::min();
    }

    __host__ __device__ static constexpr int32_t Max()
    {
        return std::numeric_limits<int32_t>::max();
    }
};

} // namespace ck
#endif
