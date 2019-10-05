#ifndef CK_VECTOR_TYPE_HPP
#define CK_VECTOR_TYPE_HPP

#include "config.hpp"
#include "integral_constant.hpp"

namespace ck {

template <class T, index_t N>
struct vector_type
{
    typedef struct
    {
        T scalar[N];
    } MemoryType;
};

template <>
struct vector_type<float, 1>
{
    using MemoryType = float;

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }
};

template <>
struct vector_type<float, 2>
{
    using MemoryType = float2_t;

    union DataType
    {
        MemoryType vector;
        float scalar[2];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(float s0, float s1)
    {
        DataType data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<float, 4>
{
    using MemoryType = float4_t;

    __host__ __device__ static constexpr index_t GetSize() { return 4; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }
};

template <>
struct vector_type<half, 1>
{
    using MemoryType = half;

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<half*>(&v) + I) = s;
    }
};

template <>
struct vector_type<half, 2>
{
    using MemoryType = half2_t;

    union DataType
    {
        MemoryType vector;
        half scalar[2];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<half*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(half s0, half s1)
    {
        DataType data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<half, 4>
{
    using MemoryType = half4_t;

    union DataType
    {
        MemoryType vector;
        half scalar[4];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<half*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(half s0, half s1, half s2, half s3)
    {
        DataType data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        data.scalar[2] = s2;
        data.scalar[3] = s3;
        return data.vector;
    }
};

template <>
struct vector_type<ushort, 1>
{
    using MemoryType = ushort;

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, ushort s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<ushort*>(&v) + I) = s;
    }
};

template <>
struct vector_type<ushort, 2>
{
    using MemoryType = ushort2_t;

    union DataType
    {
        MemoryType vector;
        ushort scalar[2];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, ushort s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<ushort*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(ushort s0, ushort s1)
    {
        DataType data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<ushort, 4>
{
    using MemoryType = ushort4_t;

    union DataType
    {
        MemoryType vector;
        ushort scalar[4];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, ushort s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<ushort*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(ushort s0, ushort s1, ushort s2, ushort s3)
    {
        DataType data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        data.scalar[2] = s2;
        data.scalar[3] = s3;
        return data.vector;
    }
};
} // namespace ck

#endif
