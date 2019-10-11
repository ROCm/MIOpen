#ifndef CK_FLOAT_TYPE_AMD_HPP
#define CK_FLOAT_TYPE_AMD_HPP

namespace ck {

// For some reason, HIP compiler need this definition to generate optimal ISA
// float
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float4_t __attribute__((ext_vector_type(4)));
typedef float float32_t __attribute__((ext_vector_type(32)));

// float16
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

// bfloat16
typedef ushort ushort2_t __attribute__((ext_vector_type(2)));
typedef ushort ushort4_t __attribute__((ext_vector_type(4)));

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

// data type conversion
template <typename T>
struct type_convert
{
    template <typename X>
    __device__ T operator()(X x) const
    {
        return static_cast<T>(x);
    }
};

template <>
template <>
__device__ float type_convert<float>::operator()<ushort>(ushort x) const
{
    return bfloat16_to_float(x);
}

template <>
template <>
__device__ ushort type_convert<ushort>::operator()<float>(float x) const
{
    return float_to_bfloat16(x);
}

template <typename T>
struct inner_product_with_conversion
{
    static constexpr auto convert = type_convert<T>();

    __device__ T operator()(float a, float b) const { return convert(a) * convert(b); }

    __device__ T operator()(half2_t a, half2_t b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }

        return acc;
    }

    __device__ T operator()(half4_t a, half4_t b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }
        return acc;
    }

    __device__ T operator()(ushort2_t a, ushort2_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }

        return acc;
    }

    __device__ T operator()(ushort4_t a, ushort4_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }
        return acc;
    }
};

} // namespace ck
#endif
