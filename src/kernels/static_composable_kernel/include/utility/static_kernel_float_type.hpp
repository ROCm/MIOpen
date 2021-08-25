#ifndef CK_FLOAT_TYPE_AMD_HPP
#define CK_FLOAT_TYPE_AMD_HPP

namespace ck {

// For some reason, HIP compiler need this definition to generate optimal ISA
// float
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float4_t __attribute__((ext_vector_type(4)));
typedef float float16_t __attribute__((ext_vector_type(16)));
typedef float float32_t __attribute__((ext_vector_type(32)));

// float16
typedef _Float16 half_t;
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));

// bfloat16
typedef ushort ushort2_t __attribute__((ext_vector_type(2)));
typedef ushort ushort4_t __attribute__((ext_vector_type(4)));
typedef ushort ushort8_t __attribute__((ext_vector_type(8)));

struct c_vec32_4_t
{
    union VecType
    {
        struct
        {
            float32_t x;
            float32_t y;
            float32_t z;
            float32_t w;
        } s;
        float n[128];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        c.s.y = 0;
        c.s.z = 0;
        c.s.w = 0;
        return c;
    }
};

struct c_vec32_2_t
{
    union VecType
    {
        struct
        {
            float32_t x;
            float32_t y;
        } s;
        float n[64];
    } l;

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        c.s.y = 0;
        return c;
    }
};

struct c_vec32_2_2_t
{
    union VecType
    {
        struct
        {
            c_vec32_2_t x;
            c_vec32_2_t y;
        } s;
        float n[128];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x.l.s.x = 0;
        c.s.x.l.s.y = 0;
        c.s.y.l.s.x = 0;
        c.s.y.l.s.y = 0;
        return c;
    }
};

struct c_vec32_1_t
{
    union VecType
    {
        struct
        {
            float32_t x;
        } s;
        float n[32];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        return c;
    }
};

struct c_vec16_1_t
{
    union VecType
    {
        struct
        {
            float16_t x;
        } s;
        float n[16];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        return c;
    }
};

struct c_vec4_2_t
{
    union VecType
    {
        struct
        {
            float4_t x;
            float4_t y;
        } s;
        float n[8];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        c.s.y = 0;
        return c;
    }
};

struct c_vec4_1_t
{
    union VecType
    {
        struct
        {
            float4_t x;
        } s;
        float n[4];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        return c;
    }
};

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
struct vector_type<half_t, 1>
{
    using MemoryType = half_t;

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half_t s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<half_t*>(&v) + I) = s;
    }
};

template <>
struct vector_type<half_t, 2>
{
    using MemoryType = half2_t;

    union DataType
    {
        MemoryType vector;
        half_t scalar[2];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half_t s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<half_t*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(half_t s0, half_t s1)
    {
        DataType data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<half_t, 4>
{
    using MemoryType = half4_t;

    union DataType
    {
        MemoryType vector;
        half_t scalar[4];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half_t s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<half_t*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(half_t s0, half_t s1, half_t s2, half_t s3)
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
struct vector_type<half_t, 8>
{
    using MemoryType = half8_t;

    union DataType
    {
        MemoryType vector;
        half_t scalar[8];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half_t s, Number<I>)
    {
        static_assert(I < 8, "wrong");
        *(reinterpret_cast<half_t*>(&v) + I) = s;
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

template <>
struct vector_type<ushort, 8>
{
    using MemoryType = ushort8_t;

    union DataType
    {
        MemoryType vector;
        ushort scalar[8];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, ushort s, Number<I>)
    {
        static_assert(I < 8, "wrong");
        *(reinterpret_cast<ushort*>(&v) + I) = s;
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

    __device__ T operator()(float4_t a, float4_t b) const
    {
        const float* p_a_float = reinterpret_cast<const float*>(&a);
        const float* p_b_float = reinterpret_cast<const float*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_float[v]) * convert(p_b_float[v]);
        }

        return acc;
    }

    __device__ T operator()(float2_t a, float2_t b) const
    {
        const float* p_a_float = reinterpret_cast<const float*>(&a);
        const float* p_b_float = reinterpret_cast<const float*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_float[v]) * convert(p_b_float[v]);
        }

        return acc;
    }

    __device__ T operator()(float a, float b) const { return convert(a) * convert(b); }

    __device__ T operator()(half2_t a, half2_t b) const
    {
        const half_t* p_a_half = reinterpret_cast<const half_t*>(&a);
        const half_t* p_b_half = reinterpret_cast<const half_t*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }

        return acc;
    }

    __device__ T operator()(half4_t a, half4_t b) const
    {
        const half_t* p_a_half = reinterpret_cast<const half_t*>(&a);
        const half_t* p_b_half = reinterpret_cast<const half_t*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }
        return acc;
    }

    __device__ T operator()(half8_t a, half8_t b) const
    {
        const half_t* p_a_half = reinterpret_cast<const half_t*>(&a);
        const half_t* p_b_half = reinterpret_cast<const half_t*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 8; ++v)
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

    __device__ T operator()(ushort8_t a, ushort8_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 8; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }
        return acc;
    }
};

} // namespace ck
#endif
