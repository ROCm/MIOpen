/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cassert>
#include <type_traits>

#include <miopen/config.h>
#include <miopen/handle.hpp>

#if MIOPEN_USE_ROCBLAS
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-macros"
#define ROCBLAS_BETA_FEATURES_API 1
#pragma clang diagnostic pop
#include <half/half.hpp>
#if MIOPEN_ROCBLAS_VERSION_FLAT < 2045000
#include <rocblas.h>
#else
#include <rocblas/rocblas.h>
/// rocblas_gemm_ex3 supports F8 datatypes.
/// strided_batched_ex3 introduced in rocblas 4.0
#define USE_ROCBLAS_EX3                                                                   \
    ((MIOPEN_ROCBLAS_VERSION_FLAT >= 4000000 && MIOPEN_ROCBLAS_VERSION_FLAT < 5000000) && \
     ROCBLAS_BETA_FEATURES_API)
#endif
#endif

namespace miopen::solver::mha {
// TODO: Issue #2748
template <typename T, typename S = std::enable_if_t<std::is_unsigned_v<T>, T>>
constexpr S Ceil(const T val, const T div)
{
    return (val - 1 + div) / div;
}

template <typename T, typename S = std::enable_if_t<std::is_unsigned_v<T>, T>>
constexpr S RoundUpToMultiple(T val, T mul)
{
    return Ceil(val, mul) * mul;
}

template <typename T>
constexpr T nextPow2(T v)
{
    static_assert(std::is_unsigned_v<T>);

    if(v == 1)
    {
        return (v << 1);
    }
    else
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        if constexpr(sizeof(T) > 1)
            v |= v >> 8;
        if constexpr(sizeof(T) > 2)
            v |= v >> 16;
        if constexpr(sizeof(T) > 4)
            v |= v >> 32;
        v++;
        return v;
    }
}

//// implies CType is always miopenFloat
//// beta is always 0.0f
//// input matricies are always row-major
inline void gemm(const Handle& handle,
                 bool transA,
                 bool transB,
                 int m,
                 int n,
                 int k,
                 int lda,
                 int ldb,
                 int ldc,
                 int batch_count,
                 long long int strideA,
                 long long int strideB,
                 long long int strideC,
                 float alpha,
                 miopenDataType_t AType,
                 ConstData_t A,
                 miopenDataType_t BType,
                 ConstData_t B,
                 Data_t C,
                 bool deterministic)
{
#if MIOPEN_USE_ROCBLAS
    rocblas_atomics_mode cur_mode =
        rocblas_atomics_mode::rocblas_atomics_allowed; // default value from rocblas
    if(deterministic)
    {
        rocblas_get_atomics_mode(handle.rhandle().get(), &cur_mode);
        if(cur_mode == rocblas_atomics_mode::rocblas_atomics_allowed)
        {
            rocblas_set_atomics_mode(handle.rhandle().get(), rocblas_atomics_not_allowed);
        }
    }

    float beta = 0.0f;

    auto cvtMiopen2Rocblas = [](miopenDataType_t miopen) {
        switch(miopen)
        {
        case miopenFloat: return rocblas_datatype::rocblas_datatype_f32_r;
#if USE_ROCBLAS_EX3
        case miopenFloat8_fnuz: return rocblas_datatype::rocblas_datatype_f8_r;
        case miopenBFloat8_fnuz: return rocblas_datatype::rocblas_datatype_bf8_r;
#endif
        default: return rocblas_datatype::rocblas_datatype_invalid;
        }
    };

    // fp32 x fp32 case
    if(AType == miopenFloat && AType == BType)
    {
        // rocblas operates in column-major mode, so A and B (and M and N) are swapped
        (rocblas_gemm_strided_batched_ex)(
            handle.rhandle().get(),
            transB ? rocblas_operation_transpose : rocblas_operation_none,
            transA ? rocblas_operation_transpose : rocblas_operation_none,
            n,
            m,
            k,
            &alpha,
            B,
            rocblas_datatype::rocblas_datatype_f32_r,
            ldb,
            strideB,
            A,
            rocblas_datatype::rocblas_datatype_f32_r,
            lda,
            strideA,
            &beta,
            C,
            rocblas_datatype::rocblas_datatype_f32_r,
            ldc,
            strideC,
            C,
            rocblas_datatype::rocblas_datatype_f32_r,
            ldc,
            strideC,
            batch_count,
            rocblas_datatype::rocblas_datatype_f32_r,
            rocblas_gemm_algo::rocblas_gemm_algo_standard,
            0,
            0);
    }
    // only bfp8 x fp32, fp32 x bfp8 and [b]fp8 x [b]fp8 combinations are supported
    else if(cvtMiopen2Rocblas(AType) != rocblas_datatype::rocblas_datatype_invalid             //
            && cvtMiopen2Rocblas(BType) != rocblas_datatype::rocblas_datatype_invalid          //
            && (AType == BType || AType == miopenBFloat8_fnuz || BType == miopenBFloat8_fnuz)) //
    {
        assert(handle.GetDeviceName() == "gfx942");
#if USE_ROCBLAS_EX3
        rocblas_gemm_strided_batched_ex3(
            handle.rhandle().get(),
            transB ? rocblas_operation_transpose : rocblas_operation_none,
            transA ? rocblas_operation_transpose : rocblas_operation_none,
            n,
            m,
            k,
            &alpha,
            B,
            cvtMiopen2Rocblas(BType),
            ldb,
            strideB,
            A,
            cvtMiopen2Rocblas(AType),
            lda,
            strideA,
            &beta,
            C,
            rocblas_datatype::rocblas_datatype_f32_r,
            ldc,
            strideC,
            C,
            rocblas_datatype::rocblas_datatype_f32_r,
            ldc,
            strideC,
            batch_count,
            AType == miopenFloat   ? rocblas_computetype::rocblas_compute_type_bf8_f8_f32
            : BType == miopenFloat ? rocblas_computetype::rocblas_compute_type_f8_bf8_f32
                                   : rocblas_computetype::rocblas_compute_type_f32,
            rocblas_gemm_algo::rocblas_gemm_algo_standard,
            0,
            0);
#else
        MIOPEN_THROW("rocblas GEMM operations is not supported!");
#endif
    }
    else
    {
        MIOPEN_THROW("Unsupported type combination for rocblas GEMM");
    }

    if(deterministic)
    {
        rocblas_set_atomics_mode(handle.rhandle().get(), cur_mode);
    }
#endif
}
} // namespace miopen::solver::mha
