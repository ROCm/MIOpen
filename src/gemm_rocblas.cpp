/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/gemm_rocblas.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>
#include <miopen/finddb_kernel_cache_key.hpp>
#include <miopen/gemm.hpp>

#if MIOPEN_BACKEND_HIP
#include <miopen/hipoc_kernel.hpp>
#endif

#if MIOPEN_USE_ROCBLAS
#include <half.hpp>
#include <rocblas.h>
#include <miopen/perf_field.hpp>
#endif

#include <boost/range/adaptors.hpp>
#include <tuple> // std::ignore

#if MIOPEN_USE_ROCBLAS

#define MIOPEN_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 + ROCBLAS_VERSION_MINOR)

/// Avoid warnings "The workspace_size and workspace arguments are obsolete" and
/// "disabled expansion of recursive macro" injected by rocblas headers.
#define AVOID_ROCBLAS_WRAPPERS_204 (MIOPEN_ROCBLAS_VERSION_DECIMAL >= 204)

/// Maintain API compatibility with various rocBLAS version
#define USE_GEMM_FLAGS_PACK_INT8X4 (MIOPEN_ROCBLAS_VERSION_DECIMAL >= 238)

/// Maintain API compatibility for versions not supporting FP16 alternate implementations
#define USE_GEMM_FLAGS_FP16_ALT_IMPL (MIOPEN_ROCBLAS_VERSION_DECIMAL >= 243)
/// Some 2.42 versions have rocblas_gemm_flags_fp16_alt_impl, but
/// some do not, and that leads to build errors.
/// Let's pass literal value as a workaround; there should be no harm.
#define USE_GEMM_FLAGS_FP16_ALT_IMPL_242 (MIOPEN_ROCBLAS_VERSION_DECIMAL == 242)

template <class... Ts>
auto miopen_rocblas_gemm_ex(Ts... xs)
{
#if AVOID_ROCBLAS_WRAPPERS_204
    return (rocblas_gemm_ex)(xs...);
#else
    std::size_t zero = 0;
    return rocblas_gemm_ex(xs..., &zero, nullptr);
#endif
}

template <class... Ts>
auto miopen_rocblas_gemm_strided_batched_ex(Ts... xs)
{
#if AVOID_ROCBLAS_WRAPPERS_204
    return (rocblas_gemm_strided_batched_ex)(xs...);
#else
    std::size_t zero = 0;
    return rocblas_gemm_strided_batched_ex(xs..., &zero, nullptr);
#endif
}

#endif // MIOPEN_USE_ROCBLAS

MIOPEN_DECLARE_ENV_VAR(MIOPEN_GEMM_ENFORCE_BACKEND)

namespace miopen {

#if MIOPEN_BACKEND_HIP
inline void ProfilingRecordStart(const Handle& handle, HipEventPtr& start, HipEventPtr& stop)
{
    start = make_hip_event();
    stop  = make_hip_event();
    hipEventRecord(start.get(), handle.GetStream());
}

inline void ProfilingRecordStop(const Handle& handle, HipEventPtr& start, HipEventPtr& stop)
{
    hipEventRecord(stop.get(), handle.GetStream());
    hipEventSynchronize(stop.get());
    float mS = 0;
    hipEventElapsedTime(&mS, start.get(), stop.get());
    handle.ResetKernelTime();
    handle.AccumKernelTime(mS);
}
#endif    

miopenStatus_t CallGemmRocblas(const Handle& handle,
                                GemmNewDescriptor gemm_desc,
                                ConstData_t A,
                                int a_offset,
                                ConstData_t B,
                                int b_offset,
                                Data_t C,
                                int c_offset,
                                GemmCallBackend_t gemm_backend)
{
    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    if(!gemm_desc.GetIsColMajor())
    {
        gemm_desc.isColMajor = !gemm_desc.isColMajor;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    switch(gemm_backend)
    {
    case GemmCallBackend_t::ROCBLAS: {
#if MIOPEN_USE_ROCBLAS
        MIOPEN_LOG_FUNCTION("rocBLAS");

        HipEventPtr start = nullptr;
        HipEventPtr stop  = nullptr;
        if(handle.IsProfilingEnabled())
        {
            ProfilingRecordStart(handle, start, stop);
        }

        rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

        switch(gemm_desc.dataType)
        {
        case miopenInt8x4:
        case miopenInt8: {
            assert(gemm_desc.k % 4 == 0);

            auto alpha = int(gemm_desc.alpha);
            auto beta  = int(gemm_desc.beta);

            rb_status = miopen_rocblas_gemm_ex(
                handle.rhandle().get(),
                gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.m,
                gemm_desc.n,
                gemm_desc.k,
                &alpha,
                static_cast<const int8_t*>(A) + a_offset,
                rocblas_datatype::rocblas_datatype_i8_r,
                gemm_desc.lda,
                static_cast<const int8_t*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_i8_r,
                gemm_desc.ldb,
                &beta,
                static_cast<const rocblas_int*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_i32_r,
                gemm_desc.ldc,
                static_cast<rocblas_int*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_i32_r,
                gemm_desc.ldc,
                rocblas_datatype::rocblas_datatype_i32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
#if USE_GEMM_FLAGS_PACK_INT8X4
                rocblas_gemm_flags_pack_int8x4
#else
                0
#endif
            );
        }
        break;
        case miopenInt32: break;
        case miopenHalf: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            rb_status = miopen_rocblas_gemm_ex(
                handle.rhandle().get(),
                gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.m,
                gemm_desc.n,
                gemm_desc.k,
                &alpha,
                static_cast<const rocblas_half*>(A) + a_offset,
                rocblas_datatype::rocblas_datatype_f16_r,
                gemm_desc.lda,
                static_cast<const rocblas_half*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_f16_r,
                gemm_desc.ldb,
                &beta,
                static_cast<const rocblas_half*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f16_r,
                gemm_desc.ldc,
                static_cast<rocblas_half*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f16_r,
                gemm_desc.ldc,
                rocblas_datatype::rocblas_datatype_f32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0
            );
        }
        break;

        case miopenBFloat16: {

            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            rb_status = miopen_rocblas_gemm_ex(
                handle.rhandle().get(),
                gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.m,
                gemm_desc.n,
                gemm_desc.k,
                &alpha,
                static_cast<const rocblas_bfloat16*>(A) + a_offset,
                rocblas_datatype::rocblas_datatype_bf16_r,
                gemm_desc.lda,
                static_cast<const rocblas_bfloat16*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_bf16_r,
                gemm_desc.ldb,
                &beta,
                static_cast<const rocblas_bfloat16*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_bf16_r,
                gemm_desc.ldc,
                static_cast<rocblas_bfloat16*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_bf16_r,
                gemm_desc.ldc,
                rocblas_datatype::rocblas_datatype_f32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0);
        }
        break;

        case miopenFloat: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            rb_status = miopen_rocblas_gemm_ex(
                handle.rhandle().get(),
                gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                gemm_desc.m,
                gemm_desc.n,
                gemm_desc.k,
                &alpha,
                static_cast<const float*>(A) + a_offset,
                rocblas_datatype::rocblas_datatype_f32_r,
                gemm_desc.lda,
                static_cast<const float*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_f32_r,
                gemm_desc.ldb,
                &beta,
                static_cast<const float*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f32_r,
                gemm_desc.ldc,
                static_cast<float*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f32_r,
                gemm_desc.ldc,
                rocblas_datatype::rocblas_datatype_f32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0);
        }
        break;

        case miopenDouble: {
            MIOPEN_THROW(miopenStatusBadParm,
                         "miopenDouble data type not supported by MIOpenGEMM.");
        };
        break;
        }

        if(handle.IsProfilingEnabled())
            ProfilingRecordStop(handle, start, stop);

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }
    }
    return miopenStatusUnknownError;
}

} //namespace miopen
