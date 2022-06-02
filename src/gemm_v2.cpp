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
#include <miopen/gemm_v2.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>
#include <miopen/finddb_kernel_cache_key.hpp>

#if MIOPEN_BACKEND_HIP
#include <miopen/hipoc_kernel.hpp>
#endif

#if MIOPEN_USE_MIOPENTENSILE
#include <miopentensile/gemm.h>
#endif

#if MIOPEN_USE_ROCBLAS
#include <half.hpp>
#if HIP_PACKAGE_VERSION_FLAT <= 5001999999ULL
#include <rocblas.h>
#else
#include <rocblas/rocblas.h>
#endif
#include <miopen/perf_field.hpp>
#endif

#if MIOPEN_USE_MIOPENGEMM
#include <miopen/miopengemm.hpp>
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

std::ostream& operator<<(std::ostream& stream, const GemmDescriptor& gemm_desc)
{
    return stream << "{"
                  << "isColMajor " << gemm_desc.isColMajor << ", "
                  << "transA " << gemm_desc.transA << ", "
                  << "transB " << gemm_desc.transB << ", "
                  << "m " << gemm_desc.m << ", "
                  << "n " << gemm_desc.n << ", "
                  << "k " << gemm_desc.k << ", "
                  << "lda " << gemm_desc.lda << ", "
                  << "ldb " << gemm_desc.ldb << ", "
                  << "ldc " << gemm_desc.ldc << ", "
                  << "batch_count " << gemm_desc.batch_count << ", "
                  << "strideA " << gemm_desc.strideA << ", "
                  << "strideB " << gemm_desc.strideB << ", "
                  << "strideC " << gemm_desc.strideC << ", "
                  << "alpha " << gemm_desc.alpha << ", "
                  << "beta " << gemm_desc.beta << ", "
                  << "dataType " << gemm_desc.dataType << "} ";
}

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

// hacks: control GEMM backend by enviroment variable and build option
// very nasty
static GemmBackend_t enforce_gemm_backend(miopenDataType_t data_type,
                                          GemmBackend_t gemm_backend_preferred)
{
    GemmBackend_t gemm_backend_enforced = GemmBackend_t::nogemmbackend;
    GemmBackend_t gemm_backend_env      = GemmBackend_t::nogemmbackend;

    // enforce backend based on env variable
    switch(Value(MIOPEN_GEMM_ENFORCE_BACKEND{}))
    {
    case 1: gemm_backend_env = GemmBackend_t::rocblas; break;
    case 2: gemm_backend_env = GemmBackend_t::miopengemm; break;
    case 3: gemm_backend_env = GemmBackend_t::nogemmbackend; break;
    case 4: gemm_backend_env = GemmBackend_t::miopentensile; break;
    default: gemm_backend_env = gemm_backend_preferred;
    }

// make sure backend chosen based on env variable is suppported
#if MIOPEN_USE_MIOPENTENSILE
    (void)data_type;
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::rocblas:
    case GemmBackend_t::miopengemm:
    case GemmBackend_t::miopentensile: gemm_backend_enforced = GemmBackend_t::miopentensile; break;
    }
#elif MIOPEN_USE_ROCBLAS and MIOPEN_USE_MIOPENGEMM
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::miopentensile:
    case GemmBackend_t::rocblas: gemm_backend_enforced = GemmBackend_t::rocblas; break;
    case GemmBackend_t::miopengemm:
        gemm_backend_enforced =
            (data_type == miopenFloat) ? GemmBackend_t::miopengemm : GemmBackend_t::rocblas;
        break;
    }
#elif MIOPEN_USE_ROCBLAS
    (void)data_type;
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::miopentensile:
    case GemmBackend_t::rocblas:
    case GemmBackend_t::miopengemm: gemm_backend_enforced = GemmBackend_t::rocblas; break;
    }
#elif MIOPEN_USE_MIOPENGEMM
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::miopentensile:
    case GemmBackend_t::rocblas:
    case GemmBackend_t::miopengemm:
        gemm_backend_enforced =
            (data_type == miopenFloat) ? GemmBackend_t::miopengemm : GemmBackend_t::nogemmbackend;
        break;
    }
#else
    gemm_backend_enforced = GemmBackend_t::nogemmbackend;
#endif

    return gemm_backend_enforced;
}

miopenStatus_t CallGemmTimeMeasure(const Handle& handle,
                                   GemmDescriptor gemm_desc,
                                   ConstData_t A,
                                   int a_offset,
                                   ConstData_t B,
                                   int b_offset,
                                   Data_t C,
                                   int c_offset,
                                   FindDbKCacheKey* kcache_key,
                                   bool time_precision,
                                   CallGemmType_t call_gemm_type,
                                   GemmBackend_t gemm_backend,
                                   bool gfx90a_alt_impl)
{
    switch(call_gemm_type)
    {
    case callGemm: {
        if(time_precision)
        {
            // rocBLAS need a warm-up call for accurate timing
            CallGemm(handle,
                     gemm_desc,
                     A,
                     a_offset,
                     B,
                     b_offset,
                     C,
                     c_offset,
                     nullptr,
                     gemm_backend,
                     gfx90a_alt_impl);
        }

        return CallGemm(handle,
                        gemm_desc,
                        A,
                        a_offset,
                        B,
                        b_offset,
                        C,
                        c_offset,
                        kcache_key,
                        gemm_backend,
                        gfx90a_alt_impl);
    }
    case callGemmStridedBatched: {
        if(time_precision)
        {
            // rocBLAS need extra warm-up call for accurate timing
            CallGemmStridedBatched(handle,
                                   gemm_desc,
                                   A,
                                   a_offset,
                                   B,
                                   b_offset,
                                   C,
                                   c_offset,
                                   nullptr,
                                   gemm_backend,
                                   gfx90a_alt_impl);
        }

        return CallGemmStridedBatched(handle,
                                      gemm_desc,
                                      A,
                                      a_offset,
                                      B,
                                      b_offset,
                                      C,
                                      c_offset,
                                      kcache_key,
                                      gemm_backend,
                                      gfx90a_alt_impl);
    }
    case callGemmStridedBatchedSequential: {
        if(time_precision)
        {
            // rocBLAS need a warm-up call for accurate timing
            CallGemmStridedBatchedSequential(handle,
                                             gemm_desc,
                                             A,
                                             a_offset,
                                             B,
                                             b_offset,
                                             C,
                                             c_offset,
                                             nullptr,
                                             gemm_backend,
                                             gfx90a_alt_impl);
        }

        return CallGemmStridedBatchedSequential(handle,
                                                gemm_desc,
                                                A,
                                                a_offset,
                                                B,
                                                b_offset,
                                                C,
                                                c_offset,
                                                kcache_key,
                                                gemm_backend,
                                                gfx90a_alt_impl);
    }
    }
    return miopenStatusNotImplemented;
}

#if MIOPEN_USE_MIOPENTENSILE
miopenStatus_t CallGemmMIOpenTensile(const Handle& handle,
                                     GemmDescriptor gemm_desc,
                                     ConstData_t A,
                                     int a_offset,
                                     ConstData_t B,
                                     int b_offset,
                                     Data_t C,
                                     int c_offset,
                                     FindDbKCacheKey* kcache_key)
{
    MIOPEN_LOG_FUNCTION("MIOpenTensile");

    miopen_tensile_type miotsl_in_dtype, miotsl_out_dtype;
    Data_t ptrA, ptrB, ptrC;
    switch(gemm_desc.dataType)
    {
    case miopenFloat:
        miotsl_in_dtype = miopen_tensile_type_float;
        ptrA            = Data_t(reinterpret_cast<const float*>(A) + a_offset);
        ptrB            = Data_t(reinterpret_cast<const float*>(B) + b_offset);
        ptrC            = Data_t(reinterpret_cast<float*>(C) + c_offset);
        break;
    case miopenHalf:
        miotsl_in_dtype = miopen_tensile_type_half;
        ptrA            = Data_t(reinterpret_cast<const half_float::half*>(A) + a_offset);
        ptrB            = Data_t(reinterpret_cast<const half_float::half*>(B) + b_offset);
        ptrC            = Data_t(reinterpret_cast<half_float::half*>(C) + c_offset);
        break;
    case miopenBFloat16:
        miotsl_in_dtype = miopen_tensile_type_bfloat16;
        ptrA            = Data_t(reinterpret_cast<const unsigned short*>(A) + a_offset);
        ptrB            = Data_t(reinterpret_cast<const unsigned short*>(B) + b_offset);
        ptrC            = Data_t(reinterpret_cast<unsigned short*>(C) + c_offset);
        break;
    case miopenInt32:
        miotsl_in_dtype = miopen_tensile_type_int32;
        ptrA            = Data_t(reinterpret_cast<const int32_t*>(A) + a_offset);
        ptrB            = Data_t(reinterpret_cast<const int32_t*>(B) + b_offset);
        ptrC            = Data_t(reinterpret_cast<int32_t*>(C) + c_offset);
        break;
    case miopenInt8:
    case miopenInt8x4:
        miotsl_in_dtype = miopen_tensile_type_int8x4;
        ptrA            = Data_t(reinterpret_cast<const int8_t*>(A) + a_offset);
        ptrB            = Data_t(reinterpret_cast<const int8_t*>(B) + b_offset);
        ptrC            = Data_t(reinterpret_cast<int32_t*>(C) + c_offset);
        break;
    case miopenDouble:
        MIOPEN_THROW(miopenStatusBadParm, "miopenDouble data type not supported by MIOpenGEMM.");
    }
    if(gemm_desc.dataType == miopenInt8 || gemm_desc.dataType == miopenInt8x4)
    {
        miotsl_out_dtype = miopen_tensile_type_int32;
    }
    else
    {
        miotsl_out_dtype = miotsl_in_dtype;
    }

#if MIOPEN_BACKEND_HIP
    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    if(handle.IsProfilingEnabled())
        ProfilingRecordStart(handle, start, stop);
#endif

    std::size_t m = gemm_desc.m;
    std::size_t n = gemm_desc.n;
    std::size_t k = gemm_desc.k;

    auto mtA_str0  = size_t(gemm_desc.transA ? 1 : gemm_desc.lda);
    auto mtA_str1  = size_t(gemm_desc.transA ? gemm_desc.lda : 1);
    auto mtA_b_n   = size_t(gemm_desc.batch_count);
    auto mtA_b_str = size_t(gemm_desc.strideA);
    auto mtB_str0  = size_t(gemm_desc.transB ? 1 : gemm_desc.ldb);
    auto mtB_str1  = size_t(gemm_desc.transB ? gemm_desc.ldb : 1);
    auto mtB_b_n   = size_t(gemm_desc.batch_count);
    auto mtB_b_str = size_t(gemm_desc.strideB);
    auto mtC_str0  = size_t(gemm_desc.ldc);
    auto mtC_str1  = size_t(1);
    auto mtC_b_n   = size_t(gemm_desc.batch_count);
    auto mtC_b_str = size_t(gemm_desc.strideC);

    miopen_tensile_matrix mtA{
        {m, k}, {mtA_str0, mtA_str1}, {mtA_b_n, mtA_b_str}, miotsl_in_dtype, ptrA};
    miopen_tensile_matrix mtB{
        {k, n}, {mtB_str0, mtB_str1}, {mtB_b_n, mtB_b_str}, miotsl_in_dtype, ptrB};
    miopen_tensile_matrix mtC{
        {m, n}, {mtC_str0, mtC_str1}, {mtC_b_n, mtC_b_str}, miotsl_out_dtype, ptrC};

    miopen_tensile_status mt_status = miopen_tensile_status_no_solution;
#if MIOPEN_BACKEND_HIP
    mt_status = miopen_tensile_gemm_hip(
        handle.GetStream(), &mtA, &mtB, &mtC, double(gemm_desc.alpha), double(gemm_desc.beta));

    if(handle.IsProfilingEnabled())
        ProfilingRecordStop(handle, start, stop);
#else
    (void)handle;
    (void)mtA;
    (void)mtB;
    (void)mtC;
#endif

    if(kcache_key != nullptr)
        *kcache_key = FindDbKCacheKey::MakeUnused("MIOpenTensile");

    if(mt_status != miopen_tensile_status_success)
        MIOPEN_THROW(miopenStatusInternalError, "Failed to run miopen_tensile_gemm_hip");

    return miopenStatusSuccess;
}
#endif

#if MIOPEN_USE_ROCBLAS
static inline uint32_t FlagsForRocblasFp32Fp16Call(const bool gfx90aFp16Alt)
{
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
    return gfx90aFp16Alt ? rocblas_gemm_flags_fp16_alt_impl : 0;
#elif USE_GEMM_FLAGS_FP16_ALT_IMPL_242
    return gfx90aFp16Alt ? 0x4 : 0;
#else
    std::ignore = gfx90aFp16Alt;
    return 0;
#endif
#if USE_GEMM_FLAGS_FP16_ALT_IMPL_242 // -warning: macro is not used
#endif
}
#endif // MIOPEN_USE_ROCBLAS

miopenStatus_t CallGemm(const Handle& handle,
                        GemmDescriptor gemm_desc,
                        ConstData_t A,
                        int a_offset,
                        ConstData_t B,
                        int b_offset,
                        Data_t C,
                        int c_offset,
                        FindDbKCacheKey* kcache_key,
                        GemmBackend_t gemm_backend,
                        bool gfx90a_alt_impl)
{
    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_desc.dataType, gemm_backend);

// do row-to-column major conversion here
// add macro to distinguish MIOpenTensile and rocBlas logic
#if MIOPEN_USE_MIOPENTENSILE
    if(gemm_desc.isColMajor)
#else
    if(!gemm_desc.isColMajor)
#endif
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
    case GemmBackend_t::miopentensile:
#if MIOPEN_USE_MIOPENTENSILE
        std::ignore = gfx90a_alt_impl; // Not supported.
        return CallGemmMIOpenTensile(
            handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, kcache_key);
#endif
    case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
    case GemmBackend_t::rocblas: {
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
                FlagsForRocblasFp32Fp16Call(gfx90a_alt_impl));
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

        if(kcache_key != nullptr)
            *kcache_key = FindDbKCacheKey::MakeUnused("rocBlas");

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }

    case GemmBackend_t::miopengemm: {
#if MIOPEN_USE_MIOPENGEMM
        std::ignore = gfx90a_alt_impl; // Not supported.
        if(gemm_desc.dataType != miopenFloat)
            return miopenStatusNotImplemented;

        MIOPEN_LOG_FUNCTION("MIOpenGEMM");

        // making network configs for MIOpenGEMM kernel(s),
        //   using necessary and minimal info,
        //   based on info that's always true:
        //      column-major,
        //      C is not transposed,
        //      workSpace is 0,
        //      fp32
        auto gemm_desc_to_string = [&gemm_desc]() {
            return std::to_string(static_cast<int>(gemm_desc.transA)) + "_" +
                   std::to_string(static_cast<int>(gemm_desc.transB)) + "_" +
                   std::to_string(gemm_desc.lda) + "_" + std::to_string(gemm_desc.ldb) + "_" +
                   std::to_string(gemm_desc.ldc) + "_" + std::to_string(gemm_desc.m) + "_" +
                   std::to_string(gemm_desc.n) + "_" + std::to_string(gemm_desc.k);
        };

        const std::string algorithm_name = "MIOpenGEMM";
        const std::string network_config = gemm_desc_to_string();

        if(kcache_key != nullptr)
            *kcache_key = {algorithm_name, network_config};

        auto&& kernels = handle.GetKernels(algorithm_name, network_config);

        if(kernels.empty())
        {
            MIOpenGEMM::Geometry mgg(true,
                                     gemm_desc.transA,
                                     gemm_desc.transB,
                                     false,
                                     gemm_desc.lda,
                                     gemm_desc.ldb,
                                     gemm_desc.ldc,
                                     gemm_desc.m,
                                     gemm_desc.n,
                                     gemm_desc.k,
                                     0,
                                     'f');

            AddMiopengemmSolution(
                handle, algorithm_name, network_config, mgg, A, B, C, 0.003, false);

            auto&& new_kernels = handle.GetKernels(algorithm_name, network_config);

            RunMiopengemmSolution(handle,
                                  new_kernels,
                                  gemm_desc.alpha,
                                  A,
                                  a_offset,
                                  B,
                                  b_offset,
                                  gemm_desc.beta,
                                  C,
                                  c_offset);
        }
        else
        {
            RunMiopengemmSolution(handle,
                                  kernels,
                                  gemm_desc.alpha,
                                  A,
                                  a_offset,
                                  B,
                                  b_offset,
                                  gemm_desc.beta,
                                  C,
                                  c_offset);
        }

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }
    }

    return miopenStatusUnknownError;
}

miopenStatus_t CallGemmStridedBatched(const Handle& handle,
                                      GemmDescriptor gemm_desc,
                                      ConstData_t A,
                                      int a_offset,
                                      ConstData_t B,
                                      int b_offset,
                                      Data_t C,
                                      int c_offset,
                                      FindDbKCacheKey* kcache_key,
                                      GemmBackend_t gemm_backend,
                                      bool gfx90a_alt_impl)
{
    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_desc.dataType, gemm_backend);

// do row-to-column major conversion here
// add macro to distinguish MIOpenTensile and rocBlas logic
#if MIOPEN_USE_MIOPENTENSILE
    if(gemm_desc.isColMajor)
#else
    if(!gemm_desc.isColMajor)
#endif
    {
        gemm_desc.isColMajor = !gemm_desc.isColMajor;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    switch(gemm_backend)
    {
    case GemmBackend_t::miopentensile:
#if MIOPEN_USE_MIOPENTENSILE
        std::ignore = gfx90a_alt_impl; // Not supported.
        return CallGemmMIOpenTensile(
            handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, kcache_key);
#endif
    case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
    case GemmBackend_t::rocblas: {
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

            rb_status = miopen_rocblas_gemm_strided_batched_ex(
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
                gemm_desc.strideA,
                static_cast<const int8_t*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_i8_r,
                gemm_desc.ldb,
                gemm_desc.strideB,
                &beta,
                static_cast<const rocblas_int*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_i32_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                static_cast<rocblas_int*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_i32_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                gemm_desc.batch_count,
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

            rb_status = miopen_rocblas_gemm_strided_batched_ex(
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
                gemm_desc.strideA,
                static_cast<const rocblas_half*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_f16_r,
                gemm_desc.ldb,
                gemm_desc.strideB,
                &beta,
                static_cast<const rocblas_half*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f16_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                static_cast<rocblas_half*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f16_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                gemm_desc.batch_count,
                rocblas_datatype::rocblas_datatype_f32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                FlagsForRocblasFp32Fp16Call(gfx90a_alt_impl));
        }
        break;

        case miopenBFloat16: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            rb_status = miopen_rocblas_gemm_strided_batched_ex(
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
                gemm_desc.strideA,
                static_cast<const rocblas_bfloat16*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_bf16_r,
                gemm_desc.ldb,
                gemm_desc.strideB,
                &beta,
                static_cast<const rocblas_bfloat16*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_bf16_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                static_cast<rocblas_bfloat16*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_bf16_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                gemm_desc.batch_count,
                rocblas_datatype::rocblas_datatype_f32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0);
        }
        break;

        case miopenFloat: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            rb_status = miopen_rocblas_gemm_strided_batched_ex(
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
                gemm_desc.strideA,
                static_cast<const float*>(B) + b_offset,
                rocblas_datatype::rocblas_datatype_f32_r,
                gemm_desc.ldb,
                gemm_desc.strideB,
                &beta,
                static_cast<const float*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f32_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                static_cast<float*>(C) + c_offset,
                rocblas_datatype::rocblas_datatype_f32_r,
                gemm_desc.ldc,
                gemm_desc.strideC,
                gemm_desc.batch_count,
                rocblas_datatype::rocblas_datatype_f32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0);
        }
        break;

        case miopenDouble: {
            MIOPEN_THROW(miopenStatusBadParm,
                         "miopenDouble data type not supported by MIOpenGEMM.");
        }
        break;
        }

        if(handle.IsProfilingEnabled())
            ProfilingRecordStop(handle, start, stop);

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        if(kcache_key != nullptr)
            *kcache_key = FindDbKCacheKey::MakeUnused("rocBlas");

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }

    case GemmBackend_t::miopengemm: {
#if MIOPEN_USE_MIOPENGEMM
        std::ignore = gfx90a_alt_impl; // Not supported.
        return CallGemmStridedBatchedSequential(
            handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, kcache_key, gemm_backend);
#else
        return miopenStatusNotImplemented;
#endif
    }
    }

    return miopenStatusUnknownError;
}

miopenStatus_t CallGemmStridedBatchedSequential(const Handle& handle,
                                                GemmDescriptor gemm_desc,
                                                ConstData_t A,
                                                int a_offset,
                                                ConstData_t B,
                                                int b_offset,
                                                Data_t C,
                                                int c_offset,
                                                FindDbKCacheKey* kcache_key,
                                                GemmBackend_t gemm_backend,
                                                bool gfx90a_alt_impl)
{
    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_desc.dataType, gemm_backend);

// do row-to-column major conversion here
// add macro to distinguish MIOpenTensile and rocBlas logic
#if MIOPEN_USE_MIOPENTENSILE
    if(gemm_desc.isColMajor)
#else
    if(!gemm_desc.isColMajor)
#endif
    {
        gemm_desc.isColMajor = !gemm_desc.isColMajor;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    switch(gemm_backend)
    {
    case GemmBackend_t::miopentensile:
#if MIOPEN_USE_MIOPENTENSILE
        std::ignore = gfx90a_alt_impl; // Not supported.
        return CallGemmMIOpenTensile(
            handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, kcache_key);
#endif
    case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
    case GemmBackend_t::rocblas: {
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

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = miopen_rocblas_gemm_ex(
                    handle.rhandle().get(),
                    gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.m,
                    gemm_desc.n,
                    gemm_desc.k,
                    &alpha,
                    static_cast<const int8_t*>(A) + a_offset + i * gemm_desc.strideA,
                    rocblas_datatype::rocblas_datatype_i8_r,
                    gemm_desc.lda,
                    static_cast<const int8_t*>(B) + b_offset + i * gemm_desc.strideB,
                    rocblas_datatype::rocblas_datatype_i8_r,
                    gemm_desc.ldb,
                    &beta,
                    static_cast<const rocblas_int*>(C) + c_offset + i * gemm_desc.strideC,
                    rocblas_datatype::rocblas_datatype_i32_r,
                    gemm_desc.ldc,
                    static_cast<rocblas_int*>(C) + c_offset + i * gemm_desc.strideC,
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
        }
        break;
        case miopenInt32: break;
        case miopenHalf: {

            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = miopen_rocblas_gemm_ex(
                    handle.rhandle().get(),
                    gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.m,
                    gemm_desc.n,
                    gemm_desc.k,
                    &alpha,
                    static_cast<const rocblas_half*>(A) + a_offset + i * gemm_desc.strideA,
                    rocblas_datatype::rocblas_datatype_f16_r,
                    gemm_desc.lda,
                    static_cast<const rocblas_half*>(B) + b_offset + i * gemm_desc.strideB,
                    rocblas_datatype::rocblas_datatype_f16_r,
                    gemm_desc.ldb,
                    &beta,
                    static_cast<const rocblas_half*>(C) + c_offset + i * gemm_desc.strideC,
                    rocblas_datatype::rocblas_datatype_f16_r,
                    gemm_desc.ldc,
                    static_cast<rocblas_half*>(C) + c_offset + i * gemm_desc.strideC,
                    rocblas_datatype::rocblas_datatype_f16_r,
                    gemm_desc.ldc,
                    rocblas_datatype::rocblas_datatype_f32_r,
                    rocblas_gemm_algo::rocblas_gemm_algo_standard,
                    0,
                    FlagsForRocblasFp32Fp16Call(gfx90a_alt_impl));
            }
        }
        break;

        case miopenBFloat16: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = miopen_rocblas_gemm_ex(
                    handle.rhandle().get(),
                    gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.m,
                    gemm_desc.n,
                    gemm_desc.k,
                    &alpha,
                    static_cast<const rocblas_bfloat16*>(A) + a_offset + i * gemm_desc.strideA,
                    rocblas_datatype::rocblas_datatype_bf16_r,
                    gemm_desc.lda,
                    static_cast<const rocblas_bfloat16*>(B) + b_offset + i * gemm_desc.strideB,
                    rocblas_datatype::rocblas_datatype_bf16_r,
                    gemm_desc.ldb,
                    &beta,
                    static_cast<const rocblas_bfloat16*>(C) + c_offset + i * gemm_desc.strideC,
                    rocblas_datatype::rocblas_datatype_bf16_r,
                    gemm_desc.ldc,
                    static_cast<rocblas_half*>(C) + c_offset + i * gemm_desc.strideC,
                    rocblas_datatype::rocblas_datatype_bf16_r,
                    gemm_desc.ldc,
                    rocblas_datatype::rocblas_datatype_f32_r,
                    rocblas_gemm_algo::rocblas_gemm_algo_standard,
                    0,
                    0);
            }
        }
        break;

        case miopenFloat: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = miopen_rocblas_gemm_ex(
                    handle.rhandle().get(),
                    gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                    gemm_desc.m,
                    gemm_desc.n,
                    gemm_desc.k,
                    &alpha,
                    static_cast<const float*>(A) + a_offset + i * gemm_desc.strideA,
                    rocblas_datatype::rocblas_datatype_f32_r,
                    gemm_desc.lda,
                    static_cast<const float*>(B) + b_offset + i * gemm_desc.strideB,
                    rocblas_datatype::rocblas_datatype_f32_r,
                    gemm_desc.ldb,
                    &beta,
                    static_cast<const float*>(C) + c_offset + i * gemm_desc.strideC,
                    rocblas_datatype::rocblas_datatype_f32_r,
                    gemm_desc.ldc,
                    static_cast<float*>(C) + c_offset + i * gemm_desc.strideC,
                    rocblas_datatype::rocblas_datatype_f32_r,
                    gemm_desc.ldc,
                    rocblas_datatype::rocblas_datatype_f32_r,
                    rocblas_gemm_algo::rocblas_gemm_algo_standard,
                    0,
                    0);
            }
        }
        break;

        case miopenDouble: {
            MIOPEN_THROW(miopenStatusBadParm,
                         "miopenDouble data type not supported by MIOpenGEMM.");
        }
        break;
        }

        if(handle.IsProfilingEnabled())
            ProfilingRecordStop(handle, start, stop);

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        if(kcache_key != nullptr)
            *kcache_key = FindDbKCacheKey::MakeUnused("rocBlas");

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }

    case GemmBackend_t::miopengemm: {
#if MIOPEN_USE_MIOPENGEMM
        std::ignore = gfx90a_alt_impl; // Not supported.
        if(gemm_desc.dataType != miopenFloat)
            MIOPEN_THROW(miopenStatusNotImplemented, "fp16 is not implemented in MIOPENGEMM");

        MIOPEN_LOG_FUNCTION("MIOpenGEMM");

        // making network configs for MIOpenGEMM kernel(s),
        //   using necessary and minimal info,
        //   based on info that's always true:
        //      column-major,
        //      C is not transposed,
        //      workSpace is 0,
        //      fp32
        auto gemm_desc_to_string = [&gemm_desc]() {
            return std::to_string(static_cast<int>(gemm_desc.transA)) + "_" +
                   std::to_string(static_cast<int>(gemm_desc.transB)) + "_" +
                   std::to_string(gemm_desc.lda) + "_" + std::to_string(gemm_desc.ldb) + "_" +
                   std::to_string(gemm_desc.ldc) + "_" + std::to_string(gemm_desc.m) + "_" +
                   std::to_string(gemm_desc.n) + "_" + std::to_string(gemm_desc.k);
        };

        const std::string algorithm_name = "MIOpenGEMM";
        const std::string network_config = gemm_desc_to_string();

        if(kcache_key != nullptr)
            *kcache_key = {algorithm_name, network_config};

        auto&& old_kernels = handle.GetKernels(algorithm_name, network_config);

        if(old_kernels.empty())
        {
            MIOpenGEMM::Geometry mgg(true,
                                     gemm_desc.transA,
                                     gemm_desc.transB,
                                     false,
                                     gemm_desc.lda,
                                     gemm_desc.ldb,
                                     gemm_desc.ldc,
                                     gemm_desc.m,
                                     gemm_desc.n,
                                     gemm_desc.k,
                                     0,
                                     'f');

            AddMiopengemmSolution(
                handle, algorithm_name, network_config, mgg, A, B, C, 0.003, false);

            auto&& new_kernels = handle.GetKernels(algorithm_name, network_config);

            float gemm_time = 0;

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                RunMiopengemmSolution(handle,
                                      new_kernels,
                                      gemm_desc.alpha,
                                      A,
                                      a_offset + i * static_cast<int>(gemm_desc.strideA),
                                      B,
                                      b_offset + i * static_cast<int>(gemm_desc.strideB),
                                      gemm_desc.beta,
                                      C,
                                      c_offset + i * static_cast<int>(gemm_desc.strideC));

                if(handle.IsProfilingEnabled())
                {
                    if(i == gemm_desc.batch_count - 1)
                        handle.AccumKernelTime(gemm_time);
                    else
                        gemm_time += handle.GetKernelTime();
                }
            }
        }
        else
        {
            float gemm_time = 0;

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                RunMiopengemmSolution(handle,
                                      old_kernels,
                                      gemm_desc.alpha,
                                      A,
                                      a_offset + i * static_cast<int>(gemm_desc.strideA),
                                      B,
                                      b_offset + i * static_cast<int>(gemm_desc.strideB),
                                      gemm_desc.beta,
                                      C,
                                      c_offset + i * static_cast<int>(gemm_desc.strideC));

                if(handle.IsProfilingEnabled())
                {
                    if(i == gemm_desc.batch_count - 1)
                        handle.AccumKernelTime(gemm_time);
                    else
                        gemm_time += handle.GetKernelTime();
                }
            }
        }

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }
    }

    return miopenStatusUnknownError;
}

// y = w * Im2Col(x)
GemmDescriptor CreateGemmDescriptorConvFwd(const TensorDescriptor& wDesc,
                                           const TensorDescriptor& xDesc,
                                           const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType());
    if(wDesc.GetType() != miopenInt8 && wDesc.GetType() != miopenInt8x4)
        assert(wDesc.GetType() == yDesc.GetType());
#endif

    int in_c  = xDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, wDesc.GetLengths().size());
    auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, yDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = (wDesc.GetType() == miopenInt8);
    int m           = wei_k;
    int n = std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k =
        in_c * std::accumulate(wei_spatial.begin(), wei_spatial.end(), 1, std::multiplies<int>());
    int lda               = k;
    int ldb               = wDesc.GetType() == miopenInt8 ? k : n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = Col2Im(transpose(w) * dy)
GemmDescriptor CreateGemmDescriptorConvBwdData(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& dxDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_c  = dxDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, wDesc.GetLengths().size());
    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, dyDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = true;
    bool transB     = false;
    int m =
        in_c * std::accumulate(wei_spatial.begin(), wei_spatial.end(), 1, std::multiplies<int>());
    int n   = std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k   = wei_k;
    int lda = m;
    int ldb = n;
    int ldc = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// dw = dy * transpose(Im2Col(x))
GemmDescriptor CreateGemmDescriptorConvBwdWeight(const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& xDesc,
                                                 const TensorDescriptor& dwDesc)
{
#ifndef NDEBUG
    assert(dwDesc.GetType() == xDesc.GetType() && dwDesc.GetType() == dyDesc.GetType());
#endif

    std::size_t in_c  = xDesc.GetLengths()[1];
    std::size_t wei_k = dwDesc.GetLengths()[0];

    auto wei_spatial = boost::adaptors::slice(dwDesc.GetLengths(), 2, dwDesc.GetLengths().size());
    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, dyDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = true;
    int m           = wei_k;
    int n           = static_cast<int>(in_c) *
            std::accumulate(wei_spatial.begin(), wei_spatial.end(), 1, std::multiplies<int>());
    int k   = std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int lda = k;
    int ldb = k;
    int ldc = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// y = CNHW2NCHW(w * NCHW2CNHW(x))
GemmDescriptor CreateGemmDescriptorConvCNHWFwd(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType());
    if(wDesc.GetType() != miopenInt8 && wDesc.GetType() != miopenInt8x4)
        assert(wDesc.GetType() == yDesc.GetType());
#endif

    int in_n  = xDesc.GetLengths()[0];
    int in_c  = xDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, yDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = (wDesc.GetType() == miopenInt8);
    int m           = wei_k;
    int n =
        in_n * std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k                 = in_c;
    int lda               = k;
    int ldb               = wDesc.GetType() == miopenInt8 ? k : n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
GemmDescriptor CreateGemmDescriptorConvCNHWBwdData(const TensorDescriptor& wDesc,
                                                   const TensorDescriptor& dyDesc,
                                                   const TensorDescriptor& dxDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_n  = dxDesc.GetLengths()[0];
    int in_c  = dxDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, dyDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = true;
    bool transB     = false;
    int m           = in_c;
    int n =
        in_n * std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k                 = wei_k;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// y[i] = w * x[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedDescriptorConv1x1Fwd(const TensorDescriptor& wDesc,
                                                            const TensorDescriptor& xDesc,
                                                            const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType());
    if(wDesc.GetType() != miopenInt8 && wDesc.GetType() != miopenInt8x4)
        assert(wDesc.GetType() == yDesc.GetType());
#else
    (void)yDesc;
#endif

    int in_n  = xDesc.GetLengths()[0];
    int in_c  = xDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto in_spatial = boost::adaptors::slice(xDesc.GetLengths(), 2, xDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = (wDesc.GetType() == miopenInt8);
    int m           = wei_k;
    int n   = std::accumulate(in_spatial.begin(), in_spatial.end(), 1, std::multiplies<int>());
    int k   = in_c;
    int lda = k;
    int ldb = wDesc.GetType() == miopenInt8 ? k : n;
    int ldc = n;
    int batch_count       = in_n;
    long long int strideA = 0;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx[i] = transpose(w) * dy[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedDescriptorConv1x1BwdData(const TensorDescriptor& wDesc,
                                                                const TensorDescriptor& dyDesc,
                                                                const TensorDescriptor& dxDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#else
    (void)dyDesc;
#endif

    int in_n  = dxDesc.GetLengths()[0];
    int in_c  = dxDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto in_spatial = boost::adaptors::slice(dxDesc.GetLengths(), 2, dxDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = true;
    bool transB     = false;
    int m           = in_c;
    int n   = std::accumulate(in_spatial.begin(), in_spatial.end(), 1, std::multiplies<int>());
    int k   = wei_k;
    int lda = m;
    int ldb = n;
    int ldc = n;
    int batch_count       = in_n;
    long long int strideA = 0;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
GemmDescriptor CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(const TensorDescriptor& dyDesc,
                                                                  const TensorDescriptor& xDesc,
                                                                  const TensorDescriptor& dwDesc)
{
#ifndef NDEBUG
    assert(dwDesc.GetType() == xDesc.GetType() && dwDesc.GetType() == dyDesc.GetType());
#else
    (void)dyDesc;
#endif

    int in_n  = xDesc.GetLengths()[0];
    int in_c  = xDesc.GetLengths()[1];
    int wei_k = dwDesc.GetLengths()[0];

    auto in_spatial = boost::adaptors::slice(xDesc.GetLengths(), 2, xDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = true;
    int m           = wei_k;
    int n           = in_c;
    int k   = std::accumulate(in_spatial.begin(), in_spatial.end(), 1, std::multiplies<int>());
    int lda = k;
    int ldb = k;
    int ldc = n;
    int batch_count       = in_n;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// y = w * Im2Col(x)
GemmDescriptor CreateGemmDescriptorGroupConvFwd(const TensorDescriptor& wDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& yDesc,
                                                int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType() && wDesc.GetType() == yDesc.GetType());
#endif

    int in_c  = xDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, wDesc.GetLengths().size());
    auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, yDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = false;
    int m           = wei_k / groupCount;
    int n = std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k = (in_c / groupCount) *
            std::accumulate(wei_spatial.begin(), wei_spatial.end(), 1, std::multiplies<int>());
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = Col2Im(transpose(w) * dy)
GemmDescriptor CreateGemmDescriptorGroupConvBwdData(const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& dxDesc,
                                                    int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_c  = dxDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, wDesc.GetLengths().size());
    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, dyDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = true;
    bool transB     = false;
    int m           = (in_c / groupCount) *
            std::accumulate(wei_spatial.begin(), wei_spatial.end(), 1, std::multiplies<int>());
    int n   = std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k   = wei_k / groupCount;
    int lda = m;
    int ldb = n;
    int ldc = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// dw = dy * transpose(Im2Col(x))
GemmDescriptor CreateGemmDescriptorGroupConvBwdWeight(const TensorDescriptor& dyDesc,
                                                      const TensorDescriptor& xDesc,
                                                      const TensorDescriptor& dwDesc,
                                                      int groupCount)
{
#ifndef NDEBUG
    assert(dwDesc.GetType() == xDesc.GetType() && dwDesc.GetType() == dyDesc.GetType());
#endif

    int in_c  = xDesc.GetLengths()[1];
    int wei_k = dwDesc.GetLengths()[0];

    auto wei_spatial = boost::adaptors::slice(dwDesc.GetLengths(), 2, dwDesc.GetLengths().size());
    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, dyDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = true;
    int m           = wei_k / groupCount;
    int n           = (in_c / groupCount) *
            std::accumulate(wei_spatial.begin(), wei_spatial.end(), 1, std::multiplies<int>());
    int k   = std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int lda = k;
    int ldb = k;
    int ldc = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// y = CNHW2NCHW(w * NCHW2CNHW(x))
GemmDescriptor CreateGemmDescriptorGroupConvCNHWFwd(const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& xDesc,
                                                    const TensorDescriptor& yDesc,
                                                    int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType() && wDesc.GetType() == yDesc.GetType());
#endif

    int in_n  = xDesc.GetLengths()[0];
    int in_c  = xDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, yDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = false;
    bool transB     = false;
    int m           = wei_k / groupCount;
    int n =
        in_n * std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k                 = in_c / groupCount;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
GemmDescriptor CreateGemmDescriptorGroupConvCNHWBwdData(const TensorDescriptor& wDesc,
                                                        const TensorDescriptor& dyDesc,
                                                        const TensorDescriptor& dxDesc,
                                                        int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_n  = dxDesc.GetLengths()[0];
    int in_c  = dxDesc.GetLengths()[1];
    int wei_k = wDesc.GetLengths()[0];

    auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, dyDesc.GetLengths().size());

    bool isColMajor = false;
    bool transA     = true;
    bool transB     = false;
    int m           = in_c / groupCount;
    int n =
        in_n * std::accumulate(out_spatial.begin(), out_spatial.end(), 1, std::multiplies<int>());
    int k                 = wei_k / groupCount;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

} // namespace miopen
