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

#if MIOPEN_USE_ROCBLAS
#include <half.hpp>
#include <rocblas.h>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/perf_field.hpp>
#endif

#if MIOPEN_USE_MIOPENGEMM
#include <miopen/miopengemm.hpp>
#endif

#include <boost/range/adaptors.hpp>

#if MIOPEN_USE_ROCBLAS
#define ROCBLAS_TIMING_MEMSET_SIZE (10 * 1024 * 1024)
#endif

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

#if MIOPEN_USE_ROCBLAS
// Enqueue gpu memset for rocblas kernel timing purpose
// Be careful, will set mem to 0
static void
dummy_memset(Handle& handle, Data_t mem, std::size_t mem_len, miopenDataType_t data_type)
{
    MIOPEN_LOG_I2("dummy gpu memset");

    std::size_t data_size = 0;

    switch(data_type)
    {
    case miopenInt8x4:
    case miopenInt8:
    {
        data_size = sizeof(int8_t);
        break;
    }
    case miopenInt32:
    {
        data_size = sizeof(int);
        break;
    }
    case miopenHalf:
    {
        data_size = sizeof(half_float::half);
        break;
    }
    case miopenFloat:
    {
        data_size = sizeof(float);
        break;
    }
    }

    std::size_t sz = mem_len * data_size;

    for(std::size_t i = 0; i < ROCBLAS_TIMING_MEMSET_SIZE; i += sz)
        hipMemsetAsync(mem, 0, sz, handle.GetStream());
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
    case 1: gemm_backend_env  = GemmBackend_t::rocblas; break;
    case 2: gemm_backend_env  = GemmBackend_t::miopengemm; break;
    case 3: gemm_backend_env  = GemmBackend_t::nogemmbackend; break;
    default: gemm_backend_env = gemm_backend_preferred;
    }

// make sure backend chosen based on env variable is suppported
#if MIOPEN_USE_ROCBLAS and MIOPEN_USE_MIOPENGEMM
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::rocblas: gemm_backend_enforced       = GemmBackend_t::rocblas; break;
    case GemmBackend_t::miopengemm:
        gemm_backend_enforced =
            (data_type == miopenFloat) ? GemmBackend_t::miopengemm : GemmBackend_t::rocblas;
        break;
    }
#elif MIOPEN_USE_ROCBLAS
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::rocblas:
    case GemmBackend_t::miopengemm: gemm_backend_enforced = GemmBackend_t::rocblas; break;
    }
#elif MIOPEN_USE_MIOPENGEMM
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
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

miopenStatus_t CallGemmTimeMeasure(Handle& handle,
                                   GemmDescriptor gemm_desc,
                                   ConstData_t A,
                                   int a_offset,
                                   ConstData_t B,
                                   int b_offset,
                                   Data_t C,
                                   int c_offset,
                                   std::string* kcache_key,
                                   bool time_precision,
                                   CallGemmType_t call_gemm_type,
                                   GemmBackend_t gemm_backend)
{
    switch(call_gemm_type)
    {
    case callGemm:
    {
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
                     false,
                     gemm_backend);
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
                        time_precision,
                        gemm_backend);
    }
    case callGemmStridedBatched:
    {
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
                                   false,
                                   gemm_backend);
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
                                      time_precision,
                                      gemm_backend);
    }
    case callGemmStridedBatchedSequential:
    {
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
                                             false,
                                             gemm_backend);
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
                                                time_precision,
                                                gemm_backend);
    }
    }
    return miopenStatusNotImplemented;
}

miopenStatus_t CallGemm(Handle& handle,
                        GemmDescriptor gemm_desc,
                        ConstData_t A,
                        int a_offset,
                        ConstData_t B,
                        int b_offset,
                        Data_t C,
                        int c_offset,
                        std::string* kcache_key,
                        bool enqueue_dummy_kernel,
                        GemmBackend_t gemm_backend)
{
#if !MIOPEN_USE_ROCBLAS
    (void)enqueue_dummy_kernel;
#endif

    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_desc.dataType, gemm_backend);

    // do row-to-column major conversion here
    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    switch(gemm_backend)
    {
    case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
    case GemmBackend_t::rocblas: {
#if MIOPEN_USE_ROCBLAS
        MIOPEN_LOG_FUNCTION("rocBLAS");

        HipEventPtr start = nullptr;
        HipEventPtr stop  = nullptr;
        if(handle.IsProfilingEnabled())
        {
            if(enqueue_dummy_kernel)
            {
                dummy_memset(
                    handle,
                    C,
                    gemm_desc.m * gemm_desc.n,
                    ((gemm_desc.dataType == miopenInt8 || gemm_desc.dataType == miopenInt8x4)
                         ? miopenInt32
                         : gemm_desc.dataType));
            }

            start = make_hip_event();
            stop  = make_hip_event();
            hipEventRecord(start.get(), handle.GetStream());
        }

        rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

        switch(gemm_desc.dataType)
        {
        case miopenInt8x4:
        case miopenInt8:
        {
            assert(gemm_desc.k % 4 == 0);

            auto alpha = int(gemm_desc.alpha);
            auto beta  = int(gemm_desc.beta);

            std::size_t zero = 0;
            rb_status        = rocblas_gemm_ex(
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
                0,
                &zero,
                nullptr);
        }
        break;

        case miopenInt32: break;
        case miopenHalf:
        {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            std::size_t zero = 0;
            rb_status        = rocblas_gemm_ex(
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
                0,
                &zero,
                nullptr);
        }
        break;

        case miopenFloat:
        {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            std::size_t zero = 0;
            rb_status        = rocblas_gemm_ex(
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
                0,
                &zero,
                nullptr);
        }
        break;
        }

        if(handle.IsProfilingEnabled())
        {
            hipEventRecord(stop.get(), handle.GetStream());
            hipEventSynchronize(stop.get());
            float mS = 0;
            hipEventElapsedTime(&mS, start.get(), stop.get());
            handle.ResetKernelTime();
            handle.AccumKernelTime(mS);
        }

        if(kcache_key != nullptr)
            *kcache_key = FindDbData::GetUnusedKCacheKey();

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }

    case GemmBackend_t::miopengemm: {
#if MIOPEN_USE_MIOPENGEMM
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
            *kcache_key = network_config;

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

miopenStatus_t CallGemmStridedBatched(Handle& handle,
                                      GemmDescriptor gemm_desc,
                                      ConstData_t A,
                                      int a_offset,
                                      ConstData_t B,
                                      int b_offset,
                                      Data_t C,
                                      int c_offset,
                                      std::string* kcache_key,
                                      bool enqueue_dummy_kernel,
                                      GemmBackend_t gemm_backend)
{
#if !MIOPEN_USE_ROCBLAS
    (void)enqueue_dummy_kernel;
#endif

    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_desc.dataType, gemm_backend);

    // do row-to-column major conversion here
    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    switch(gemm_backend)
    {
    case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
    case GemmBackend_t::rocblas: {
#if MIOPEN_USE_ROCBLAS
        MIOPEN_LOG_FUNCTION("rocBLAS");

        HipEventPtr start = nullptr;
        HipEventPtr stop  = nullptr;
        if(handle.IsProfilingEnabled())
        {
            if(enqueue_dummy_kernel)
            {
                dummy_memset(
                    handle,
                    C,
                    gemm_desc.m * gemm_desc.n * gemm_desc.batch_count,
                    ((gemm_desc.dataType == miopenInt8 || gemm_desc.dataType == miopenInt8x4)
                         ? miopenInt32
                         : gemm_desc.dataType));
            }

            start = make_hip_event();
            stop  = make_hip_event();
            hipEventRecord(start.get(), handle.GetStream());
        }

        rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

        switch(gemm_desc.dataType)
        {
        case miopenInt8x4:
        case miopenInt8:
        {
            assert(gemm_desc.k % 4 == 0);

            auto alpha = int(gemm_desc.alpha);
            auto beta  = int(gemm_desc.beta);

            std::size_t zero = 0;
            rb_status        = rocblas_gemm_strided_batched_ex(
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
                0,
                &zero,
                nullptr);
        }
        break;

        case miopenInt32: break;
        case miopenHalf:
        {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            std::size_t zero = 0;
            rb_status        = rocblas_gemm_strided_batched_ex(
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
                0,
                &zero,
                nullptr);
        }
        break;

        case miopenFloat:
        {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            std::size_t zero = 0;
            rb_status        = rocblas_gemm_strided_batched_ex(
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
                0,
                &zero,
                nullptr);
        }
        break;
        }

        if(handle.IsProfilingEnabled())
        {
            hipEventRecord(stop.get(), handle.GetStream());
            hipEventSynchronize(stop.get());
            float mS = 0;
            hipEventElapsedTime(&mS, start.get(), stop.get());
            handle.ResetKernelTime();
            handle.AccumKernelTime(mS);
        }

        if(kcache_key != nullptr)
            *kcache_key = FindDbData::GetUnusedKCacheKey();

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }

    case GemmBackend_t::miopengemm: {
#if MIOPEN_USE_MIOPENGEMM
        return CallGemmStridedBatchedSequential(handle,
                                                gemm_desc,
                                                A,
                                                a_offset,
                                                B,
                                                b_offset,
                                                C,
                                                c_offset,
                                                kcache_key,
                                                enqueue_dummy_kernel,
                                                gemm_backend);
#else
        return miopenStatusNotImplemented;
#endif
    }
    }

    return miopenStatusUnknownError;
}

miopenStatus_t CallGemmStridedBatchedSequential(Handle& handle,
                                                GemmDescriptor gemm_desc,
                                                ConstData_t A,
                                                int a_offset,
                                                ConstData_t B,
                                                int b_offset,
                                                Data_t C,
                                                int c_offset,
                                                std::string* kcache_key,
                                                bool enqueue_dummy_kernel,
                                                GemmBackend_t gemm_backend)
{
#if !MIOPEN_USE_ROCBLAS
    (void)enqueue_dummy_kernel;
#endif

    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_desc.dataType, gemm_backend);

    // do row-to-column major conversion here
    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    switch(gemm_backend)
    {
    case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
    case GemmBackend_t::rocblas: {
#if MIOPEN_USE_ROCBLAS
        MIOPEN_LOG_FUNCTION("rocBLAS");

        HipEventPtr start = nullptr;
        HipEventPtr stop  = nullptr;
        if(handle.IsProfilingEnabled())
        {
            if(enqueue_dummy_kernel)
            {
                dummy_memset(
                    handle,
                    C,
                    gemm_desc.m * gemm_desc.n,
                    ((gemm_desc.dataType == miopenInt8 || gemm_desc.dataType == miopenInt8x4)
                         ? miopenInt32
                         : gemm_desc.dataType));
            }

            start = make_hip_event();
            stop  = make_hip_event();
            hipEventRecord(start.get(), handle.GetStream());
        }

        rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

        switch(gemm_desc.dataType)
        {
        case miopenInt8x4:
        case miopenInt8:
        {
            assert(gemm_desc.k % 4 == 0);

            auto alpha = int(gemm_desc.alpha);
            auto beta  = int(gemm_desc.beta);

            std::size_t zero = 0;
            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = rocblas_gemm_ex(
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
                    0,
                    &zero,
                    nullptr);
            }
        }
        break;

        case miopenInt32: break;
        case miopenHalf:
        {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            std::size_t zero = 0;
            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = rocblas_gemm_ex(
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
                    0,
                    &zero,
                    nullptr);
            }
        }
        break;

        case miopenFloat:
        {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            std::size_t zero = 0;
            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = rocblas_gemm_ex(
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
                    0,
                    &zero,
                    nullptr);
            }
        }
        break;
        }

        if(handle.IsProfilingEnabled())
        {
            hipEventRecord(stop.get(), handle.GetStream());
            hipEventSynchronize(stop.get());
            float mS = 0;
            hipEventElapsedTime(&mS, start.get(), stop.get());
            handle.ResetKernelTime();
            handle.AccumKernelTime(mS);
        }

        if(kcache_key != nullptr)
            *kcache_key = FindDbData::GetUnusedKCacheKey();

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }

    case GemmBackend_t::miopengemm: {
#if MIOPEN_USE_MIOPENGEMM
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
            *kcache_key = network_config;

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
