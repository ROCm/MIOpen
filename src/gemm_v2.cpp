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
#include <miopen/config.h>
#include <miopen/gemm_v2.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>
#include <miopen/datatype.hpp>
#include <miopen/hipoc_kernel.hpp>

#if MIOPEN_USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
// Only enable BF8 support if hipBLASLt is 0.8 or above
#if MIOPEN_HIPBLASLT_VERSION_FLAT >= 8000
#define ENABLE_HIPBLASLT_BF8
#endif
#endif

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
#ifdef _WIN32
#define USE_ROCBLAS_GEMM_EX3 ((MIOPEN_ROCBLAS_VERSION_FLAT >= 3000000) && ROCBLAS_BETA_FEATURES_API)
#else
#define USE_ROCBLAS_GEMM_EX3 ((MIOPEN_ROCBLAS_VERSION_FLAT >= 2047000) && ROCBLAS_BETA_FEATURES_API)
#endif
#endif
#include <miopen/perf_field.hpp>
#endif

#include <boost/range/adaptors.hpp>
#include <tuple> // std::ignore

#if MIOPEN_USE_ROCBLAS

/// Avoid warnings "The workspace_size and workspace arguments are obsolete" and
/// "disabled expansion of recursive macro" injected by rocblas headers.
#define AVOID_ROCBLAS_WRAPPERS_204 (MIOPEN_ROCBLAS_VERSION_FLAT >= 2004000)

/// Maintain API compatibility for versions not supporting FP16 alternate implementations
#define USE_GEMM_FLAGS_FP16_ALT_IMPL (MIOPEN_ROCBLAS_VERSION_FLAT >= 2043000)
/// Some 2.42 versions have rocblas_gemm_flags_fp16_alt_impl, but
/// some do not, and that leads to build errors.
/// Let's pass literal value as a workaround; there should be no harm.
#define USE_GEMM_FLAGS_FP16_ALT_IMPL_242 (MIOPEN_ROCBLAS_VERSION_FLAT == 2042000)

static inline uint32_t
FlagsForRocblasFp32Fp16Call(const miopen::GemmDescriptor& desc) // bool gfx90aFp16Alt)
{
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
    return desc.gfx90a_alt_impl ? rocblas_gemm_flags_fp16_alt_impl : 0;
#elif USE_GEMM_FLAGS_FP16_ALT_IMPL_242
    return desc.gfx90a_alt_impl ? 0x4 : 0;
#else
    std::ignore = desc;
    MIOPEN_LOG_W("The gfx90aFp16Alt is not supported by rocBlas");
    return 0;
#endif
#if USE_GEMM_FLAGS_FP16_ALT_IMPL_242 // -warning: macro is not used
#endif
}

#if USE_ROCBLAS_GEMM_EX3
static inline rocblas_computetype rocBlasComputeType_ex3(const miopen::GemmDescriptor& desc)
{
    if(desc.a_cast_type == miopenFloat8 && desc.b_cast_type == miopenFloat8)
    {
        return rocblas_compute_type_f8_f8_f32;
    }
    else if(desc.a_cast_type == miopenFloat8 && desc.b_cast_type == miopenBFloat8)
    {
        return rocblas_compute_type_f8_bf8_f32;
    }
    else if(desc.a_cast_type == miopenBFloat8 && desc.b_cast_type == miopenFloat8)
    {
        return rocblas_compute_type_bf8_f8_f32;
    }
    else if(desc.a_cast_type == miopenBFloat8 && desc.b_cast_type == miopenBFloat8)
    {
        return rocblas_compute_type_bf8_bf8_f32;
    }
    else
    {
        return rocblas_compute_type_f32;
    }
}
#endif

static inline rocblas_datatype rocBlasComputeType(const miopen::GemmDescriptor& desc)
{
    if(desc.dataType == miopenInt8)
        return rocblas_datatype::rocblas_datatype_i32_r;
    else
        return rocblas_datatype::rocblas_datatype_f32_r;
}

auto rocBlasDataType(miopenDataType_t data_type)
{
    /// \todo Not all supported data types are handled here.
    /// This is fine so far because this function is used only with FP16/F8.
#if USE_ROCBLAS_GEMM_EX3
    if(data_type == miopenFloat8)
        return rocblas_datatype::rocblas_datatype_f8_r;
    if(data_type == miopenBFloat8)
        return rocblas_datatype::rocblas_datatype_bf8_r;
#endif
    if(data_type == miopenHalf)
        return rocblas_datatype::rocblas_datatype_f16_r;
    MIOPEN_THROW(miopenStatusInternalError, "Invalid data type passed");
}

template <typename T>
rocblas_status miopen_rocblas_gemm_ex3(const miopen::Handle& handle,
                                       const miopen::GemmDescriptor& gemm_desc,
                                       ConstData_t A,
                                       std::size_t a_offset,
                                       ConstData_t B,
                                       std::size_t b_offset,
                                       Data_t C,
                                       std::size_t c_offset)
{
    rocblas_status rb_status =
        rocblas_status::rocblas_status_internal_error; // cppcheck-suppress redundantInitialization
#if USE_ROCBLAS_GEMM_EX3
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    float alpha = gemm_desc.alpha;
    float beta  = gemm_desc.beta;
    auto flags  = FlagsForRocblasFp32Fp16Call(gemm_desc);
    if(gemm_desc.conv_attributes.fp8rounding_mode.Get() == miopenF8RoundingModeStochastic)
        flags = flags | rocblas_gemm_flags::rocblas_gemm_flags_stochastic_rounding;

    rb_status = // cppcheck-suppress redundantInitialization
        rocblas_gemm_ex3(handle.rhandle().get(),
                         gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                         gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                         gemm_desc.m,
                         gemm_desc.n,
                         gemm_desc.k,
                         &alpha,
                         static_cast<const T*>(A) + a_offset,
                         rocBlasDataType(gemm_desc.dataType),
                         gemm_desc.lda,
                         static_cast<const T*>(B) + b_offset,
                         rocBlasDataType(gemm_desc.dataType),
                         gemm_desc.ldb,
                         &beta,
                         static_cast<const T*>(C) + c_offset,
                         rocBlasDataType(gemm_desc.dataType),
                         gemm_desc.ldc,
                         static_cast<T*>(C) + c_offset,
                         rocBlasDataType(gemm_desc.dataType),
                         gemm_desc.ldc,
                         rocBlasComputeType_ex3(gemm_desc),
                         rocblas_gemm_algo::rocblas_gemm_algo_standard,
                         0,
                         flags); // gfx90a_alt_impl));
    return rb_status;
#pragma clang diagnostic pop
#else
    std::ignore      = A;
    std::ignore      = a_offset;
    std::ignore      = B;
    std::ignore      = b_offset;
    std::ignore      = C;
    std::ignore      = c_offset;
#endif
    MIOPEN_THROW(miopenStatusInternalError,
                 "An appropriate version of rocBLAS is required for this op");
    std::ignore = handle;
    std::ignore = gemm_desc;
    return rb_status;
}

template <class... Ts>
auto miopen_rocblas_gemm_ex(const miopen::Handle& handle,
                            const miopen::GemmDescriptor& gemm_desc,
                            Ts... xs)
{
    std::ignore = handle;
    std::ignore = gemm_desc;
#if AVOID_ROCBLAS_WRAPPERS_204
    return (rocblas_gemm_ex)(handle.rhandle().get(), xs...);
#else
    std::size_t zero = 0;
    return rocblas_gemm_ex(handle.rhandle().get(), xs..., &zero, nullptr);
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

template <typename T>
rocblas_status miopen_rocblas_gemm_strided_batched_ex3(const miopen::Handle& handle,
                                                       const miopen::GemmDescriptor& gemm_desc,
                                                       ConstData_t A,
                                                       std::size_t a_offset,
                                                       ConstData_t B,
                                                       std::size_t b_offset,
                                                       Data_t C,
                                                       std::size_t c_offset)
{
    rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;
    // Until there is a batched counter part to the ex3 rocBlas call we need to iterate over the
    // batched GEMM
    for(int bCount = 0; bCount < gemm_desc.batch_count; ++bCount)
    {
        rb_status = miopen_rocblas_gemm_ex3<T>(handle,
                                               gemm_desc,
                                               A,
                                               a_offset + (bCount * gemm_desc.strideA),
                                               B,
                                               b_offset + (bCount * gemm_desc.strideB),
                                               C,
                                               c_offset + (bCount * gemm_desc.strideC));
    }
    return rb_status;
}

#endif // MIOPEN_USE_ROCBLAS

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_GEMM_ENFORCE_BACKEND)

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
                  << "dataType " << GetDataType(gemm_desc.dataType) << ", "
                  << "a_cast_type " << GetDataType(gemm_desc.a_cast_type) << ", "
                  << "b_cast_type " << GetDataType(gemm_desc.b_cast_type) << "} ";
}

#if MIOPEN_USE_ROCBLAS

inline rocblas_atomics_mode DisableRocblasAtomics(const miopen::Handle& handle)
{
    MIOPEN_LOG_I2("");
    rocblas_atomics_mode cur_mode;
    [[maybe_unused]] rocblas_status status =
        rocblas_get_atomics_mode(handle.rhandle().get(), &cur_mode);
    assert(status == rocblas_status::rocblas_status_success);
    if(cur_mode == rocblas_atomics_allowed)
    {
        status = rocblas_set_atomics_mode(handle.rhandle().get(), rocblas_atomics_not_allowed);
        assert(status == rocblas_status::rocblas_status_success);
    }
    return cur_mode;
}

inline void SetRocblasAtomics(const miopen::Handle& handle, rocblas_atomics_mode mode)
{
    MIOPEN_LOG_I2("");
    [[maybe_unused]] rocblas_status status = rocblas_set_atomics_mode(handle.rhandle().get(), mode);
    assert(status == rocblas_status::rocblas_status_success);
}

#endif

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

#if MIOPEN_USE_HIPBLASLT

struct HipBLASLtMemoryHandles
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatmulDesc_t matmul;
    hipblasLtMatmulPreference_t pref;

    HipBLASLtMemoryHandles()
        : matA(nullptr), matB(nullptr), matC(nullptr), matD(nullptr), matmul(nullptr), pref(nullptr)
    {
    }

    ~HipBLASLtMemoryHandles()
    {
        hipblasLtMatrixLayoutDestroy(matA);
        hipblasLtMatrixLayoutDestroy(matB);
        hipblasLtMatrixLayoutDestroy(matC);
        hipblasLtMatrixLayoutDestroy(matD);
        hipblasLtMatmulDescDestroy(matmul);
        hipblasLtMatmulPreferenceDestroy(pref);
    }
};

static inline void check_hipblas_status(hipblasStatus_t status)
{
    if(status != hipblasStatus_t::HIPBLAS_STATUS_SUCCESS)
    {
        std::cout << "error msg" << std::endl;
        MIOPEN_THROW(miopenStatusInternalError, "hipBlasLt error encountered");
    }
}

template <typename DataTypeAB, typename DataTypeC>
static void miopen_hipblasLt_gemm(const miopen::Handle& handle,
                                  const miopen::GemmDescriptor& gemm_desc,
                                  ConstData_t A,
                                  std::size_t a_offset,
                                  ConstData_t B,
                                  std::size_t b_offset,
                                  hipDataType hip_type_AB,
                                  Data_t C,
                                  std::size_t c_offset,
                                  hipDataType hip_type_C,
                                  bool skip_batches)
{
    HipBLASLtMemoryHandles hipBLASLtHandles;

    if(gemm_desc.transA)
    {
        check_hipblas_status(hipblasLtMatrixLayoutCreate(
            &hipBLASLtHandles.matA, hip_type_AB, gemm_desc.k, gemm_desc.m, gemm_desc.lda));
    }
    else
    {
        check_hipblas_status(hipblasLtMatrixLayoutCreate(
            &hipBLASLtHandles.matA, hip_type_AB, gemm_desc.m, gemm_desc.k, gemm_desc.lda));
    }

    if(gemm_desc.transB)
    {
        check_hipblas_status(hipblasLtMatrixLayoutCreate(
            &hipBLASLtHandles.matB, hip_type_AB, gemm_desc.n, gemm_desc.k, gemm_desc.ldb));
    }
    else
    {
        check_hipblas_status(hipblasLtMatrixLayoutCreate(
            &hipBLASLtHandles.matB, hip_type_AB, gemm_desc.k, gemm_desc.n, gemm_desc.ldb));
    }

    check_hipblas_status(hipblasLtMatrixLayoutCreate(
        &hipBLASLtHandles.matC, hip_type_C, gemm_desc.m, gemm_desc.n, gemm_desc.ldc));
    check_hipblas_status(hipblasLtMatrixLayoutCreate(
        &hipBLASLtHandles.matD, hip_type_C, gemm_desc.m, gemm_desc.n, gemm_desc.ldc));

    if(gemm_desc.batch_count > 1 && !skip_batches)
    {
        check_hipblas_status(hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matA,
                                                               HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                               &gemm_desc.batch_count,
                                                               sizeof(gemm_desc.batch_count)));
        check_hipblas_status(
            hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matA,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &gemm_desc.strideA,
                                              sizeof(gemm_desc.strideA)));
        check_hipblas_status(hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matB,
                                                               HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                               &gemm_desc.batch_count,
                                                               sizeof(gemm_desc.batch_count)));
        check_hipblas_status(
            hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matB,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &gemm_desc.strideB,
                                              sizeof(gemm_desc.strideB)));
        check_hipblas_status(hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matC,
                                                               HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                               &gemm_desc.batch_count,
                                                               sizeof(gemm_desc.batch_count)));
        check_hipblas_status(
            hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matC,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &gemm_desc.strideC,
                                              sizeof(gemm_desc.strideC)));
        check_hipblas_status(hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matD,
                                                               HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                               &gemm_desc.batch_count,
                                                               sizeof(gemm_desc.batch_count)));
        check_hipblas_status(
            hipblasLtMatrixLayoutSetAttribute(hipBLASLtHandles.matD,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &gemm_desc.strideC,
                                              sizeof(gemm_desc.strideC)));
    }

    check_hipblas_status(
        hipblasLtMatmulDescCreate(&hipBLASLtHandles.matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t opTypeA = gemm_desc.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opTypeB = gemm_desc.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    check_hipblas_status(hipblasLtMatmulDescSetAttribute(
        hipBLASLtHandles.matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opTypeA, sizeof(opTypeA)));
    check_hipblas_status(hipblasLtMatmulDescSetAttribute(
        hipBLASLtHandles.matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opTypeB, sizeof(opTypeB)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    check_hipblas_status(hipblasLtMatmulDescSetAttribute(
        hipBLASLtHandles.matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    /// \todo Need to request additional workspace for optimal gemm performance, and pass down
    /// workspace size & pointer. --BrianHarrisonAMD June 2024
    size_t max_workspace_size = 0;
    void* workspace           = nullptr;
    check_hipblas_status(hipblasLtMatmulPreferenceCreate(&hipBLASLtHandles.pref));
    check_hipblas_status(
        hipblasLtMatmulPreferenceSetAttribute(hipBLASLtHandles.pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int requestSolutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[requestSolutions];
    int returnedAlgoCount = 0;
    check_hipblas_status(hipblasLtMatmulAlgoGetHeuristic(handle.HipblasLtHandle().get(),
                                                         hipBLASLtHandles.matmul,
                                                         hipBLASLtHandles.matA,
                                                         hipBLASLtHandles.matB,
                                                         hipBLASLtHandles.matC,
                                                         hipBLASLtHandles.matD,
                                                         hipBLASLtHandles.pref,
                                                         requestSolutions,
                                                         heuristicResult,
                                                         &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "no solution found for hipBLASLt hipBLASLtHandles.matmul");
    }

    float alpha       = gemm_desc.alpha;
    float beta        = gemm_desc.beta;
    const void* aData = static_cast<const DataTypeAB*>(A) + a_offset;
    const void* bData = static_cast<const DataTypeAB*>(B) + b_offset;
    const void* cData = static_cast<const DataTypeC*>(C) + c_offset;
    void* dData       = static_cast<DataTypeC*>(C) + c_offset;

    {
        HipEventProfiler profiler(handle);
        check_hipblas_status(hipblasLtMatmul(handle.HipblasLtHandle().get(),
                                             hipBLASLtHandles.matmul,
                                             &alpha,
                                             aData,
                                             hipBLASLtHandles.matA,
                                             bData,
                                             hipBLASLtHandles.matB,
                                             &beta,
                                             cData,
                                             hipBLASLtHandles.matC,
                                             dData,
                                             hipBLASLtHandles.matD,
                                             &heuristicResult[0].algo,
                                             workspace,
                                             max_workspace_size,
                                             handle.GetStream()));
    }
}

static void call_miopen_hipblasLt_gemm(const miopen::Handle& handle,
                                       const miopen::GemmDescriptor& gemm_desc,
                                       ConstData_t A,
                                       std::size_t a_offset,
                                       ConstData_t B,
                                       std::size_t b_offset,
                                       Data_t C,
                                       std::size_t c_offset,
                                       bool skip_batches)
{
    switch(gemm_desc.dataType)
    {
    case miopenInt8: {
        MIOPEN_THROW(miopenStatusInternalError, "miopenInt8 is not supported for hipBLASLt");
    }
    break;
    case miopenInt32: {
        MIOPEN_THROW(miopenStatusInternalError, "miopenInt32 is not supported for hipBLASLt");
    }
    break;
    case miopenHalf: {
        miopen_hipblasLt_gemm<hipblasLtHalf, hipblasLtHalf>(handle,
                                                            gemm_desc,
                                                            A,
                                                            a_offset,
                                                            B,
                                                            b_offset,
                                                            HIP_R_16F,
                                                            C,
                                                            c_offset,
                                                            HIP_R_16F,
                                                            skip_batches);
    }
    break;
    case miopenBFloat16: {
        miopen_hipblasLt_gemm<hipblasLtBfloat16, hipblasLtBfloat16>(handle,
                                                                    gemm_desc,
                                                                    A,
                                                                    a_offset,
                                                                    B,
                                                                    b_offset,
                                                                    HIP_R_16BF,
                                                                    C,
                                                                    c_offset,
                                                                    HIP_R_16BF,
                                                                    skip_batches);
    }
    break;
    case miopenFloat: {
        miopen_hipblasLt_gemm<hipblasLtFloat, hipblasLtFloat>(handle,
                                                              gemm_desc,
                                                              A,
                                                              a_offset,
                                                              B,
                                                              b_offset,
                                                              HIP_R_32F,
                                                              C,
                                                              c_offset,
                                                              HIP_R_32F,
                                                              skip_batches);
    }
    break;
    case miopenFloat8: {
        const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
        if(is_gfx94x)
        {
            miopen_hipblasLt_gemm<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz>(handle,
                                                                        gemm_desc,
                                                                        A,
                                                                        a_offset,
                                                                        B,
                                                                        b_offset,
                                                                        HIP_R_8F_E4M3_FNUZ,
                                                                        C,
                                                                        c_offset,
                                                                        HIP_R_8F_E4M3_FNUZ,
                                                                        skip_batches);
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "miopenFloat8 is only supported for hipBlasLt on gfx94x");
        }
    }
    break;
    case miopenBFloat8: {
        const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
        if(is_gfx94x)
        {
#ifdef ENABLE_HIPBLASLT_BF8
            miopen_hipblasLt_gemm<hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz>(handle,
                                                                          gemm_desc,
                                                                          A,
                                                                          a_offset,
                                                                          B,
                                                                          b_offset,
                                                                          HIP_R_8F_E5M2_FNUZ,
                                                                          C,
                                                                          c_offset,
                                                                          HIP_R_8F_E5M2_FNUZ,
                                                                          skip_batches);
#else
            MIOPEN_THROW(miopenStatusInternalError,
                         "miopenBFloat8 is not supported for this version of hipBlasLt on gfx94x");
#endif
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "miopenBFloat8 is only supported for hipBlasLt on gfx94x");
        }
    }
    break;
    case miopenDouble: {
        MIOPEN_THROW(miopenStatusInternalError, "miopenDouble is not supported for hipBlasLt");
    }
    break;
    case miopenInt64: {
        MIOPEN_THROW(miopenStatusInternalError, "miopenInt64 is not supported for hipBlasLt");
    }
    break;
    }
}
#endif

// hacks: control GEMM backend by enviroment variable and build option
// very nasty
static GemmBackend_t enforce_gemm_backend(GemmBackend_t gemm_backend_preferred)
{
    GemmBackend_t gemm_backend_env = GemmBackend_t::nogemmbackend;

    // enforce backend based on env variable
    // I have left the commented lines here to preserve values for the enforce and hint at why are
    // they 1, 3, and 5
    switch(env::value(MIOPEN_GEMM_ENFORCE_BACKEND))
    {
#if MIOPEN_USE_ROCBLAS
    case 1: gemm_backend_env = GemmBackend_t::rocblas; break;
#endif
    // case 2: gemm_backend_env = GemmBackend_t::miopengemm; break;
    case 3:
        gemm_backend_env = GemmBackend_t::nogemmbackend;
        break;
        // case 4: gemm_backend_env = GemmBackend_t::miopentensile; break;
#if MIOPEN_USE_HIPBLASLT
    case 5: gemm_backend_env = GemmBackend_t::hipblaslt; break;
#endif
    default: gemm_backend_env = gemm_backend_preferred;
    }

    return gemm_backend_env;
}

miopenStatus_t CallGemm(const Handle& handle,
                        GemmDescriptor gemm_desc,
                        ConstData_t A,
                        std::size_t a_offset,
                        ConstData_t B,
                        std::size_t b_offset,
                        Data_t C,
                        std::size_t c_offset,
                        GemmBackend_t gemm_backend)
{
    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_backend);

    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = !gemm_desc.isColMajor;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.a_cast_type, gemm_desc.b_cast_type);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    switch(gemm_backend)
    {
    case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
    case GemmBackend_t::rocblas: {
#if MIOPEN_USE_ROCBLAS
        MIOPEN_LOG_I2("rocBLAS");

        HipEventPtr start = nullptr;
        HipEventPtr stop  = nullptr;
        if(handle.IsProfilingEnabled())
        {
            ProfilingRecordStart(handle, start, stop);
        }
        rocblas_atomics_mode cur_mode =
            rocblas_atomics_mode::rocblas_atomics_allowed; // default value from rocblas
        if(gemm_desc.deterministic)
            cur_mode = DisableRocblasAtomics(handle);

        rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

        switch(gemm_desc.dataType)
        {
        case miopenInt8: {
            assert(gemm_desc.k % 4 == 0);

            auto alpha = int(gemm_desc.alpha);
            auto beta  = int(gemm_desc.beta);

            rb_status = miopen_rocblas_gemm_ex(
                handle,
                gemm_desc,
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
                rocBlasComputeType(gemm_desc), // rocblas_datatype::rocblas_datatype_i32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0);
        }
        break;
        case miopenInt32: break;
        case miopenHalf: {
            const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
            // We need ex3 API if any of the dataType or the cast type is an 8-bit floating type
            const auto needs_ex3 = [&]() {
                if((gemm_desc.dataType == miopenFloat8 || gemm_desc.dataType == miopenBFloat8) ||
                   (gemm_desc.a_cast_type == miopenFloat8 ||
                    gemm_desc.a_cast_type == miopenBFloat8) ||
                   (gemm_desc.b_cast_type == miopenBFloat8 ||
                    gemm_desc.b_cast_type == miopenFloat8))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }();
            // ex3 API only works on the gfx94x ASIC;
            if(needs_ex3)
            {
                if(is_gfx94x)
                {
                    rb_status = miopen_rocblas_gemm_ex3<rocblas_half>(
                        handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset);
                }
                else
                {
                    MIOPEN_THROW(miopenStatusInternalError,
                                 "8-bit floating types are only supported on gfx94x");
                }
            }
            else
            {
                float alpha = gemm_desc.alpha;
                float beta  = gemm_desc.beta;
                rb_status   = miopen_rocblas_gemm_ex(
                    handle,
                    gemm_desc,
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
                    rocBlasComputeType(gemm_desc),
                    rocblas_gemm_algo::rocblas_gemm_algo_standard,
                    0,
                    FlagsForRocblasFp32Fp16Call(gemm_desc)); // gfx90a_alt_impl));
            }
        }
        break;

        case miopenBFloat16: {

            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            rb_status = miopen_rocblas_gemm_ex(
                handle,
                gemm_desc,
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
                rocBlasComputeType(gemm_desc),
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0);
        }
        break;

        case miopenFloat: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            rb_status = miopen_rocblas_gemm_ex(
                handle,
                gemm_desc,
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
                rocBlasComputeType(gemm_desc), // rocblas_datatype::rocblas_datatype_f32_r,
                rocblas_gemm_algo::rocblas_gemm_algo_standard,
                0,
                0);
        }
        break;

        case miopenFloat8:
        case miopenBFloat8: {
            const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
            if(is_gfx94x)
            {
                rb_status = miopen_rocblas_gemm_ex3<char>(
                    handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset);
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "8-bit floating types are only supported on gfx94x");
            }
        };
        break;

        case miopenDouble: {
            MIOPEN_THROW(miopenStatusInternalError,
                         "miopenDouble data type not supported by rocBLAS.");
        };
        break;

        case miopenInt64: {
            MIOPEN_THROW(miopenStatusInternalError, "miopenInt64 is not currently supported.");
        }
        break;
        }

        if(handle.IsProfilingEnabled())
            ProfilingRecordStop(handle, start, stop);

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        if(gemm_desc.deterministic)
            SetRocblasAtomics(handle, cur_mode);
        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }
    case GemmBackend_t::hipblaslt: {
#if MIOPEN_USE_HIPBLASLT
        call_miopen_hipblasLt_gemm(handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, true);
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
                                      std::size_t a_offset,
                                      ConstData_t B,
                                      std::size_t b_offset,
                                      Data_t C,
                                      std::size_t c_offset,
                                      GemmBackend_t gemm_backend)
{
    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_backend);

    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = !gemm_desc.isColMajor;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.a_cast_type, gemm_desc.b_cast_type);
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
        MIOPEN_LOG_I2("rocBLAS");

        HipEventPtr start = nullptr;
        HipEventPtr stop  = nullptr;
        if(handle.IsProfilingEnabled())
        {
            ProfilingRecordStart(handle, start, stop);
        }
        rocblas_atomics_mode cur_mode =
            rocblas_atomics_mode::rocblas_atomics_allowed; // default value from rocblas
        if(gemm_desc.deterministic)
            cur_mode = DisableRocblasAtomics(handle);

        rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

        switch(gemm_desc.dataType)
        {
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
                0);
        }
        break;
        case miopenInt32: break;

        case miopenHalf: {
            const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
            // We need ex3 API if any of the dataType or the cast type is an 8-bit floating type
            const auto needs_ex3 = [&]() {
                if((gemm_desc.dataType == miopenFloat8 || gemm_desc.dataType == miopenBFloat8) ||
                   (gemm_desc.a_cast_type == miopenFloat8 ||
                    gemm_desc.a_cast_type == miopenBFloat8) ||
                   (gemm_desc.b_cast_type == miopenBFloat8 ||
                    gemm_desc.b_cast_type == miopenFloat8))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }();
            // ex3 API only works on the gfx94x ASIC;
            if(needs_ex3)
            {
                if(is_gfx94x)
                {
                    rb_status = miopen_rocblas_gemm_strided_batched_ex3<rocblas_half>(
                        handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset);
                }
                else
                {
                    MIOPEN_THROW(miopenStatusInternalError,
                                 "8-bit floating types are only supported on gfx94x");
                }
            }
            else
            {

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
                    FlagsForRocblasFp32Fp16Call(gemm_desc));
            }
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

        case miopenFloat8:
        case miopenBFloat8: {
            const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
            if(is_gfx94x)
            {
                rb_status = miopen_rocblas_gemm_strided_batched_ex3<char>(
                    handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset);
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "8-bit floating types are only supported on gfx94x");
            }

            break;
        }

        case miopenDouble: {
            MIOPEN_THROW(miopenStatusInternalError,
                         "miopenDouble data type not supported by rocBLAS.");
        }
        break;
        case miopenInt64: {
            MIOPEN_THROW(miopenStatusInternalError, "miopenInt64 is not currently supported.");
        }
        break;
        }

        if(handle.IsProfilingEnabled())
            ProfilingRecordStop(handle, start, stop);

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        if(gemm_desc.deterministic)
            SetRocblasAtomics(handle, cur_mode);

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }
    case GemmBackend_t::hipblaslt: {
#if MIOPEN_USE_HIPBLASLT
        call_miopen_hipblasLt_gemm(handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, false);
        return miopenStatusSuccess;
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
                                                std::size_t a_offset,
                                                ConstData_t B,
                                                std::size_t b_offset,
                                                Data_t C,
                                                std::size_t c_offset,
                                                GemmBackend_t gemm_backend)
{
    MIOPEN_LOG_I2("gemm_desc: " << gemm_desc);

    gemm_backend = enforce_gemm_backend(gemm_backend);

    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = !gemm_desc.isColMajor;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.a_cast_type, gemm_desc.b_cast_type);
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
        MIOPEN_LOG_I2("rocBLAS");

        HipEventPtr start = nullptr;
        HipEventPtr stop  = nullptr;
        if(handle.IsProfilingEnabled())
        {
            ProfilingRecordStart(handle, start, stop);
        }

        rocblas_atomics_mode cur_mode =
            rocblas_atomics_mode::rocblas_atomics_allowed; // default value from rocblas
        if(gemm_desc.deterministic)
        {
            cur_mode = DisableRocblasAtomics(handle);
        }
        rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

        switch(gemm_desc.dataType)
        {
        case miopenInt8: {
            assert(gemm_desc.k % 4 == 0);

            auto alpha = int(gemm_desc.alpha);
            auto beta  = int(gemm_desc.beta);

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = miopen_rocblas_gemm_ex(
                    handle,
                    gemm_desc,
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
                    rocBlasComputeType(gemm_desc), // rocblas_datatype::rocblas_datatype_i32_r,
                    rocblas_gemm_algo::rocblas_gemm_algo_standard,
                    0,
                    0);
            }
        }
        break;
        case miopenInt32: break;
        case miopenHalf: {
            const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
            // We need ex3 API if any of the dataType or the cast type is an 8-bit floating type
            const auto needs_ex3 = [&]() {
                if((gemm_desc.dataType == miopenFloat8 || gemm_desc.dataType == miopenBFloat8) ||
                   (gemm_desc.a_cast_type == miopenFloat8 ||
                    gemm_desc.a_cast_type == miopenBFloat8) ||
                   (gemm_desc.b_cast_type == miopenBFloat8 ||
                    gemm_desc.b_cast_type == miopenFloat8))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }();
            // ex3 API only works on the gfx94x ASIC;
            if(needs_ex3)
            {
                if(is_gfx94x)
                {
                    rb_status = miopen_rocblas_gemm_strided_batched_ex3<rocblas_half>(
                        handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset);
                }
                else
                {
                    MIOPEN_THROW(miopenStatusInternalError,
                                 "8-bit floating types are only supported on gfx94x");
                }
            }
            else
            {

                float alpha = gemm_desc.alpha;
                float beta  = gemm_desc.beta;

                for(int i = 0; i < gemm_desc.batch_count; ++i)
                {
                    rb_status = miopen_rocblas_gemm_ex(
                        handle,
                        gemm_desc,
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
                        rocBlasComputeType(gemm_desc), // rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        FlagsForRocblasFp32Fp16Call(gemm_desc));
                }
            }
        }
        break;

        case miopenBFloat16: {
            float alpha = gemm_desc.alpha;
            float beta  = gemm_desc.beta;

            for(int i = 0; i < gemm_desc.batch_count; ++i)
            {
                rb_status = miopen_rocblas_gemm_ex(
                    handle,
                    gemm_desc,
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
                    rocBlasComputeType(gemm_desc), // rocblas_datatype::rocblas_datatype_f32_r,
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
                    handle,
                    gemm_desc,
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
                    rocBlasComputeType(gemm_desc), // rocblas_datatype::rocblas_datatype_f32_r,
                    rocblas_gemm_algo::rocblas_gemm_algo_standard,
                    0,
                    0);
            }
        }
        break;

        case miopenFloat8:
        case miopenBFloat8: {
            const auto is_gfx94x = miopen::StartsWith(handle.GetDeviceName(), "gfx94");
            if(is_gfx94x)
            {
                rb_status = miopen_rocblas_gemm_strided_batched_ex3<char>(
                    handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset);
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "8-bit floating types are only supported on gfx94x");
            }

            break;
        }

        case miopenDouble: {
            MIOPEN_THROW(miopenStatusInternalError,
                         "miopenDouble data type not supported by rocBLAS.");
        }
        break;

        case miopenInt64: {
            MIOPEN_THROW(miopenStatusInternalError, "miopenInt64 is not currently supported.");
        }
        break;
        }

        if(handle.IsProfilingEnabled())
            ProfilingRecordStop(handle, start, stop);

        if(rb_status != rocblas_status::rocblas_status_success)
            MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

        if(gemm_desc.deterministic)
            SetRocblasAtomics(handle, cur_mode);

        return miopenStatusSuccess;
#else
        return miopenStatusNotImplemented;
#endif
    }
    case GemmBackend_t::hipblaslt: {
#if MIOPEN_USE_HIPBLASLT
        // todo bharriso - find out if we need to support iterative variant, or if using regular
        // batching is alright.
        call_miopen_hipblasLt_gemm(handle, gemm_desc, A, a_offset, B, b_offset, C, c_offset, false);
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
    if(wDesc.GetType() != miopenInt8)
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
    int lda         = k;
    int ldb         = wDesc.GetType() == miopenInt8 ? k : n;
    int ldc         = n;
    int batch_count = 1;
    auto strideA    = static_cast<long long>(0);
    auto strideB    = static_cast<long long>(0);
    auto strideC    = static_cast<long long>(0);
    float alpha     = 1.;
    float beta      = 0.;

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
                          xDesc.GetType(),
                          false};
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
    int batch_count = 1;
    auto strideA    = static_cast<long long>(0);
    auto strideB    = static_cast<long long>(0);
    auto strideC    = static_cast<long long>(0);
    float alpha     = 1.;
    float beta      = 0.;

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
                          dxDesc.GetType(),
                          false};
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
    int batch_count = 1;
    auto strideA    = static_cast<long long>(0);
    auto strideB    = static_cast<long long>(0);
    auto strideC    = static_cast<long long>(0);
    float alpha     = 1.;
    float beta      = 1.;

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
                          xDesc.GetType(),
                          false};
}

// y = CNHW2NCHW(w * NCHW2CNHW(x))
GemmDescriptor CreateGemmDescriptorConvCNHWFwd(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType());
    if(wDesc.GetType() != miopenInt8)
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
    int k           = in_c;
    int lda         = k;
    int ldb         = wDesc.GetType() == miopenInt8 ? k : n;
    int ldc         = n;
    int batch_count = 1;
    auto strideA    = static_cast<long long>(0);
    auto strideB    = static_cast<long long>(0);
    auto strideC    = static_cast<long long>(0);
    float alpha     = 1.;
    float beta      = 0.;

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
                          xDesc.GetType(),
                          false};
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
    int k           = wei_k;
    int lda         = m;
    int ldb         = n;
    int ldc         = n;
    int batch_count = 1;
    auto strideA    = static_cast<long long>(0);
    auto strideB    = static_cast<long long>(0);
    auto strideC    = static_cast<long long>(0);
    float alpha     = 1.;
    float beta      = 0.;

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
                          dxDesc.GetType(),
                          false};
}

// y[i] = w * x[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedDescriptorConv1x1Fwd(const TensorDescriptor& wDesc,
                                                            const TensorDescriptor& xDesc,
                                                            const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType());
    if(wDesc.GetType() != miopenInt8)
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
    int batch_count = in_n;
    auto strideA    = static_cast<long long>(0);
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(m) * n;
    float alpha     = 1.;
    float beta      = 0.;

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
                          xDesc.GetType(),
                          false};
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
    int batch_count = in_n;
    auto strideA    = static_cast<long long>(0);
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(m) * n;
    float alpha     = 1.;
    float beta      = 0;

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
                          dxDesc.GetType(),
                          false};
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
    int batch_count = in_n;
    auto strideA    = static_cast<long long>(m) * k;
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(0);
    float alpha     = 1.;
    float beta      = 1.;

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
                          xDesc.GetType(),
                          false};
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
    int lda         = k;
    int ldb         = n;
    int ldc         = n;
    int batch_count = groupCount;
    auto strideA    = static_cast<long long>(m) * k;
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(m) * n;
    float alpha     = 1.;
    float beta      = 0.;

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
                          xDesc.GetType(),
                          false};
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
    int batch_count = groupCount;
    auto strideA    = static_cast<long long>(m) * k;
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(m) * n;
    float alpha     = 1.;
    float beta      = 0.;

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
                          dxDesc.GetType(),
                          false};
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
    int batch_count = groupCount;
    auto strideA    = static_cast<long long>(m) * k;
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(m) * n;
    float alpha     = 1.;
    float beta      = 1.;

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
                          xDesc.GetType(),
                          false};
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
    int k           = in_c / groupCount;
    int lda         = k;
    int ldb         = n;
    int ldc         = n;
    int batch_count = groupCount;
    auto strideA    = static_cast<long long>(m) * k;
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(m) * n;
    float alpha     = 1.;
    float beta      = 0.;

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
                          xDesc.GetType(),
                          false};
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
    int k           = wei_k / groupCount;
    int lda         = m;
    int ldb         = n;
    int ldc         = n;
    int batch_count = groupCount;
    auto strideA    = static_cast<long long>(m) * k;
    auto strideB    = static_cast<long long>(k) * n;
    auto strideC    = static_cast<long long>(m) * n;
    float alpha     = 1.;
    float beta      = 0.;

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
                          dxDesc.GetType(),
                          false};
}

} // namespace miopen
