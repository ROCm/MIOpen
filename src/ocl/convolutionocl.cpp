/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <miopen/algorithm.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/env.hpp>
#include <miopen/find_db.hpp>
#include <miopen/finddb_kernel_cache_key.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel.hpp>
#include <miopen/solver.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/tensor.hpp>
#include <miopen/util.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/datatype.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/conv/compiled_in_parameters.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#if MIOPEN_USE_SCGEMM
#include <miopen/scgemm_utils.hpp>
#endif

#if MIOPEN_USE_GEMM
#include <miopen/gemm_v2.hpp>
#endif

#include <cassert>
#include <type_traits>

#include <boost/range/adaptors.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_SCGEMM)

#if MIOPEN_USE_GEMM
#ifdef CPPCHECK
// Keep the value unknown in cppcheck since this can differ between opencl and hip
static bool IsUseRocBlas;
#else
static const bool IsUseRocBlas = (MIOPEN_USE_ROCBLAS == 1);
#endif

static inline bool IsAnyBufferBF16(const TensorDescriptor& xDesc,
                                   const TensorDescriptor& yDesc,
                                   const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenBFloat16 || yDesc.GetType() == miopenBFloat16 ||
           wDesc.GetType() == miopenBFloat16;
}
#endif

size_t GetKernelGlobalWorkDim(const KernelInvoke& kernel, int dim)
{
#if(MIOPEN_BACKEND_HIP)
    return kernel.gdims[dim];
#else
    return kernel.global_work_dim[dim];
#endif
}

size_t GetKernelLocalWorkDim(const KernelInvoke& kernel, int dim)
{
#if(MIOPEN_BACKEND_HIP)
    return kernel.ldims[dim];
#else
    // sometimes local_work_dim = {0,0,0} look in issue #1724
    return kernel.local_work_dim[dim];
#endif
}

static inline void AddKernels(Handle& handle,
                              const std::string& algorithm_name,
                              const std::string& network_config,
                              const miopen::solver::ConvSolution& s,
                              std::vector<KernelInvoke>* const kernels)
{
    if(!algorithm_name.empty() && !network_config.empty())
    {
        handle.ClearKernels(algorithm_name, network_config);
    }
    else
    {
        assert(algorithm_name.empty() && network_config.empty());
    }
    int i = 0;
    for(auto& k : s.construction_params)
    {
        MIOPEN_LOG_I2(k.kernel_name);
        auto kernel = handle.AddKernel(algorithm_name,
                                       network_config,
                                       k.kernel_file,
                                       k.kernel_name,
                                       k.l_wk,
                                       k.g_wk,
                                       k.comp_options,
                                       i);
        if(kernels != nullptr)
        {
            kernels->push_back(kernel);
        }
        ++i;
    }
}

static inline void ValidateGroupCount(const TensorDescriptor& xDesc,
                                      const TensorDescriptor& wDesc,
                                      const ConvolutionDescriptor& conv)
{
    if(conv.group_count == 1)
    {
        if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
    if(conv.group_count > 1)
    {
        if(xDesc.GetLengths()[1] % conv.group_count != 0 ||
           wDesc.GetLengths()[0] % conv.group_count != 0 ||
           conv.group_count > xDesc.GetLengths()[1] || conv.group_count > wDesc.GetLengths()[0] ||
           conv.group_count < 1)
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if(xDesc.GetLengths()[1] / conv.group_count != wDesc.GetLengths()[1])
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
}

template <typename T>
inline int
EvaluateDataImplicitGemmSolution(Handle& handle,
                                 const miopen::solver::ConvSolution& solution,
                                 ConstData_t in, // Fwd: x, Bwd: dy, this is really a confusion trap
                                 ConstData_t weights,
                                 Data_t out,                      // Fwd: y, Bwd: dx
                                 const TensorDescriptor& outDesc, // Fwd: dyDesc, Bwd: dxDesc
                                 bool isForward,
                                 const std::vector<int>& strides,
                                 Data_t workSpace,
                                 const size_t workSpaceSize,
                                 T /*padding_val*/,
                                 float lowp_quant,
                                 float& elapsed)
{
    if(solution.workspce_sz != 0)
    {
        if(workSpace == nullptr || workSpaceSize < solution.workspce_sz)
            return -1;
    }

    std::vector<KernelInvoke> kernels;
    AddKernels(handle, "", "", solution, &kernels);
    auto kernel = kernels[0];

    elapsed = 0.0f;

    // For fp16/bfp16 backward data case, do zero init, bwd data with fp32 output
    // and cast from fp32 to fp16/bfp16
    if((outDesc.GetType() == miopenHalf || outDesc.GetType() == miopenBFloat16) &&
       (kernel.GetName() ==
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
        kernel.GetName() ==
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw" ||
        kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw"))
    {
        float zero = 0.f;
        TensorDescriptor workSpaceDesc(miopenFloat, outDesc.GetLengths(), outDesc.GetStrides());
        SetTensor(handle, workSpaceDesc, workSpace, &zero);
        elapsed += handle.GetKernelTime();

        for(auto& k : kernels)
        {
            k(in, weights, workSpace);
            elapsed += handle.GetKernelTime();
        }

        CastTensor(handle, &lowp_quant, workSpaceDesc, workSpace, outDesc, out, 0, 0);
        elapsed += handle.GetKernelTime();
    }
    else // All other cases w/o CastTensor needed
    {
        if((kernel.GetName() ==
            "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer") ||
           (kernel.GetName() ==
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer"))
        {
            /// \todo set zero within implicitGEMM kernel
            if(!isForward && (strides[0] > 1 || strides[1] > 1))
            {
                MIOPEN_LOG_I2("hasStride, call SetTensor with zero");
                float zero = 0.f;
                SetTensor(handle, outDesc, out, &zero);
                elapsed += handle.GetKernelTime();
            }
        }
        // clang-format off
        else if(kernel.GetName() ==
                    "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw" ||
                kernel.GetName() ==
                    "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
                kernel.GetName() ==
                    "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw")
        // clang-format on
        {
            // this kernel accumulate results into input tensor, therefore need to set zero
            float zero = 0.f;
            SetTensor(handle, outDesc, out, &zero);
            elapsed += handle.GetKernelTime();
        }
        else if(kernel.GetName() ==
                "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw")
        {
            // \todo this kernel doesn't always need to set-zero
            float zero = 0.f;
            SetTensor(handle, outDesc, out, &zero);
            elapsed += handle.GetKernelTime();
        }

        for(auto& k : kernels)
        {
            k(in, weights, out);
            elapsed += handle.GetKernelTime();
        }
    }

    return 0;
}

// cppcheck-suppress constParameter
inline int EvaluateSCGemmSolution(Handle& handle,
                                  const miopen::solver::ConvSolution& solution,
                                  ConstData_t x,
                                  ConstData_t w,
                                  Data_t y,
                                  Data_t workSpace,
                                  size_t workSpaceSize,
                                  const ConvolutionContext& params,
                                  int mask,
                                  float coef,
                                  float& elapsed)
{
#if MIOPEN_USE_SCGEMM
    // Fail if required workspace is not provided.
    if(solution.workspce_sz != 0)
    {
        if(workSpace == nullptr || workSpaceSize < solution.workspce_sz)
        {
            MIOPEN_LOG_E("Expected workspace is " << solution.workspce_sz << " but is "
                                                  << workSpaceSize);
            return -1;
        }
    }

    std::vector<KernelInvoke> kernels;
    AddKernels(handle, "", "", solution, &kernels);

    elapsed = CallSCGemm(handle, params, x, y, w, nullptr, workSpace, kernels, mask, coef);
    return 0;
#else
    std::ignore = handle;
    std::ignore = solution;
    std::ignore = x;
    std::ignore = w;
    std::ignore = y;
    std::ignore = workSpace;
    std::ignore = workSpaceSize;
    std::ignore = params;
    std::ignore = mask;
    std::ignore = coef;
    std::ignore = elapsed;
    elapsed     = 0;
    return -1;
#endif
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindWinogradSolutions(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{}))
        return {};
    try
    {
        return FindAllWinogradSolutions(ctx);
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return {};
    }
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindDataDirectSolutions(Handle& handle,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& wDesc,
                                               const TensorDescriptor& yDesc,
                                               bool exhaustiveSearch,
                                               bool isForward,
                                               const ConvolutionUserBuffers& bufs) const
{

    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
        return {};

    const auto dir    = isForward ? conv::Direction::Forward : conv::Direction::BackwardData;
    auto ctx          = ConvolutionContext{xDesc, wDesc, yDesc, *this, dir};
    ctx.do_search     = exhaustiveSearch;
    ctx.save_srch_req = true;
    ctx.general_compile_options = "";
    ctx.SetStream(&handle);
    ctx.SetBufs(bufs);
    ctx.DetectRocm();
    ctx.SetupFloats();

    try
    {
        return FindAllDirectSolutions(ctx);
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return {};
    }
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindDataImplicitGemmSolutions(Handle& handle,
                                                     const TensorDescriptor& xDesc,
                                                     const TensorDescriptor& wDesc,
                                                     const TensorDescriptor& yDesc,
                                                     bool exhaustiveSearch,
                                                     bool isForward,
                                                     const ConvolutionUserBuffers& bufs) const
{

    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{}))
        return {};

    const auto dir    = isForward ? conv::Direction::Forward : conv::Direction::BackwardData;
    auto ctx          = ConvolutionContext{xDesc, wDesc, yDesc, *this, dir};
    ctx.do_search     = exhaustiveSearch;
    ctx.save_srch_req = true;
    ctx.general_compile_options = "";
    ctx.SetStream(&handle);
    ctx.SetBufs(bufs);
    ctx.DetectRocm();
    ctx.SetupFloats();

    try
    {
        return FindAllImplicitGemmSolutions(ctx);
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return {};
    }
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindSCGemmSolutions(Handle& handle,
                                           const TensorDescriptor& xDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& yDesc,
                                           bool exhaustiveSearch,
                                           bool isForward,
                                           const ConvolutionUserBuffers& bufs) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_SCGEMM{}))
        return {};

    const auto dir    = isForward ? conv::Direction::Forward : conv::Direction::BackwardData;
    auto ctx          = ConvolutionContext{xDesc, wDesc, yDesc, *this, dir};
    ctx.do_search     = exhaustiveSearch;
    ctx.save_srch_req = true;
    ctx.general_compile_options = "";
    ctx.SetStream(&handle);
    ctx.SetBufs(bufs);
    ctx.DetectRocm();
    ctx.SetupFloats();

    try
    {
        return FindAllFwdSCGemmSolutions(ctx);
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return {};
    }
}

template <class InvokeParams>
static void EvaluateInvokers(Handle& handle,
                             const std::vector<solver::ConvSolution>& solutions,
                             const AlgorithmName& algorithm_name,
                             const NetworkConfig& network_config,
                             const InvokeParams& invoke_ctx,
                             DbRecord& record)
{
    miopen::solver::ConvSolution selected{miopenStatusUnknownError};
    float best = std::numeric_limits<float>::max();
    Invoker best_invoker;

    for(const auto& sol : solutions)
    {
        if(sol.workspce_sz > 0 &&
           (invoke_ctx.workSpace == nullptr || invoke_ctx.workSpaceSize < sol.workspce_sz))
        {
            MIOPEN_LOG_I("Skipping solver <" << sol.solver_id << "> due to insufficient workspace ("
                                             << invoke_ctx.workSpaceSize
                                             << " < "
                                             << sol.workspce_sz
                                             << ")");
            continue;
        }

        if(!sol.invoker_factory)
            MIOPEN_THROW("Invoker is not provided by solver " + sol.solver_id);

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        invoker(handle, invoke_ctx);
        const auto elapsed = handle.GetKernelTime();

        MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ") << best);
        if(elapsed < best)
        {
            best         = elapsed;
            selected     = sol;
            best_invoker = invoker;
        }
    }

    if(selected.Succeeded())
    {
        handle.RegisterInvoker(best_invoker, network_config, selected.solver_id, algorithm_name);
        MIOPEN_LOG_I(
            "Selected: " << selected << ": " << best << ", workspce_sz = " << selected.workspce_sz);
        record.SetValues(algorithm_name,
                         FindDbData{selected.solver_id,
                                    best,
                                    selected.workspce_sz,
                                    FindDbKCacheKey::MakeUnused(algorithm_name)});
    }
}

static void DirConvFindCore(Handle& handle,
                            const TensorDescriptor& xDesc,
                            ConstData_t x,
                            const TensorDescriptor& wDesc,
                            ConstData_t w,
                            const TensorDescriptor& yDesc,
                            Data_t y,
                            Data_t workSpace,
                            size_t workSpaceSize,
                            const ConvolutionDescriptor& conv,
                            bool exhaustiveSearch,
                            DbRecord& record)
{
    AutoEnableProfiling enableProfiling{handle};

    ValidateGroupCount(xDesc, wDesc, conv);

    auto ctx = ConvolutionContext{xDesc, wDesc, yDesc, conv, conv::Direction::Forward};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    const auto use_winograd_only = conv.IsWinograd3x3SupportedAndFast(ctx);

#if MIOPEN_USE_GEMM
    if(!use_winograd_only && !miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}) &&
       !(IsAnyBufferBF16(xDesc, yDesc, wDesc) && !IsUseRocBlas))
    { // GEMM algo
        std::size_t in_n, in_c;
        std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

        std::size_t wei_k = wDesc.GetLengths()[0];

        std::size_t spatial_dim = conv.GetSpatialDimension();

        auto in_spatial  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
        auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
        auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

        float time_gemm           = 0;
        const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));
        // Use transpose path 1x1, stride=2
        if(conv.GetSpatialDimension() == 2 &&
           miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }))
        {
            size_t workspace_req = conv.ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
            if(workSpace != nullptr && workSpaceSize >= workspace_req)
            {
                if(conv.group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1 u2xv2");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1 u2xv2");
                }

                // y = CNHW2NCHW(w * NCHW2CNHW(x))
                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    in_c,
                                    in_spatial[0],
                                    in_spatial[1],
                                    out_spatial[0],
                                    out_spatial[1],
                                    x,
                                    workSpace,
                                    0,
                                    0,
                                    conv.GetConvStrides()[0],
                                    conv.GetConvStrides()[1],
                                    xDesc.GetType());
                time_gemm = handle.GetKernelTime();

                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t x_t_size = in_n * in_c * out_spatial_size;

                std::size_t wksp_offset = 0;
                if(wDesc.GetType() == miopenInt8)
                {
                    wksp_offset = x_t_size;
                    transpose_packed_MN2NM(handle,
                                           in_c,
                                           static_cast<int>(in_n * out_spatial_size),
                                           0,
                                           wksp_offset,
                                           workSpace,
                                           workSpace,
                                           xDesc.GetType());

                    time_gemm += handle.GetKernelTime();

                    x_t_size *= 2;
                }
                if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
                   (yDesc.GetType() == miopenInt32 || yDesc.GetType() == miopenFloat))
                    x_t_size /= 4;

                FindDbKCacheKey kcache_key;

                GemmDescriptor gemm_desc =
                    conv.group_count > 1 ? CreateGemmDescriptorGroupConvCNHWFwd(
                                               wDesc, xDesc, yDesc, conv.group_count)
                                         : CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

                miopenStatus_t gemm_status =
                    CallGemmTimeMeasure(handle,
                                        gemm_desc,
                                        w,
                                        0,
                                        workSpace,
                                        wksp_offset,
                                        workSpace,
                                        x_t_size,
                                        &kcache_key,
                                        time_precision,
                                        conv.group_count > 1 ? callGemmStridedBatched : callGemm);

                time_gemm += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    wei_k,
                                    out_spatial[0],
                                    out_spatial[1],
                                    out_spatial[0],
                                    out_spatial[1],
                                    workSpace,
                                    y,
                                    x_t_size,
                                    0,
                                    1,
                                    1,
                                    yDesc.GetType());
                time_gemm += handle.GetKernelTime();

                if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
                   yDesc.GetType() != miopenInt32)
                {
                    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                    CastTensor(handle, &conv.lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                    time_gemm += handle.GetKernelTime();
                }

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues(
                        "miopenConvolutionFwdAlgoGEMM",
                        FindDbData{
                            "gemm", time_gemm, workspace_req, kcache_key}); // Todo: gemm solver id?
            }
        }
        // 1x1_stride=1 with GEMM and zero workspace
        else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
                miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
        {
            if(conv.group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");
            }

            // y = w * x
            FindDbKCacheKey kcache_key;
            miopenStatus_t gemm_status = miopenStatusNotInitialized;

            if(wDesc.GetType() == miopenInt8)
            {
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

                std::size_t out_offset      = 0;
                std::size_t in_offset       = 0;
                std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                              in_spatial.end(),
                                                              std::size_t(1),
                                                              std::multiplies<std::size_t>());
                transpose_packed_MN2NM(
                    handle, in_c, in_spatial_size, in_offset, 0, x, workSpace, xDesc.GetType());

                time_gemm += (in_n * handle.GetKernelTime());

                gemm_status = CallGemmTimeMeasure(handle,
                                                  gemm_desc,
                                                  w,
                                                  0,
                                                  workSpace,
                                                  0,
                                                  y,
                                                  out_offset,
                                                  &kcache_key,
                                                  time_precision,
                                                  callGemm);

                time_gemm += (in_n * handle.GetKernelTime());
            }
            else
            {
                GemmDescriptor gemm_desc =
                    conv.group_count > 1
                        ? CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count)
                        : CreateGemmStridedBatchedDescriptorConv1x1Fwd(wDesc, xDesc, yDesc);

                gemm_status = CallGemmTimeMeasure(handle,
                                                  gemm_desc,
                                                  w,
                                                  0,
                                                  x,
                                                  0,
                                                  y,
                                                  0,
                                                  &kcache_key,
                                                  time_precision,
                                                  callGemmStridedBatched);

                time_gemm = handle.GetKernelTime();
                if(conv.group_count > 1)
                    time_gemm *= in_n;
            }

            if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
               yDesc.GetType() != miopenInt32)
            {
                TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                CastTensor(handle, &conv.lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                time_gemm += handle.GetKernelTime();
            }

            if(gemm_status == miopenStatusSuccess)
                record.SetValues(
                    "miopenConvolutionFwdAlgoGEMM",
                    FindDbData{"gemm", time_gemm, 0, kcache_key}); // Todo: gemm solver id?
        }
        // if not 1x1
        else if(workSpace != nullptr &&
                workSpaceSize >=
                    (conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * conv.group_count))
        {
            if(conv.group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");
            }

            // y = w * Im2Col(x)
            float time_im2col = 0;
            int in_offset     = 0;
            time_im2col       = Im2ColGPU(handle,
                                    conv.GetSpatialDimension(),
                                    x,
                                    in_offset,
                                    in_c,
                                    in_spatial,
                                    wei_spatial,
                                    out_spatial,
                                    conv.GetConvPads(),
                                    conv.GetConvStrides(),
                                    conv.GetConvDilations(),
                                    workSpace,
                                    xDesc.GetType());

            std::size_t wksp_offset = 0;
            if(wDesc.GetType() == miopenInt8)
            {
                std::size_t wei_spatial_size = std::accumulate(wei_spatial.begin(),
                                                               wei_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                wksp_offset = in_c * wei_spatial_size * out_spatial_size;

                transpose_packed_MN2NM(handle,
                                       static_cast<int>(in_c * wei_spatial_size),
                                       out_spatial_size,
                                       0,
                                       wksp_offset,
                                       workSpace,
                                       workSpace,
                                       xDesc.GetType());
                time_gemm += (in_n * handle.GetKernelTime());
            }

            FindDbKCacheKey kcache_key;

            GemmDescriptor gemm_desc =
                conv.group_count > 1
                    ? CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count)
                    : CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

            miopenStatus_t gemm_status = CallGemmTimeMeasure(
                handle,
                gemm_desc,
                w,
                0,
                workSpace,
                wksp_offset,
                y,
                0,
                &kcache_key,
                time_precision,
                conv.group_count > 1 ? callGemmStridedBatched : callGemm,
                (conv.group_count > 1 || wDesc.GetType() == miopenInt8 ||
                 wDesc.GetType() == miopenInt8x4 || wDesc.GetType() == miopenBFloat16)
                    ? GemmBackend_t::rocblas
                    : GemmBackend_t::miopengemm);

            time_gemm += (in_n * (time_im2col + handle.GetKernelTime()));

            if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
               yDesc.GetType() != miopenInt32)
            {
                TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                CastTensor(handle, &conv.lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                time_gemm += handle.GetKernelTime();
            }

            if(gemm_status == miopenStatusSuccess)
                record.SetValues(
                    "miopenConvolutionFwdAlgoGEMM",
                    FindDbData{"gemm",
                               time_gemm,
                               (conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * conv.group_count),
                               kcache_key}); // Todo: gemm solver id?
        }
    }
#endif

    const auto network_config = ctx.BuildConfKey();
    const auto invoke_ctx =
        conv::DataInvokeParams{{xDesc, x, wDesc, w, yDesc, y}, workSpace, workSpaceSize};

    // Winograd algo
    {
        ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
        bufs.SetFwd(x, w, y);
        ctx.SetBufs(bufs);
        const auto all = conv.FindWinogradSolutions(ctx);
        PrecompileSolutions(handle, all);
        const auto algorithm_name = AlgorithmName{"miopenConvolutionFwdAlgoWinograd"};
        PrecompileSolutions(handle, all);
        EvaluateInvokers(handle, all, algorithm_name, network_config, invoke_ctx, record);
    }

    // Direct algo
    if(!use_winograd_only)
    {
        ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
        bufs.SetFwd(x, w, y);
        const auto all =
            conv.FindDataDirectSolutions(handle, xDesc, wDesc, yDesc, exhaustiveSearch, true, bufs);
        PrecompileSolutions(handle, all);
        const auto algorithm_name = AlgorithmName{"miopenConvolutionFwdAlgoDirect"};
        EvaluateInvokers(handle, all, algorithm_name, network_config, invoke_ctx, record);
    }

    // Implicit GEMM algo
    if(!use_winograd_only)
    {
        ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
        bufs.SetFwd(x, w, y);
        const auto all = conv.FindDataImplicitGemmSolutions(
            handle, xDesc, wDesc, yDesc, exhaustiveSearch, true, bufs);
        PrecompileSolutions(handle, all);
        miopen::solver::ConvSolution selected{miopenStatusUnknownError};
        float best = std::numeric_limits<float>::max();
        visit_float(xDesc.GetType(), [&](auto as_float) {
            for(const auto& sol : all)
            {
                float elapsed = 0.0f;
                const int rc  = EvaluateDataImplicitGemmSolution(handle,
                                                                sol,
                                                                x,
                                                                w,
                                                                y,
                                                                yDesc,
                                                                true,
                                                                conv.GetConvStrides(),
                                                                workSpace,
                                                                workSpaceSize,
                                                                as_float(0.0f),
                                                                conv.lowp_quant,
                                                                elapsed);

                if(rc != 0)
                {
                    MIOPEN_LOG_E(sol << " returns " << rc);
                }
                else
                {
                    MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ")
                                     << best);
                    if(elapsed < best)
                    {
                        best     = elapsed;
                        selected = sol;
                    }
                }
            }
        });

        if(selected.Succeeded())
        {
            const std::string algorithm_name = "miopenConvolutionFwdAlgoImplicitGEMM";
            AddKernels(handle, algorithm_name, network_config, selected, nullptr);
            MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                      << selected.workspce_sz);
            record.SetValues(algorithm_name,
                             FindDbData{selected.solver_id,
                                        best,
                                        selected.workspce_sz,
                                        {algorithm_name, network_config}});
        }
    }

    // FFT algo
    if(!use_winograd_only && conv.GetSpatialDimension() == 2 &&
       miopen::all_of(conv.GetConvDilations(), [](auto v) { return v == 1; }) &&
       conv.group_count == 1 && wDesc.GetType() != miopenInt8 && wDesc.GetType() != miopenInt8x4)
    {
        std::vector<KernelInvoke> kernels_fft;
        size_t workspace_fft = conv.ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc);
        if(conv.FindFwdFFTKernel(
               handle, xDesc, wDesc, yDesc, workspace_fft, kernels_fft, network_config) == 0)
        {
            (void)kernels_fft; // not used now, but needed as fft coverage widens
            if(workSpace != nullptr && workSpaceSize >= workspace_fft)
            {
                float time_fft = conv.ExecuteFwdFFTKernel(handle,
                                                          xDesc,
                                                          x,
                                                          wDesc,
                                                          w,
                                                          yDesc,
                                                          y,
                                                          workSpace,
                                                          workSpaceSize,
                                                          network_config,
                                                          true);
                record.SetValues("miopenConvolutionFwdAlgoFFT",
                                 FindDbData{"fft",
                                            time_fft,
                                            workspace_fft,
                                            {"miopenConvolutionFwdAlgoFFT",
                                             network_config}}); // Todo: fft solver id?
            }
        }
    }

    // static compiled gemm algo
    if(!use_winograd_only)
    {
        ConvolutionContext params(xDesc, wDesc, yDesc, conv, conv::Direction::Forward, 0);
        ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
        bufs.SetFwd(x, w, y);
        const auto all =
            conv.FindSCGemmSolutions(handle, xDesc, wDesc, yDesc, exhaustiveSearch, true, bufs);
        PrecompileSolutions(handle, all);
        miopen::solver::ConvSolution selected{miopenStatusUnknownError};

        float best = std::numeric_limits<float>::max();

        visit_float(xDesc.GetType(), [&](auto as_float) {
            for(const auto& sol : all)
            {

                float elapsed = 0.0f;
                const int rc  = EvaluateSCGemmSolution(handle,
                                                      sol,
                                                      x,
                                                      w,
                                                      y,
                                                      workSpace,
                                                      workSpaceSize,
                                                      params,
                                                      0,
                                                      as_float(0.0f),
                                                      elapsed);
                if(rc != 0)
                {
                    MIOPEN_LOG_E(sol << " returns " << rc);
                }
                else
                {
                    MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ")
                                     << best);
                    if(elapsed < best)
                    {
                        best     = elapsed;
                        selected = sol;
                    }
                }
            }
        });

        if(selected.Succeeded())
        {
            const std::string algorithm_name = "miopenConvolutionFwdAlgoStaticCompiledGEMM";
            AddKernels(handle, algorithm_name, network_config, selected, nullptr);

            MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                      << selected.workspce_sz);
            record.SetValues(algorithm_name,
                             FindDbData{selected.solver_id,
                                        best,
                                        selected.workspce_sz,
                                        {algorithm_name, network_config}});
        }
    }
}

void ConvolutionDescriptor::FindConvFwdAlgorithm(Handle& handle,
                                                 const TensorDescriptor& xDesc,
                                                 ConstData_t x,
                                                 const TensorDescriptor& wDesc,
                                                 ConstData_t w,
                                                 const TensorDescriptor& yDesc,
                                                 Data_t y,
                                                 const int requestAlgoCount,
                                                 int* const returnedAlgoCount,
                                                 miopenConvAlgoPerf_t* perfResults,
                                                 Data_t workSpace,
                                                 size_t workSpaceSize,
                                                 bool exhaustiveSearch) const
{
    MIOPEN_LOG_I("requestAlgoCount = " << requestAlgoCount << ", workspace = " << workSpaceSize);
    if(x == nullptr || w == nullptr || y == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");

    *returnedAlgoCount = 0;

    const ProblemDescription problem(xDesc, wDesc, yDesc, *this, conv::Direction::Forward);

    std::vector<PerfField> perf_db =
        UserFindDbRecord::TryLoad(handle, problem, [&](DbRecord& record) {
            DirConvFindCore(handle,
                            xDesc,
                            x,
                            wDesc,
                            w,
                            yDesc,
                            y,
                            workSpace,
                            workSpaceSize,
                            *this,
                            exhaustiveSearch,
                            record);
        });

    if(perf_db.empty())
        MIOPEN_THROW("Fwd Convolution cannot be executed due to incorrect params");

    std::sort(begin(perf_db), end(perf_db));

    for(const auto& entry : perf_db)
        MIOPEN_LOG_I(entry.name << "\t" << entry.time << "\t" << entry.workspace);

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(perf_db.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].fwd_algo = StringToConvolutionFwdAlgo(perf_db[i].name);
        perfResults[i].time     = perf_db[i].time;
        perfResults[i].memory   = perf_db[i].workspace;
    }

    MIOPEN_LOG_I("FW Chosen Algorithm: " << perf_db[0].solver_id << " , " << perf_db[0].workspace
                                         << ", "
                                         << perf_db[0].time);
}

void ValidateConvTensors(const ConvTensors& tensors)
{
    const auto invalid_buffers =
        tensors.x == nullptr || tensors.w == nullptr || tensors.y == nullptr;

    const auto tensor_sizes_not_matched = tensors.xDesc.GetSize() != tensors.yDesc.GetSize() ||
                                          tensors.xDesc.GetSize() != tensors.wDesc.GetSize();

    const auto tensor_types_not_matched =
        (tensors.xDesc.GetType() != tensors.yDesc.GetType() &&
         tensors.xDesc.GetType() != miopenInt8 && tensors.xDesc.GetType() != miopenInt8x4) ||
        tensors.xDesc.GetType() != tensors.wDesc.GetType();

    // if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
    //    MIOPEN_THROW(miopenStatusBadParm);
    //}

    const auto x_tensor_invalid = tensors.xDesc.GetSize() < 3;

    const auto bad_parameters =
        invalid_buffers || tensor_sizes_not_matched || tensor_types_not_matched || x_tensor_invalid;

    if(bad_parameters)
        MIOPEN_THROW(miopenStatusBadParm);
}

void ValidateAlphaBeta(const void* alpha, const void* beta)
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Only alpha=1 and beta=0 is supported");
    }
}

static void ConvForwardCheckNumerics(Handle& handle,
                                     const ConvFwdTensors& tensors,
                                     std::function<void()>&& worker)
{
    if(!miopen::CheckNumericsEnabled())
    {
        worker();
        return;
    }

    miopen::checkNumericsInput(handle, tensors.xDesc, tensors.x);
    miopen::checkNumericsInput(handle, tensors.wDesc, tensors.w);

    worker();

    miopen::checkNumericsOutput(handle, tensors.yDesc, tensors.y);
}

template <class TKernels>
void ConvFwdImplicitGemm(const ConvolutionContext& ctx,
                         Handle& handle,
                         const ConvFwdTensors& tensors,
                         Data_t workSpace,
                         std::size_t workSpaceSize,
                         const TKernels& kernels);

template <class TKernels>
void ConvFwdSCGemm(const ConvolutionContext& ctx,
                   Handle& handle,
                   const ConvFwdTensors& tensors,
                   Data_t workSpace,
                   std::size_t workSpaceSize,
                   const TKernels& kernels);

void ConvolutionDescriptor::ConvolutionForward(Handle& handle,
                                               const void* alpha,
                                               const TensorDescriptor& xDesc,
                                               ConstData_t x,
                                               const TensorDescriptor& wDesc,
                                               ConstData_t w,
                                               miopenConvFwdAlgorithm_t algo,
                                               const void* beta,
                                               const TensorDescriptor& yDesc,
                                               Data_t y,
                                               Data_t workSpace,
                                               size_t workSpaceSize) const
{
    MIOPEN_LOG_I("algo = " << algo << ", workspace = " << workSpaceSize);
    const auto tensors = ConvFwdTensors{xDesc, x, wDesc, w, yDesc, y};
    ValidateConvTensors(tensors);
    ValidateAlphaBeta(alpha, beta);

    if(algo != miopenConvolutionFwdAlgoGEMM &&
       (xDesc.GetType() == miopenInt8 || xDesc.GetType() == miopenInt8x4))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    ConvForwardCheckNumerics(handle, tensors, [&]() {
        ValidateGroupCount(xDesc, wDesc, *this);

        const auto algorithm_name = AlgorithmName{ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(algo), conv::Direction::Forward)};

        auto ctx =
            ConvolutionContext{xDesc, wDesc, yDesc, *this, conv::Direction::Forward}; // forward
        ctx.SetStream(&handle);
        const auto network_config = ctx.BuildConfKey();
        const auto& invoker       = handle.GetInvoker(network_config, boost::none, algorithm_name);

        if(invoker)
        {
            const auto& invoke_ctx = conv::DataInvokeParams{tensors, workSpace, workSpaceSize};
            (*invoker)(handle, invoke_ctx);
            return;
        }

        switch(algo)
        {
        case miopenConvolutionFwdAlgoDirect:
        case miopenConvolutionFwdAlgoWinograd:
            MIOPEN_THROW("No invoker was registered for convolution forward. Was find executed?");

        case miopenConvolutionFwdAlgoImplicitGEMM:
        {
            auto&& kernels = handle.GetKernels(algorithm_name, network_config);
            ConvFwdImplicitGemm(ctx, handle, tensors, workSpace, workSpaceSize, kernels);
        }
        break;

        case miopenConvolutionFwdAlgoGEMM:
            ConvFwdGemm(handle, tensors, workSpace, workSpaceSize);
            break;

        case miopenConvolutionFwdAlgoFFT:
            ConvFwdFFT(handle, tensors, workSpace, workSpaceSize, network_config);
            break;
        case miopenConvolutionFwdAlgoStaticCompiledGEMM:
        {
            auto&& kernels = handle.GetKernels(algorithm_name, network_config);
            ConvFwdSCGemm(ctx, handle, tensors, workSpace, workSpaceSize, kernels);
        }
        break;
        }
    });
}

template <class TKernels>
void ConvFwdImplicitGemm(const ConvolutionContext& /*ctx*/,
                         Handle& handle,
                         const ConvFwdTensors& tensors,
                         Data_t /*workSpace*/,
                         std::size_t /*workSpaceSize*/,
                         const TKernels& kernels)
{
    if(kernels.empty())
        MIOPEN_THROW(
            "Error running Implicit GEMM Forward convolution. Was Find() executed previously?");

    auto kernel = kernels[0];

    float elapsed = 0;

    // clang-format off
    if(kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r1_gnchw_gkcyx_gnkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw" ||
       kernel.GetName() ==
           "gridwise_convolution_implicit_gemm_v4r4_ncdhw_kczyx_nkdhw")
    // clang-format on
    {
        kernel(tensors.x, tensors.w, tensors.y);

        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();
    }
    else
    {
        MIOPEN_THROW("Error running Implicit GEMM Forward convolution (none workspace?)");
    }
    if(handle.IsProfilingEnabled())
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed);
    }
}

void ConvolutionDescriptor::ConvFwdGemm(Handle& handle,
                                        const ConvFwdTensors& tensors,
                                        Data_t workSpace,
                                        std::size_t workSpaceSize) const
{
#if MIOPEN_USE_GEMM
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
    {
        MIOPEN_THROW("GEMM convolution is disabled");
    }
    if(IsAnyBufferBF16(tensors.xDesc, tensors.yDesc, tensors.wDesc) && !IsUseRocBlas)
    {
        MIOPEN_THROW("GEMM convolution is unsupported");
    }

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(tensors.xDesc.GetLengths());

    std::size_t wei_k = tensors.wDesc.GetLengths()[0];

    std::size_t spatial_dim = GetSpatialDimension();

    auto in_spatial  = boost::adaptors::slice(tensors.xDesc.GetLengths(), 2, 2 + spatial_dim);
    auto wei_spatial = boost::adaptors::slice(tensors.wDesc.GetLengths(), 2, 2 + spatial_dim);
    auto out_spatial = boost::adaptors::slice(tensors.yDesc.GetLengths(), 2, 2 + spatial_dim);

    // Use transpose path for 1x1, stride=2
    if(GetSpatialDimension() == 2 && miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; }))
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, 1x1 u2xv2");
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, 1x1 u2xv2");
        }

        assert(workSpace != nullptr &&
               workSpaceSize >= ForwardGetWorkSpaceSizeGEMMTranspose(tensors.xDesc, tensors.yDesc));

        float t1 = 0;
        transpose_NCHW2CNHW(handle,
                            in_n,
                            in_c,
                            in_spatial[0],
                            in_spatial[1],
                            out_spatial[0],
                            out_spatial[1],
                            tensors.x,
                            workSpace,
                            0,
                            0,
                            GetConvStrides()[0],
                            GetConvStrides()[1],
                            tensors.xDesc.GetType());
        if(handle.IsProfilingEnabled())
            t1 = handle.GetKernelTime();

        std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        std::size_t x_t_size = in_n * in_c * out_spatial_size;

        std::size_t wksp_offset = 0;
        if(tensors.wDesc.GetType() == miopenInt8)
        {
            wksp_offset = x_t_size;

            transpose_packed_MN2NM(handle,
                                   in_c,
                                   static_cast<int>(in_n * out_spatial_size),
                                   0,
                                   wksp_offset,
                                   workSpace,
                                   workSpace,
                                   tensors.xDesc.GetType());
            if(handle.IsProfilingEnabled())
                t1 += handle.GetKernelTime();

            x_t_size *= 2;
        }

        if(tensors.wDesc.GetType() == miopenInt8 || tensors.wDesc.GetType() == miopenInt8x4)
        {
            const auto xts = GetTypeSize(tensors.xDesc.GetType());
            if(xts > 0)
            {
                const auto yts_div_xts = GetTypeSize(tensors.yDesc.GetType()) / xts;
                if(yts_div_xts > 0)
                    x_t_size /= yts_div_xts;
            }
        }

        if(group_count > 1)
        {
            GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvCNHWFwd(
                tensors.wDesc, tensors.xDesc, tensors.yDesc, group_count);

            CallGemmStridedBatched(
                handle, gemm_desc, tensors.w, 0, workSpace, 0, workSpace, x_t_size, nullptr, false);
        }
        else
        {
            // tensors.y = CNHW2NCHW(tensors.w * NCHW2CNHW(tensors.x))
            GemmDescriptor gemm_desc =
                CreateGemmDescriptorConvCNHWFwd(tensors.wDesc, tensors.xDesc, tensors.yDesc);

            // tensors.y = CNHW2NCHW(tensors.w * NCHW2CNHW(tensors.x))
            CallGemm(handle,
                     gemm_desc,
                     tensors.w,
                     0,
                     workSpace,
                     wksp_offset,
                     workSpace,
                     x_t_size,
                     nullptr,
                     false);
        }
        if(handle.IsProfilingEnabled())
            t1 += handle.GetKernelTime();

        transpose_CNHW2NCHW(handle,
                            in_n,
                            wei_k,
                            out_spatial[0],
                            out_spatial[1],
                            out_spatial[0],
                            out_spatial[1],
                            workSpace,
                            tensors.y,
                            x_t_size,
                            0,
                            1,
                            1,
                            tensors.yDesc.GetType());
        if(handle.IsProfilingEnabled())
            t1 += handle.GetKernelTime();

        if((tensors.wDesc.GetType() == miopenInt8 || tensors.wDesc.GetType() == miopenInt8x4) &&
           tensors.yDesc.GetType() != miopenInt32)
        {
            TensorDescriptor ygemmDesc(
                miopenInt32, tensors.yDesc.GetLengths(), tensors.yDesc.GetStrides());

            CastTensor(handle, &lowp_quant, ygemmDesc, tensors.y, tensors.yDesc, tensors.y, 0, 0);
            if(handle.IsProfilingEnabled())
                t1 += handle.GetKernelTime();
        }

        if(handle.IsProfilingEnabled())
        {
            handle.ResetKernelTime();
            handle.AccumKernelTime(t1);
        }
    }
    else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, 1x1");

            GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvFwd(
                tensors.wDesc, tensors.xDesc, tensors.yDesc, group_count);
            float time_0 = 0;

            std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                           out_spatial.end(),
                                                           std::size_t(1),
                                                           std::multiplies<std::size_t>());

            std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                          in_spatial.end(),
                                                          std::size_t(1),
                                                          std::multiplies<std::size_t>());

            for(std::size_t i = 0; i < in_n; i++)
            {
                std::size_t out_offset = i * wei_k * out_spatial_size;

                std::size_t in_offset = i * in_c * in_spatial_size;

                CallGemmStridedBatched(handle,
                                       gemm_desc,
                                       tensors.w,
                                       0,
                                       tensors.x,
                                       in_offset,
                                       tensors.y,
                                       out_offset,
                                       nullptr,
                                       false);
                if(handle.IsProfilingEnabled())
                {
                    if(i == in_n - 1)
                        handle.AccumKernelTime(time_0);
                    time_0 += handle.GetKernelTime();
                }
            }
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, 1x1");
            float time_0 = 0;
            float t1     = 0;

            if(tensors.wDesc.GetType() == miopenInt8)
            {
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorConvFwd(tensors.wDesc, tensors.xDesc, tensors.yDesc);

                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                              in_spatial.end(),
                                                              std::size_t(1),
                                                              std::multiplies<std::size_t>());

                for(std::size_t i = 0; i < in_n; i++)
                {
                    std::size_t out_offset = i * wei_k * out_spatial_size;

                    std::size_t in_offset = i * in_c * in_spatial_size;

                    transpose_packed_MN2NM(handle,
                                           in_c,
                                           in_spatial_size,
                                           in_offset,
                                           0,
                                           tensors.x,
                                           workSpace,
                                           tensors.xDesc.GetType());
                    if(handle.IsProfilingEnabled())
                        t1 += handle.GetKernelTime();

                    CallGemm(handle,
                             gemm_desc,
                             tensors.w,
                             0,
                             workSpace,
                             0,
                             tensors.y,
                             out_offset,
                             nullptr,
                             false);
                    if(handle.IsProfilingEnabled())
                        time_0 += handle.GetKernelTime();
                }
            }
            else
            {
                // tensors.y = tensors.w * tensors.x
                GemmDescriptor gemm_desc = CreateGemmStridedBatchedDescriptorConv1x1Fwd(
                    tensors.wDesc, tensors.xDesc, tensors.yDesc);

                // tensors.y = tensors.w * tensors.x
                CallGemmStridedBatched(
                    handle, gemm_desc, tensors.w, 0, tensors.x, 0, tensors.y, 0, nullptr, false);
                if(handle.IsProfilingEnabled())
                    time_0 += handle.GetKernelTime();
            }

            if((tensors.wDesc.GetType() == miopenInt8 || tensors.wDesc.GetType() == miopenInt8x4) &&
               tensors.yDesc.GetType() != miopenInt32)
            {
                TensorDescriptor ygemmDesc(
                    miopenInt32, tensors.yDesc.GetLengths(), tensors.yDesc.GetStrides());

                CastTensor(
                    handle, &lowp_quant, ygemmDesc, tensors.y, tensors.yDesc, tensors.y, 0, 0);
                if(handle.IsProfilingEnabled())
                    handle.AccumKernelTime(t1 + time_0);
            }
        }
    }
    // if not 1x1
    else
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, non 1x1");
        }
        assert(workSpace != nullptr &&
               workSpaceSize >=
                   (ForwardGetWorkSpaceSizeGEMM(tensors.wDesc, tensors.yDesc) * group_count));

        // tensors.y = tensors.w * Im2Col(tensors.x)
        GemmDescriptor gemm_desc{};
        if(group_count > 1)
            gemm_desc = CreateGemmDescriptorGroupConvFwd(
                tensors.wDesc, tensors.xDesc, tensors.yDesc, group_count);
        else
            gemm_desc = CreateGemmDescriptorConvFwd(tensors.wDesc, tensors.xDesc, tensors.yDesc);

        std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        std::size_t in_spatial_size = std::accumulate(
            in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        float time_0 = 0;
        float t1     = 0;
        for(std::size_t i = 0; i < in_n; i++)
        {
            std::size_t out_offset = i * wei_k * out_spatial_size;

            std::size_t in_offset = i * in_c * in_spatial_size;

            Im2ColGPU(handle,
                      GetSpatialDimension(),
                      tensors.x,
                      in_offset,
                      in_c,
                      in_spatial,
                      wei_spatial,
                      out_spatial,
                      GetConvPads(),
                      GetConvStrides(),
                      GetConvDilations(),
                      workSpace,
                      tensors.xDesc.GetType());

            if(handle.IsProfilingEnabled())
                t1 = handle.GetKernelTime();

            std::size_t wksp_offset = 0;
            if(tensors.wDesc.GetType() == miopenInt8)
            {
                std::size_t wei_spatial_size = std::accumulate(wei_spatial.begin(),
                                                               wei_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                wksp_offset = in_c * wei_spatial_size * out_spatial_size;

                transpose_packed_MN2NM(handle,
                                       static_cast<int>(in_c * wei_spatial_size),
                                       out_spatial_size,
                                       0,
                                       wksp_offset,
                                       workSpace,
                                       workSpace,
                                       tensors.xDesc.GetType());

                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();
            }

            // tensors.y = tensors.w * Im2Col(tensors.x)
            if(group_count > 1)
                CallGemmStridedBatched(handle,
                                       gemm_desc,
                                       tensors.w,
                                       0,
                                       workSpace,
                                       0,
                                       tensors.y,
                                       out_offset,
                                       nullptr,
                                       false);
            else
                CallGemm(handle,
                         gemm_desc,
                         tensors.w,
                         0,
                         workSpace,
                         wksp_offset,
                         tensors.y,
                         out_offset,
                         nullptr,
                         false,
                         (tensors.wDesc.GetType() == miopenInt8 ||
                          tensors.wDesc.GetType() == miopenInt8x4)
                             ? GemmBackend_t::rocblas
                             : GemmBackend_t::miopengemm);

            // Update times for both the kernels
            if(handle.IsProfilingEnabled())
            {
                if(i == in_n - 1)
                {
                    handle.AccumKernelTime(t1 + time_0);
                    time_0 = handle.GetKernelTime();
                }
                else
                {
                    handle.AccumKernelTime(t1);
                    time_0 += handle.GetKernelTime();
                }
            }
        }

        if((tensors.wDesc.GetType() == miopenInt8 || tensors.wDesc.GetType() == miopenInt8x4) &&
           tensors.yDesc.GetType() != miopenInt32)
        {
            TensorDescriptor ygemmDesc(
                miopenInt32, tensors.yDesc.GetLengths(), tensors.yDesc.GetStrides());

            CastTensor(handle, &lowp_quant, ygemmDesc, tensors.y, tensors.yDesc, tensors.y, 0, 0);
            if(handle.IsProfilingEnabled())
                handle.AccumKernelTime(time_0);
        }
    }
#ifdef NDEBUG
    (void)workSpaceSize;
#endif
#else
    (void)handle;
    (void)tensors;
    (void)workSpace;
    (void)workSpaceSize;
    MIOPEN_THROW("GEMM is not supported");
#endif
}

void ConvolutionDescriptor::ConvFwdFFT(Handle& handle,
                                       const ConvFwdTensors& tensors,
                                       Data_t workSpace,
                                       std::size_t workSpaceSize,
                                       const NetworkConfig& kcache_key) const
{
    if(group_count > 1)
        MIOPEN_THROW("FFT is not supported for group conv");

    assert(workSpaceSize >=
           ForwardGetWorkSpaceSizeFFT(tensors.wDesc, tensors.xDesc, tensors.yDesc));

    if(workSpace == nullptr || workSpaceSize == 0)
        MIOPEN_THROW("Error running FFT: none workspace");

    bool timed  = handle.IsProfilingEnabled();
    float timev = ExecuteFwdFFTKernel(handle,
                                      tensors.xDesc,
                                      tensors.x,
                                      tensors.wDesc,
                                      tensors.w,
                                      tensors.yDesc,
                                      tensors.y,
                                      workSpace,
                                      workSpaceSize,
                                      kcache_key,
                                      timed);
    if(timed)
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(timev);
    }
}

template <class TKernels>
void ConvFwdSCGemm(const ConvolutionContext& ctx,
                   Handle& handle,
                   const ConvFwdTensors& tensors,
                   Data_t workSpace,
                   std::size_t workSpaceSize,
                   const TKernels& kernels)
{
#if MIOPEN_USE_SCGEMM
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_SCGEMM{}))
    {
        MIOPEN_THROW("Static Compiled GEMM is disabled");
    }

    if(kernels.empty() /*|| scgParams.params == nullptr*/)
        MIOPEN_THROW(
            "Error running Static Compiled GEMM convolution. Was Find() executed previously?");

    auto ks = std::vector<KernelInvoke>{kernels.begin(), kernels.end()};

    float elapsed = 0;

    elapsed = CallSCGemm(handle, ctx, tensors.x, tensors.y, tensors.w, nullptr, workSpace, ks);

    if(handle.IsProfilingEnabled())
    {
        MIOPEN_LOG_I2("CallSCGemm elapsed time = " << elapsed << " ms");
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed);
    }
    std::ignore = workSpaceSize;
#else
    std::ignore = ctx;
    std::ignore = handle;
    std::ignore = tensors;
    std::ignore = workSpace;
    std::ignore = workSpaceSize;
    std::ignore = kernels;
    MIOPEN_THROW("Static Compiled GEMM is not supported");
#endif
}

std::size_t ConvolutionDescriptor::GetFwdSolutionCountFallback(const TensorDescriptor& wDesc,
                                                               const TensorDescriptor& xDesc,
                                                               const TensorDescriptor& yDesc) const
{
    // This is needed on fallback path only.
    // Regular (find-db) path have been verified during Find().
    ValidateGroupCount(xDesc, wDesc, *this);

    if(IsGemmApplicableFwd(wDesc, xDesc, yDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        return 1;
    }
    MIOPEN_LOG_I("Fallback path, GEMM disabled");
    /// When count=0 the reason could be:
    /// * (1) Convolution is not implemented in the library at all, so Find() would fail as
    ///   well. This is case when rc = miopenStatusNotImplemented is correct.
    /// * (2) Variant of the above: Convolution is implemented, but implementation is disabled,
    ///   for example, rocBLAS is not installed or some convolutions are disabled by the
    ///   environment setting.
    /// * (3) There is none relevant record in the find-db and fallback path was unable to
    ///   choose suitable solution.
    ///
    /// We can't distinguish these three cases.
    /// Let's do like Find() does:
    MIOPEN_THROW(miopenStatusNotImplemented,
                 "Requested convolution is not supported or immedate mode fallback has failed.");
}

std::size_t ConvolutionDescriptor::GetBwdSolutionCountFallback(const TensorDescriptor& dyDesc,
                                                               const TensorDescriptor& wDesc,
                                                               const TensorDescriptor& dxDesc) const
{
    ValidateGroupCount(dxDesc, wDesc, *this); // See comment in Forward method.

    if(IsGemmApplicableBwd(dyDesc, wDesc, dxDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        return 1;
    }
    MIOPEN_LOG_I("Fallback path, GEMM disabled");
    // See comment in Forward method.
    MIOPEN_THROW(miopenStatusNotImplemented,
                 "Requested convolution is not supported or immedate mode fallback has failed.");
}

bool ConvolutionDescriptor::IsGemmApplicableWrw(const TensorDescriptor& dyDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& dwDesc) const
{
#if MIOPEN_USE_GEMM
    if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}) &&
       !(IsAnyBufferBF16(xDesc, dyDesc, dwDesc) && !IsUseRocBlas))
    {
        const std::size_t spatial_dim = GetSpatialDimension();
        const auto wei_spatial = boost::adaptors::slice(dwDesc.GetLengths(), 2, 2 + spatial_dim);

        // if not 1x1
        if((miopen::any_of(wei_spatial, [](auto v) { return v != 1; }) ||
            miopen::any_of(GetConvPads(), [](auto v) { return v != 0; }) ||
            miopen::any_of(GetConvStrides(), [](auto v) { return v != 1; })))
            return true;

        if(miopen::any_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::any_of(GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::any_of(GetConvStrides(), [](auto v) { return v == 1; }))
            return true;

        return false;
    }
#else
    std::ignore = dyDesc;
    std::ignore = xDesc;
    std::ignore = dwDesc;
#endif
    return false;
}

bool ConvolutionDescriptor::IsGemmApplicableFwd(const TensorDescriptor& wDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& yDesc) const
{
#if MIOPEN_USE_GEMM
    return !miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}) &&
           !(IsAnyBufferBF16(xDesc, yDesc, wDesc) && !IsUseRocBlas);
#else
    std::ignore = wDesc;
    std::ignore = xDesc;
    std::ignore = yDesc;
    return false;
#endif
}

bool ConvolutionDescriptor::IsGemmApplicableBwd(const TensorDescriptor& dyDesc,
                                                const TensorDescriptor& wDesc,
                                                const TensorDescriptor& dxDesc) const
{
#if MIOPEN_USE_GEMM
    return !miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}) &&
           !(IsAnyBufferBF16(dxDesc, dyDesc, wDesc) && !IsUseRocBlas);
#else
    std::ignore = dyDesc;
    std::ignore = wDesc;
    std::ignore = dxDesc;
    return false;
#endif
}

std::size_t ConvolutionDescriptor::GetWrwSolutionCountFallback(const TensorDescriptor& dyDesc,
                                                               const TensorDescriptor& xDesc,
                                                               const TensorDescriptor& dwDesc) const
{
    ValidateGroupCount(xDesc, dwDesc, *this); // See comment in Forward method.

    if(IsGemmApplicableWrw(xDesc, dyDesc, dwDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        return 1;
    }
    MIOPEN_LOG_I("Fallback path, GEMM disabled");
    // See comment in Forward method.
    MIOPEN_THROW(miopenStatusNotImplemented,
                 "Requested convolution is not supported or immedate mode fallback has failed.");
}

std::size_t GetSolutionCount(Handle& handle, const ProblemDescription& problem)
{
    const FindDbRecord fdb_record{handle, problem};
    if(fdb_record.empty())
        return 0;
    return std::distance(fdb_record.begin(), fdb_record.end());
}

std::size_t ConvolutionDescriptor::GetForwardSolutionCount(Handle& handle,
                                                           const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& xDesc,
                                                           const TensorDescriptor& yDesc) const
{
    MIOPEN_LOG_I("");
    const auto problem = ProblemDescription{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
    const auto n       = GetSolutionCount(handle, problem);
    if(n > 0)
        return n;
    return GetFwdSolutionCountFallback(wDesc, xDesc, yDesc);
}

static inline bool IsAlgorithmDisabled(const miopenConvAlgorithm_t algo)
{
    switch(algo)
    { // clang-format off
    case miopenConvolutionAlgoGEMM:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}) || !MIOPEN_USE_GEMM;
    case miopenConvolutionAlgoDirect:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{});
    case miopenConvolutionAlgoFFT:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_FFT{});
    case miopenConvolutionAlgoWinograd:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{});
    case miopenConvolutionAlgoImplicitGEMM:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{});
    case miopenConvolutionAlgoStaticCompiledGEMM:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_SCGEMM{}) || !MIOPEN_USE_SCGEMM;
    default: // Disable future algos by default to enforce explicit handling:
        return true;
    } // clang-format on
}

void GetSolutions(Handle& handle,
                  const ProblemDescription& problem,
                  const size_t maxSolutionCount,
                  size_t* solutionCount,
                  miopenConvSolution_t* solutions,
                  std::function<int(const std::string&)>&& algoResolver)
{
    const FindDbRecord fdb_record{handle, problem};

    if(fdb_record.empty())
    {
        *solutionCount = 0;
        return;
    }

    // Read all what we have, then sort and write out up to max asked.
    // Fallback path currently returns only one solution, so no need to sort there.
    struct SortWrapper : miopenConvSolution_t // For emplace and sort.
    {
        SortWrapper(const float& t,
                    const size_t& ws,
                    const uint64_t& id,
                    const miopenConvAlgorithm_t& algo)
            : miopenConvSolution_t{t, ws, id, algo}
        {
        }
        bool operator<(const SortWrapper& other) const { return (time < other.time); }
    };
    std::vector<SortWrapper> interim;
    interim.reserve(maxSolutionCount); // For speed. In most cases we have less entries than asked.

    // Individual Solvers can be enabled/disabled by environment settings.
    // Applicability is also affected by presence of external tools (e.g. assembler)
    // ROCm version, specific features of GPU (like xnack) etc.
    // All the above can be found by calling IsApplicable().
    // We need fully initialized context for this, see below.
    auto ctx = ConvolutionContext{problem};
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    for(const auto& pair : fdb_record)
    {
        const auto algo = static_cast<miopenConvAlgorithm_t>(algoResolver(pair.first));
        if(IsAlgorithmDisabled(algo))
            continue;

        const auto solver_id = solver::Id{pair.second.solver_id};
        // Wrong IDs can't be used to call IsApplicable(), so let's
        // ignore obsolete or invalid IDs read from find-db first.
        if(!solver_id.IsValid())
        {
            // Do not disturb users with warnings unless detailed log is enabled.
            MIOPEN_LOG_I("[Warning] incorrect solver_id: " << pair.second.solver_id);
            continue;
        }
        // gemm and fft are always applicable.
        // These can be disabled/enabled at algorithm level.
        if(!(solver_id == solver::Id::gemm() || solver_id == solver::Id::fft()))
            if(!solver_id.GetSolver().IsApplicable(ctx))
                continue;

        interim.emplace_back(pair.second.time, pair.second.workspace, solver_id.Value(), algo);
    }
    std::sort(begin(interim), end(interim));

    auto i = std::size_t{0};
    for(const auto& entry : interim)
    {
        if(i >= maxSolutionCount)
            break;
        solutions[i] = entry;
        ++i;
    }
    *solutionCount = i;
}

void ConvolutionDescriptor::GetForwardSolutionsFallback(Handle& handle,
                                                        const TensorDescriptor& wDesc,
                                                        const TensorDescriptor& xDesc,
                                                        const TensorDescriptor& yDesc,
                                                        const size_t maxSolutionCount,
                                                        size_t* const solutionCount,
                                                        miopenConvSolution_t* const solutions) const
{
    // This check is needed on fallback path only.
    // Regular (find-db) path have been verified during Find().
    ValidateGroupCount(xDesc, wDesc, *this);
    auto i = std::size_t{0};

    if(IsGemmApplicableFwd(wDesc, xDesc, yDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        if(i < maxSolutionCount)
        {
            solutions[i].algorithm = miopenConvolutionAlgoGEMM;
            solutions[i].time      = -1.0; /// \todo Evaluate time.
            solutions[i].workspace_size =
                ForwardGetValidWorkSpaceSizeGemm(handle, wDesc, xDesc, yDesc);
            solutions[i].solution_id = solver::Id::gemm().Value();
            ++i;
        }
    }
    else
        MIOPEN_LOG_I("Fallback path, GEMM disabled");

    *solutionCount = i;
}

void ConvolutionDescriptor::GetBwdSolutionsFallback(Handle& /*handle*/,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dxDesc,
                                                    const size_t maxSolutionCount,
                                                    size_t* const solutionCount,
                                                    miopenConvSolution_t* const solutions) const
{
    ValidateGroupCount(dxDesc, wDesc, *this);
    auto i = std::size_t{0};

    if(IsGemmApplicableBwd(dyDesc, wDesc, dxDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        if(i < maxSolutionCount)
        {
            solutions[i].algorithm      = miopenConvolutionAlgoGEMM;
            solutions[i].time           = -1.0; /// \todo Evaluate time.
            solutions[i].workspace_size = BackwardGetValidWorkSpaceSizeGemm(dyDesc, wDesc, dxDesc);
            solutions[i].solution_id    = solver::Id::gemm().Value();
            ++i;
        }
    }
    else
        MIOPEN_LOG_I("Fallback path, GEMM disabled");

    *solutionCount = i;
}

void ConvolutionDescriptor::GetWrwSolutionsFallback(Handle& /*handle*/,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& xDesc,
                                                    const TensorDescriptor& dwDesc,
                                                    const size_t maxSolutionCount,
                                                    size_t* const solutionCount,
                                                    miopenConvSolution_t* const solutions) const
{
    ValidateGroupCount(xDesc, dwDesc, *this);
    auto i = std::size_t{0};

    if(IsGemmApplicableWrw(dyDesc, xDesc, dwDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        if(i < maxSolutionCount)
        {
            solutions[i].algorithm      = miopenConvolutionAlgoGEMM;
            solutions[i].time           = -1.0; /// \todo Evaluate time.
            solutions[i].workspace_size = WrwGetValidWorkSpaceSizeGemm(dyDesc, xDesc, dwDesc);
            solutions[i].solution_id    = solver::Id::gemm().Value();
            ++i;
        }
    }
    else
        MIOPEN_LOG_I("Fallback path, GEMM disabled");

    *solutionCount = i;
}

void ConvolutionDescriptor::GetForwardSolutions(Handle& handle,
                                                const TensorDescriptor& wDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& yDesc,
                                                const size_t maxSolutionCount,
                                                size_t* const solutionCount,
                                                miopenConvSolution_t* const solutions) const
{
    MIOPEN_LOG_I("");
    if(solutionCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutionCount cannot be nullptr");
    if(solutions == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutions cannot be nullptr");

    const auto problem = ProblemDescription{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
    GetSolutions(
        handle, problem, maxSolutionCount, solutionCount, solutions, StringToConvolutionFwdAlgo);

    if(*solutionCount == 0)
        GetForwardSolutionsFallback(
            handle, wDesc, xDesc, yDesc, maxSolutionCount, solutionCount, solutions);
}

std::size_t
ConvolutionDescriptor::GetFwdSolutionWorkspaceSizeFallback(Handle& handle,
                                                           const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& xDesc,
                                                           const TensorDescriptor& yDesc,
                                                           solver::Id solver_id) const
{
    ValidateGroupCount(xDesc, wDesc, *this);
    if(solver_id == solver::Id::gemm() && IsGemmApplicableFwd(wDesc, xDesc, yDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        return ForwardGetValidWorkSpaceSizeGemm(handle, wDesc, xDesc, yDesc);
    }
    MIOPEN_THROW(miopenStatusNotImplemented);
}

std::size_t
ConvolutionDescriptor::BackwardGetValidWorkSpaceSizeGemm(const TensorDescriptor& dyDesc,
                                                         const TensorDescriptor& wDesc,
                                                         const TensorDescriptor& dxDesc) const
{
    const auto wei_spatial =
        boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + GetSpatialDimension());

    if(GetSpatialDimension() == 2 && miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; }))
        return BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc);

    if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
        return 0;

    return BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count;
}

std::size_t
ConvolutionDescriptor::GetBwdSolutionWorkspaceSizeFallback(const TensorDescriptor& dyDesc,
                                                           const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& dxDesc,
                                                           solver::Id solver_id) const
{
    ValidateGroupCount(dxDesc, wDesc, *this);
    if(solver_id == solver::Id::gemm() && IsGemmApplicableBwd(dyDesc, wDesc, dxDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        return BackwardGetValidWorkSpaceSizeGemm(dyDesc, wDesc, dxDesc);
    }
    MIOPEN_THROW(miopenStatusNotImplemented);
}

std::size_t
ConvolutionDescriptor::GetWrwSolutionWorkspaceSizeFallback(Handle& /*handle*/,
                                                           const TensorDescriptor& dyDesc,
                                                           const TensorDescriptor& xDesc,
                                                           const TensorDescriptor& dwDesc,
                                                           solver::Id solver_id) const
{
    ValidateGroupCount(xDesc, dwDesc, *this);
    if(solver_id == solver::Id::gemm() && IsGemmApplicableWrw(dyDesc, xDesc, dwDesc))
    {
        MIOPEN_LOG_I("Fallback path, GEMM");
        return WrwGetValidWorkSpaceSizeGemm(dyDesc, xDesc, dwDesc);
    }
    MIOPEN_THROW(miopenStatusNotImplemented);
}

std::size_t ConvolutionDescriptor::GetForwardSolutionWorkspaceSize(Handle& handle,
                                                                   const TensorDescriptor& wDesc,
                                                                   const TensorDescriptor& xDesc,
                                                                   const TensorDescriptor& yDesc,
                                                                   solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString());
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm, "invalid solution id = " + solver_id.ToString());
    if(solver_id != solver::Id::gemm() && solver_id != solver::Id::fft())
    {
        auto sol = solver_id.GetSolver();
        auto ctx = ConvolutionContext{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        if(sol.IsApplicable(ctx))
            return sol.GetWorkspaceSize(ctx);
        else
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "The supplied solution id: " + solver_id.ToString() +
                             " is not applicable to the current problem");
        }
    }
    else if(solver_id == solver::Id::fft())
        return ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc);
    // handles the GEMM case
    return GetFwdSolutionWorkspaceSizeFallback(handle, wDesc, xDesc, yDesc, solver_id);
}

// Todo: remove when all immediate mode calls will support invokers
static std::vector<KernelInvoke> CompileSolver(Handle& handle,
                                               ConvolutionContext& ctx,
                                               solver::Id solver_id,
                                               const FindDbKCacheKey& key)
{
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto solver   = solver_id.GetSolver();
    auto db             = GetDb(ctx);
    const auto solution = solver.FindSolution(ctx, db);

    std::vector<KernelInvoke> kernels;
    AddKernels(handle, key.algorithm_name, key.network_config, solution, &kernels);
    return kernels;
}

static Invoker PrepareInvoker(Handle& handle,
                              ConvolutionContext& ctx,
                              const NetworkConfig& config,
                              solver::Id solver_id)
{
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto solver = solver_id.GetSolver();
    auto db           = GetDb(ctx);
    auto solution     = solver.FindSolution(ctx, db);
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);

    handle.RegisterInvoker(invoker, config, solver_id);
    return invoker;
}

static Invoker LoadOrPrepareInvoker(Handle& handle, ConvolutionContext& ctx, solver::Id solver_id)
{
    const auto config = ctx.BuildConfKey();
    auto invoker      = handle.GetInvoker(config, solver_id);
    if(invoker)
        return *invoker;
    return PrepareInvoker(handle, ctx, config, solver_id);
}

static bool CheckInvokerSupport(const solver::Id solver_id, conv::Direction dir)
{
    const auto& algo = solver_id.GetAlgo(dir);
    return CheckInvokerSupport(algo);
}

static void CompileSolution(Handle& handle,
                            const solver::Id solver_id,
                            ConvolutionContext& ctx,
                            conv::Direction dir,
                            std::function<void()>&& fft_finder)
{
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm, "solver_id = " + solver_id.ToString());

    if(CheckInvokerSupport(solver_id, dir))
    {
        LoadOrPrepareInvoker(handle, ctx, solver_id);
        return;
    }

    // Todo: remove when all finds will use invokers.
    if(solver_id == solver::Id::gemm())
    {
        // Todo: gemm precompilation?
        return;
    }

    const FindDbRecord fdb_record{handle, ctx};
    for(const auto& pair : fdb_record)
    {
        if(solver::Id{pair.second.solver_id} != solver_id)
            continue;

        const auto&& kernels = handle.GetKernels(pair.second.kcache_key.algorithm_name,
                                                 pair.second.kcache_key.network_config);

        if(!kernels.empty())
            return;

        if(solver_id == solver::Id::fft())
        {
            fft_finder();
            return;
        }

        CompileSolver(handle, ctx, solver_id, pair.second.kcache_key);
        return;
    }

    // Todo: solver not found in find-db.
    MIOPEN_THROW(miopenStatusNotImplemented);
}

void ConvolutionDescriptor::CompileForwardSolution(Handle& handle,
                                                   const TensorDescriptor& wDesc,
                                                   const TensorDescriptor& xDesc,
                                                   const TensorDescriptor& yDesc,
                                                   const solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString());

    auto ctx = ConvolutionContext{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
    ctx.SetStream(&handle);
    ctx.disable_search_enforce = true;

    CompileSolution(handle, solver_id, ctx, conv::Direction::Forward, [&]() {
        const auto workspace_fft = ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc);
        std::vector<KernelInvoke> ignore0;
        const auto network_config = ctx.BuildConfKey();
        FindFwdFFTKernel(handle, xDesc, wDesc, yDesc, workspace_fft, ignore0, network_config);
    });
}

void ConvolutionDescriptor::ConvolutionForwardImmediate(Handle& handle,
                                                        const TensorDescriptor& wDesc,
                                                        ConstData_t w,
                                                        const TensorDescriptor& xDesc,
                                                        ConstData_t x,
                                                        const TensorDescriptor& yDesc,
                                                        Data_t y,
                                                        Data_t workSpace,
                                                        const std::size_t workSpaceSize,
                                                        const solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString() << ", workspace = " << workSpaceSize);
    const auto tensors = ConvFwdTensors{xDesc, x, wDesc, w, yDesc, y};

    ValidateConvTensors(tensors);
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm);

    ConvForwardCheckNumerics(handle, tensors, [&]() {
        auto ctx = ConvolutionContext{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
        ctx.SetStream(&handle);

        if(CheckInvokerSupport(solver_id, conv::Direction::Forward))
        {
            const auto invoker    = LoadOrPrepareInvoker(handle, ctx, solver_id);
            const auto invoke_ctx = conv::DataInvokeParams{tensors, workSpace, workSpaceSize};
            invoker(handle, invoke_ctx);
            return;
        }

        // Todo: remove when all algorithms would support invokers
        if(solver_id == solver::Id::gemm())
        {
            ConvFwdGemm(handle, tensors, workSpace, workSpaceSize);
            return;
        }

        const auto network_config = ctx.BuildConfKey();
        const auto algo_name      = solver_id.GetAlgo(conv::Direction::Forward);
        const auto&& chk_kernels  = handle.GetKernels(algo_name, network_config);
        auto v_chk_kernels = std::vector<KernelInvoke>{chk_kernels.begin(), chk_kernels.end()};

        if(!v_chk_kernels.empty())
        {
            MIOPEN_LOG_I2(
                "Found previously compiled kernels for solution: " << solver_id.ToString());
            if(solver_id == solver::Id::fft())
                ConvFwdFFT(handle, tensors, workSpace, workSpaceSize, network_config);
            else if(algo_name == "miopenConvolutionFwdAlgoImplicitGEMM")
                ConvFwdImplicitGemm(ctx, handle, tensors, workSpace, workSpaceSize, v_chk_kernels);
            else if(algo_name == "miopenConvolutionFwdAlgoStaticCompiledGEMM")
                ConvFwdSCGemm(ctx, handle, tensors, workSpace, workSpaceSize, v_chk_kernels);
            else
                MIOPEN_THROW("Invalid algorithm: " + algo_name);
            return;
        }

        const auto problem =
            ProblemDescription{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
        const FindDbRecord fdb_record{handle, problem};

        for(const auto& pair : fdb_record)
        {
            if(solver::Id{pair.second.solver_id} != solver_id)
                continue;

            const auto&& kernels = handle.GetKernels(pair.second.kcache_key.algorithm_name,
                                                     pair.second.kcache_key.network_config);
            auto v_kernels = std::vector<KernelInvoke>{kernels.begin(), kernels.end()};

            if(solver_id == solver::Id::fft())
            {
                if(v_kernels.empty())
                    FindFwdFFTKernel(
                        handle, xDesc, wDesc, yDesc, workSpaceSize, v_kernels, network_config);
                ConvFwdFFT(handle, tensors, workSpace, workSpaceSize, network_config);
                return;
            }

            if(v_kernels.empty())
                v_kernels = CompileSolver(handle, ctx, solver_id, pair.second.kcache_key);

            if(algo_name == "miopenConvolutionFwdAlgoImplicitGEMM")
                ConvFwdImplicitGemm(ctx, handle, tensors, workSpace, workSpaceSize, v_kernels);
            else if(algo_name == "miopenConvolutionFwdAlgoStaticCompiledGEMM")
                ConvFwdSCGemm(ctx, handle, tensors, workSpace, workSpaceSize, v_kernels);
            else
                MIOPEN_THROW("Invalid algorithm: " + pair.second.kcache_key.algorithm_name);
            return;
        }

        // Todo: solver not found in find-db.
        MIOPEN_THROW(miopenStatusNotImplemented);
    });
}

// FindBackwardDataAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdDataAlgorithm(Handle& handle,
                                                     const TensorDescriptor& dyDesc,
                                                     ConstData_t dy,
                                                     const TensorDescriptor& wDesc,
                                                     ConstData_t w,
                                                     const TensorDescriptor& dxDesc,
                                                     Data_t dx,
                                                     const int requestAlgoCount,
                                                     int* const returnedAlgoCount,
                                                     miopenConvAlgoPerf_t* perfResults,
                                                     Data_t workSpace,
                                                     size_t workSpaceSize,
                                                     bool exhaustiveSearch) const
{
    MIOPEN_LOG_I("requestAlgoCount = " << requestAlgoCount << ", workspace = " << workSpaceSize);
    if(dx == nullptr || w == nullptr || dy == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");
    if(wDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    *returnedAlgoCount = 0;

    AutoEnableProfiling enableProfiling{handle};

    const ProblemDescription problem(dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData);

    const auto use_winograd_only = [&]() {
        auto ctx = ConvolutionContext{problem};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        return IsWinograd3x3SupportedAndFast(ctx);
    }();

    // < algorith_name, <time, workspace_size> >
    std::vector<PerfField> perf_db =
        UserFindDbRecord::TryLoad(handle, problem, [&](DbRecord& record) {
            const auto network_config = problem.BuildConfKey();
            const auto invoke_ctx     = conv::DataInvokeParams{
                {dyDesc, dy, wDesc, w, dxDesc, dx}, workSpace, workSpaceSize};

            // Winograd algo
            {
                ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
                bufs.SetBwd(dx, w, dy);
                auto ctx = ConvolutionContext{problem};
                ctx.SetBufs(bufs);
                ctx.SetStream(&handle);
                ctx.DetectRocm();
                const auto all            = FindWinogradSolutions(ctx);
                const auto algorithm_name = AlgorithmName{"miopenConvolutionBwdDataAlgoWinograd"};
                PrecompileSolutions(handle, all);
                EvaluateInvokers(handle, all, algorithm_name, network_config, invoke_ctx, record);
            }

            // Direct algo
            if(!use_winograd_only)
            {
                ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
                bufs.SetBwd(dx, w, dy);
                const auto all = FindDataDirectSolutions(
                    handle, dxDesc, wDesc, dyDesc, exhaustiveSearch, false, bufs);
                const auto algorithm_name = AlgorithmName{"miopenConvolutionBwdDataAlgoDirect"};
                PrecompileSolutions(handle, all);
                EvaluateInvokers(handle, all, algorithm_name, network_config, invoke_ctx, record);
            }

            // Implicit GEMM algo
            if(!use_winograd_only)
            {
                ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
                bufs.SetBwd(dx, w, dy);
                const auto all = this->FindDataImplicitGemmSolutions(
                    handle, dxDesc, wDesc, dyDesc, exhaustiveSearch, false, bufs);
                PrecompileSolutions(handle, all);
                miopen::solver::ConvSolution selected{miopenStatusUnknownError};
                float best = std::numeric_limits<float>::max();
                visit_float(dxDesc.GetType(), [&](auto as_float) {
                    for(const auto& sol : all)
                    {
                        float elapsed = 0.0f;
                        const int rc  = EvaluateDataImplicitGemmSolution(handle,
                                                                        sol,
                                                                        dy,
                                                                        w,
                                                                        dx,
                                                                        dxDesc,
                                                                        false,
                                                                        this->GetConvStrides(),
                                                                        workSpace,
                                                                        workSpaceSize,
                                                                        as_float(0.0f),
                                                                        lowp_quant,
                                                                        elapsed);

                        if(rc != 0)
                        {
                            MIOPEN_LOG_E(sol << " returns " << rc);
                        }
                        else
                        {
                            MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ")
                                             << best);
                            if(elapsed < best)
                            {
                                best     = elapsed;
                                selected = sol;
                            }
                        }
                    }
                });

                if(selected.Succeeded())
                {
                    const std::string algorithm_name = "miopenConvolutionBwdDataAlgoImplicitGEMM";
                    AddKernels(handle, algorithm_name, network_config, selected, nullptr);
                    MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                              << selected.workspce_sz);
                    record.SetValues(algorithm_name,
                                     FindDbData{selected.solver_id,
                                                best,
                                                selected.workspce_sz,
                                                {algorithm_name, network_config}});
                }
            }

            if(GetSpatialDimension() == 2 && GetConvDilations()[0] == 1 &&
               GetConvDilations()[1] == 1 && group_count == 1 && !use_winograd_only)
            {
                // FFT algo
                std::vector<KernelInvoke> kernels_fft;
                size_t workspace_fft = BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc);
                if(FindBwdFFTKernel(
                       handle, dyDesc, wDesc, dxDesc, workspace_fft, kernels_fft, network_config) ==
                   0)
                {
                    (void)kernels_fft; // not used now, but needed as fft coverage widens
                    if(workSpace != nullptr && workSpaceSize >= workspace_fft)
                    {
                        float time_fft = ExecuteBwdFFTKernel(handle,
                                                             dyDesc,
                                                             dy,
                                                             wDesc,
                                                             w,
                                                             dxDesc,
                                                             dx,
                                                             workSpace,
                                                             workSpaceSize,
                                                             network_config,
                                                             true);
                        record.SetValues("miopenConvolutionBwdDataAlgoFFT",
                                         FindDbData{
                                             "fft",
                                             time_fft,
                                             workspace_fft,
                                             {"miopenConvolutionBwdDataAlgoFFT", network_config},
                                         });
                    }
                }
            }

/// The SCGemm Solver is applicable for Bwd Data convolutions, but it is not used here.
/// \todo Decide & use SCGemm here and in GWSS. Or, make the Solver not applicable for Bwd.

#if MIOPEN_USE_GEMM
            if(!use_winograd_only && !miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}) &&
               !(IsAnyBufferBF16(dxDesc, dyDesc, wDesc) && !IsUseRocBlas))
            { // GEMM based
                ValidateGroupCount(dxDesc, wDesc, *this);

                const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

                std::size_t in_n, in_c;
                std::tie(in_n, in_c) = tie_pick<0, 1>()(dxDesc.GetLengths());

                std::size_t wei_k = wDesc.GetLengths()[0];

                std::size_t spatial_dim = GetSpatialDimension();

                auto in_spatial  = boost::adaptors::slice(dxDesc.GetLengths(), 2, 2 + spatial_dim);
                auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
                auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);

                // 1x1 does not require col2im
                if(GetSpatialDimension() == 2 &&
                   miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                   miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                   miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; }) &&
                   workSpace != nullptr &&
                   workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc))
                {
                    if(group_count > 1)
                    {
                        MIOPEN_LOG_FUNCTION("groupconv, 1x1 u2xv2");
                    }
                    else
                    {
                        MIOPEN_LOG_FUNCTION("convolution, 1x1 u2xv2");
                    }
                    float time_gemm = 0;

                    // Initialization required for upsampling in bwd direction
                    float zero = 0.f;
                    SetTensor(handle, dxDesc, dx, &zero);
                    time_gemm = handle.GetKernelTime();

                    // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
                    transpose_NCHW2CNHW(handle,
                                        in_n,
                                        wei_k,
                                        out_spatial[0],
                                        out_spatial[1],
                                        out_spatial[0],
                                        out_spatial[1],
                                        dy,
                                        workSpace,
                                        0,
                                        0,
                                        1,
                                        1,
                                        dyDesc.GetType());
                    time_gemm += handle.GetKernelTime();

                    GemmDescriptor gemm_desc =
                        group_count > 1
                            ? CreateGemmDescriptorGroupConvCNHWBwdData(
                                  wDesc, dyDesc, dxDesc, group_count)
                            : CreateGemmDescriptorConvCNHWBwdData(wDesc, dyDesc, dxDesc);

                    auto kcache_key = FindDbKCacheKey{};

                    miopenStatus_t gemm_status =
                        CallGemmTimeMeasure(handle,
                                            gemm_desc,
                                            w,
                                            0,
                                            workSpace,
                                            0,
                                            workSpace,
                                            dyDesc.GetElementSize(),
                                            &kcache_key,
                                            time_precision,
                                            group_count > 1 ? callGemmStridedBatched : callGemm);

                    time_gemm += handle.GetKernelTime();

                    transpose_CNHW2NCHW(handle,
                                        in_n,
                                        in_c,
                                        out_spatial[0],
                                        out_spatial[1],
                                        in_spatial[0],
                                        in_spatial[1],
                                        workSpace,
                                        dx,
                                        dyDesc.GetElementSize(),
                                        0,
                                        GetConvStrides()[0],
                                        GetConvStrides()[1],
                                        dyDesc.GetType());
                    time_gemm += handle.GetKernelTime();

                    if(gemm_status == miopenStatusSuccess)
                        record.SetValues(
                            "miopenConvolutionBwdDataAlgoGEMM",
                            FindDbData{"gemm",
                                       time_gemm,
                                       BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc),
                                       kcache_key});
                }
                // 1x1_stride=1 convolutions use GEMM and zero workspace
                else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                        miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                        miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
                {
                    if(group_count > 1)
                    {
                        MIOPEN_LOG_FUNCTION("groupconv, 1x1");
                    }
                    else
                    {
                        MIOPEN_LOG_FUNCTION("convolution, 1x1");
                    }
                    // dx = transpose(w) * dy
                    GemmDescriptor gemm_desc =
                        group_count > 1 ? CreateGemmDescriptorGroupConvBwdData(
                                              wDesc, dyDesc, dxDesc, group_count)
                                        : CreateGemmStridedBatchedDescriptorConv1x1BwdData(
                                              wDesc, dyDesc, dxDesc);

                    auto kcache_key = FindDbKCacheKey{};

                    miopenStatus_t gemm_status = CallGemmTimeMeasure(handle,
                                                                     gemm_desc,
                                                                     w,
                                                                     0,
                                                                     dy,
                                                                     0,
                                                                     dx,
                                                                     0,
                                                                     &kcache_key,
                                                                     time_precision,
                                                                     callGemmStridedBatched);

                    float time_gemm = handle.GetKernelTime();
                    if(group_count > 1)
                        time_gemm *= in_n;

                    if(gemm_status == miopenStatusSuccess)
                        record.SetValues("miopenConvolutionBwdDataAlgoGEMM",
                                         FindDbData{
                                             "gemm", time_gemm, 0, kcache_key,
                                         });
                }
                // if not 1x1
                else if(workSpace != nullptr &&
                        workSpaceSize >=
                            (BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count))
                {
                    if(group_count > 1)
                    {
                        MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                    }
                    else
                    {
                        MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                    }
                    float time_col2im = 0;
                    int in_offset     = 0;

                    // dx = transpose(w) * dy
                    GemmDescriptor gemm_desc =
                        group_count > 1 ? CreateGemmDescriptorGroupConvBwdData(
                                              wDesc, dyDesc, dxDesc, group_count)
                                        : CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);

                    auto kcache_key = FindDbKCacheKey{};

                    miopenStatus_t gemm_status = CallGemmTimeMeasure(
                        handle,
                        gemm_desc,
                        w,
                        0,
                        dy,
                        0,
                        workSpace,
                        0,
                        &kcache_key,
                        time_precision,
                        group_count > 1 ? callGemmStridedBatched : callGemm,
                        group_count > 1 ? GemmBackend_t::rocblas : GemmBackend_t::miopengemm);

                    float time_gemm = in_n * handle.GetKernelTime();
                    time_col2im     = Col2ImGPU(handle,
                                            GetSpatialDimension(),
                                            workSpace,
                                            out_spatial,
                                            wei_spatial,
                                            GetConvPads(),
                                            GetConvStrides(),
                                            GetConvDilations(),
                                            in_c,
                                            in_spatial,
                                            dx,
                                            in_offset,
                                            dyDesc.GetType());

                    time_gemm += in_n * time_col2im;

                    if(gemm_status == miopenStatusSuccess)
                        record.SetValues(
                            "miopenConvolutionBwdDataAlgoGEMM",
                            FindDbData{
                                "gemm",
                                time_gemm,
                                BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count,
                                kcache_key,
                            });
                }
            }
#endif
        });

    if(perf_db.empty())
        MIOPEN_THROW(miopenStatusUnknownError, "Backward Data Algo cannot be executed");

    std::sort(begin(perf_db), end(perf_db));

    for(const auto& entry : perf_db)
        MIOPEN_LOG_I(entry.name << "\t" << entry.time << "\t" << entry.workspace);

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(perf_db.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].bwd_data_algo = StringToConvolutionBwdDataAlgo(perf_db[i].name);
        perfResults[i].time          = perf_db[i].time;
        perfResults[i].memory        = perf_db[i].workspace;
    }

    MIOPEN_LOG_I("BWD Chosen Algorithm: " << perf_db[0].solver_id << " , " << perf_db[0].workspace
                                          << ", "
                                          << perf_db[0].time);
}

template <class TKernels>
void ConvBwdImplicitGemm(const ConvolutionContext& ctx,
                         Handle& handle,
                         const ConvBwdTensors& tensors,
                         Data_t workSpace,
                         std::size_t workSpaceSize,
                         const TKernels& kernels,
                         float lowp_quant);

static void ConvBwdCheckNumerics(Handle& handle,
                                 const ConvBwdTensors& tensors,
                                 const void* beta,
                                 std::function<void()>&& worker)
{
    if(!miopen::CheckNumericsEnabled())
    {
        worker();
        return;
    }

    miopen::checkNumericsInput(handle, tensors.dyDesc, tensors.dy);
    miopen::checkNumericsInput(handle, tensors.wDesc, tensors.w);
    if(!float_equal(*(static_cast<const float*>(beta)), 0))
        miopen::checkNumericsInput(handle, tensors.dxDesc, tensors.dx);

    worker();

    miopen::checkNumericsOutput(handle, tensors.dxDesc, tensors.dx);
}

// BackwardDataAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardData(Handle& handle,
                                                    const void* alpha,
                                                    const TensorDescriptor& dyDesc,
                                                    ConstData_t dy,
                                                    const TensorDescriptor& wDesc,
                                                    ConstData_t w,
                                                    miopenConvBwdDataAlgorithm_t algo,
                                                    const void* beta,
                                                    const TensorDescriptor& dxDesc,
                                                    Data_t dx,
                                                    Data_t workSpace,
                                                    size_t workSpaceSize) const
{
    MIOPEN_LOG_I("algo = " << algo << ", workspace = " << workSpaceSize);
    auto tensors = ConvBwdTensors{dyDesc, dy, wDesc, w, dxDesc, dx};

    ValidateConvTensors(tensors);
    ValidateAlphaBeta(alpha, beta);

    if(wDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    ConvBwdCheckNumerics(handle, tensors, beta, [&]() {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        ValidateGroupCount(dxDesc, wDesc, *this);

        const auto algorithm_name = AlgorithmName{ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(algo), conv::Direction::BackwardData)};

        auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
        ctx.SetStream(&handle);
        const auto network_config = ctx.BuildConfKey();
        const auto& invoker       = handle.GetInvoker(network_config, boost::none, algorithm_name);

        if(invoker)
        {
            const auto& invoke_ctx = conv::DataInvokeParams{tensors, workSpace, workSpaceSize};
            (*invoker)(handle, invoke_ctx);
            return;
        }

        switch(algo)
        {
        case miopenConvolutionBwdDataAlgoDirect:
        case miopenConvolutionBwdDataAlgoWinograd:
            MIOPEN_THROW("No invoker was registered for convolution backward. Was find executed?");

        case miopenConvolutionBwdDataAlgoImplicitGEMM:
        {

            auto&& kernels =
                handle.GetKernels("miopenConvolutionBwdDataAlgoImplicitGEMM", network_config);
            ConvBwdImplicitGemm(
                ctx, handle, tensors, workSpace, workSpaceSize, kernels, lowp_quant);
            break;
        }

        case miopenConvolutionBwdDataAlgoGEMM:
            ConvBwdGemm(handle, tensors, workSpace, workSpaceSize);
            break;

        case miopenConvolutionBwdDataAlgoFFT:
            ConvBwdFFT(handle, tensors, workSpace, workSpaceSize, network_config);
            break;

        case miopenTransposeBwdDataAlgoGEMM: break;
        }
    });
}

template <class TKernels>
void ConvBwdImplicitGemm(const ConvolutionContext& /*ctx*/,
                         Handle& handle,
                         const ConvBwdTensors& tensors,
                         Data_t workSpace,
                         std::size_t /*workSpaceSize*/,
                         const TKernels& kernels,
                         float lowp_quant)
{
    if(kernels.empty())
        MIOPEN_THROW("Error running Direct Backward convolution. Was Find() executed previously?");

    // Miminum checks. Only check what is required to select
    // proper invocation procedure & workspace sanity.
    auto kernel = kernels[0];

    float elapsed = 0;
    if((tensors.dxDesc.GetType() == miopenHalf || tensors.dxDesc.GetType() == miopenBFloat16) &&
       (kernel.GetName() ==
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
        kernel.GetName() ==
            "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw" ||
        kernel.GetName() == "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw"))
    {
        float zero = 0.f;
        TensorDescriptor workspaceDesc(
            miopenFloat, tensors.dxDesc.GetLengths(), tensors.dxDesc.GetStrides());
        SetTensor(handle, workspaceDesc, workSpace, &zero);
        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();

        kernel(tensors.dy, tensors.w, workSpace);
        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();

        CastTensor(handle, &lowp_quant, workspaceDesc, workSpace, tensors.dxDesc, tensors.dx, 0, 0);
        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();
    }
    else if((kernel.GetName() ==
             "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer") ||
            (kernel.GetName() ==
             "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer"))
    {
        bool hasStride = (tensors.dyDesc.GetLengths()[2] != tensors.dxDesc.GetLengths()[2]) ||
                         (tensors.dyDesc.GetLengths()[3] != tensors.dxDesc.GetLengths()[3]);
        /// \todo set zero within implicitGEMM kernel
        if(hasStride)
        {
            MIOPEN_LOG_I2("hasStride, call SetTensor with zero");
            float zero = 0.f;
            SetTensor(handle, tensors.dxDesc, tensors.dx, &zero);

            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
        }

        kernel(tensors.dy, tensors.w, tensors.dx);

        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();
    }
    else if(kernel.GetName() ==
                "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw" ||
            kernel.GetName() ==
                "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw" ||
            kernel.GetName() ==
                "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_gnchw_gkcyx_gnkhw")
    {
        // this kernel accumulate results into input tensor, therefore need to set zero
        float zero = 0.f;
        SetTensor(handle, tensors.dxDesc, tensors.dx, &zero);
        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();

        kernel(tensors.dy, tensors.w, tensors.dx);
        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();
    }
    else if(kernel.GetName() ==
            "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw")
    {
        // \todo this kernel doesn't always need to set-zero
        float zero = 0.f;
        SetTensor(handle, tensors.dxDesc, tensors.dx, &zero);

        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();

        // a group kernels (compiled from same source code) will be launched
        for(const auto& k : kernels)
        {
            k(tensors.dy, tensors.w, tensors.dx);
            elapsed += handle.GetKernelTime();
        }
    }
    else
    {
        MIOPEN_THROW("Error running implicit GEMM backward data convolution (none workspace?)");
    }

    if(handle.IsProfilingEnabled())
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed);
    }
}

void ConvolutionDescriptor::ConvBwdGemm(Handle& handle,
                                        const ConvBwdTensors& tensors,
                                        Data_t workSpace,
                                        std::size_t workSpaceSize) const
{
#if MIOPEN_USE_GEMM
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
    {
        MIOPEN_THROW("GEMM convolution is disabled");
    }
    if(IsAnyBufferBF16(tensors.dxDesc, tensors.dyDesc, tensors.wDesc) && !IsUseRocBlas)
    {
        MIOPEN_THROW("GEMM convolution is unsupported");
    }

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(tensors.dxDesc.GetLengths());

    std::size_t wei_k = tensors.wDesc.GetLengths()[0];

    std::size_t spatial_dim = GetSpatialDimension();

    auto in_spatial  = boost::adaptors::slice(tensors.dxDesc.GetLengths(), 2, 2 + spatial_dim);
    auto wei_spatial = boost::adaptors::slice(tensors.wDesc.GetLengths(), 2, 2 + spatial_dim);
    auto out_spatial = boost::adaptors::slice(tensors.dyDesc.GetLengths(), 2, 2 + spatial_dim);

    if(GetSpatialDimension() == 2 && miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; }))
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, 1x1, u2xv2");
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, 1x1, u2xv2");
        }

        float t1 = 0;
        // Initialization required for upsampling in bwd direction
        float zero = 0.f;
        SetTensor(handle, tensors.dxDesc, tensors.dx, &zero);
        if(handle.IsProfilingEnabled())
            t1 = handle.GetKernelTime();

        assert(workSpace != nullptr &&
               workSpaceSize >=
                   BackwardDataGetWorkSpaceSizeGEMMTranspose(tensors.dyDesc, tensors.dxDesc));

        transpose_NCHW2CNHW(handle,
                            in_n,
                            wei_k,
                            out_spatial[0],
                            out_spatial[1],
                            out_spatial[0],
                            out_spatial[1],
                            tensors.dy,
                            workSpace,
                            0,
                            0,
                            1,
                            1,
                            tensors.dyDesc.GetType());
        if(handle.IsProfilingEnabled())
            t1 += handle.GetKernelTime();

        if(group_count > 1)
        {
            GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvCNHWBwdData(
                tensors.wDesc, tensors.dyDesc, tensors.dxDesc, group_count);

            CallGemmStridedBatched(handle,
                                   gemm_desc,
                                   tensors.w,
                                   0,
                                   workSpace,
                                   0,
                                   workSpace,
                                   tensors.dyDesc.GetElementSize(),
                                   nullptr,
                                   false);
        }
        else
        {
            // tensors.dx = CNHW2NCHW(transpose(tensors.w) * NCHW2CNHW(tensors.dy))
            GemmDescriptor gemm_desc =
                CreateGemmDescriptorConvCNHWBwdData(tensors.wDesc, tensors.dyDesc, tensors.dxDesc);

            // tensors.dx = CNHW2NCHW(transpose(tensors.w) * NCHW2CNHW(tensors.dy))
            CallGemm(handle,
                     gemm_desc,
                     tensors.w,
                     0,
                     workSpace,
                     0,
                     workSpace,
                     tensors.dyDesc.GetElementSize(),
                     nullptr,
                     false);
        }
        if(handle.IsProfilingEnabled())
            t1 += handle.GetKernelTime();

        transpose_CNHW2NCHW(handle,
                            in_n,
                            in_c,
                            out_spatial[0],
                            out_spatial[1],
                            in_spatial[0],
                            in_spatial[1],
                            workSpace,
                            tensors.dx,
                            tensors.dyDesc.GetElementSize(),
                            0,
                            GetConvStrides()[0],
                            GetConvStrides()[1],
                            tensors.dyDesc.GetType());
        if(handle.IsProfilingEnabled())
            t1 += handle.GetKernelTime();

        if(handle.IsProfilingEnabled())
        {
            handle.ResetKernelTime();
            handle.AccumKernelTime(t1);
        }
    }
    // 1x1_stride=1 convolutions use GEMM and zero workspace
    else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, 1x1");

            GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvBwdData(
                tensors.wDesc, tensors.dyDesc, tensors.dxDesc, group_count);

            float time_0 = 0;
            for(std::size_t i = 0; i < in_n; i++)
            {
                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                              in_spatial.end(),
                                                              std::size_t(1),
                                                              std::multiplies<std::size_t>());

                std::size_t out_offset = i * wei_k * out_spatial_size;

                std::size_t in_offset = i * in_c * in_spatial_size;

                CallGemmStridedBatched(handle,
                                       gemm_desc,
                                       tensors.w,
                                       0,
                                       tensors.dy,
                                       out_offset,
                                       tensors.dx,
                                       in_offset,
                                       nullptr,
                                       false);

                if(handle.IsProfilingEnabled())
                {
                    if(i == in_n - 1)
                        handle.AccumKernelTime(time_0);
                    time_0 += handle.GetKernelTime();
                }
            }
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, 1x1");

            // tensors.dx = transpose(tensors.w) * tensors.dy
            GemmDescriptor gemm_desc = CreateGemmStridedBatchedDescriptorConv1x1BwdData(
                tensors.wDesc, tensors.dyDesc, tensors.dxDesc);

            // tensors.dx = transpose(tensors.w) * tensors.dy
            CallGemmStridedBatched(
                handle, gemm_desc, tensors.w, 0, tensors.dy, 0, tensors.dx, 0, nullptr, false);
        }
    }
    // if not 1x1
    else
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, non 1x1");
        }
        assert(workSpace != nullptr &&
               workSpaceSize >=
                   (BackwardDataGetWorkSpaceSizeGEMM(tensors.wDesc, tensors.dyDesc) * group_count));

        // tensors.dx = transpose(tensors.w) * tensors.dy
        GemmDescriptor gemm_desc{};
        if(group_count > 1)
            gemm_desc = CreateGemmDescriptorGroupConvBwdData(
                tensors.wDesc, tensors.dyDesc, tensors.dxDesc, group_count);
        else
            gemm_desc =
                CreateGemmDescriptorConvBwdData(tensors.wDesc, tensors.dyDesc, tensors.dxDesc);

        handle.ResetKernelTime();

        std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        std::size_t in_spatial_size = std::accumulate(
            in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        float time_0 = 0;
        float t1     = 0;
        for(std::size_t i = 0; i < in_n; i++)
        {
            std::size_t out_offset = i * wei_k * out_spatial_size;
            std::size_t in_offset  = i * in_c * in_spatial_size;

            // tensors.dx = transpose(tensors.w) * tensors.dy
            if(group_count > 1)
                CallGemmStridedBatched(handle,
                                       gemm_desc,
                                       tensors.w,
                                       0,
                                       tensors.dy,
                                       out_offset,
                                       workSpace,
                                       0,
                                       nullptr,
                                       false);
            else
                CallGemm(handle,
                         gemm_desc,
                         tensors.w,
                         0,
                         tensors.dy,
                         out_offset,
                         workSpace,
                         0,
                         nullptr,
                         false,
                         GemmBackend_t::miopengemm);

            if(handle.IsProfilingEnabled())
                t1 = handle.GetKernelTime();

            Col2ImGPU(handle,
                      GetSpatialDimension(),
                      workSpace,
                      out_spatial,
                      wei_spatial,
                      GetConvPads(),
                      GetConvStrides(),
                      GetConvDilations(),
                      in_c,
                      in_spatial,
                      tensors.dx,
                      in_offset,
                      tensors.dyDesc.GetType());

            // Update times for both the kernels
            if(handle.IsProfilingEnabled())
            {
                if(i == in_n - 1)
                    handle.AccumKernelTime(t1 + time_0);
                else
                    handle.AccumKernelTime(t1);
                time_0 += handle.GetKernelTime();
            }
        }
    }
#ifdef NDEBUG
    std::ignore = workSpaceSize;
#endif
#else
    std::ignore = handle;
    std::ignore = tensors;
    std::ignore = workSpace;
    std::ignore = workSpaceSize;
    MIOPEN_THROW("GEMM is not supported");
#endif
}

void ConvolutionDescriptor::ConvBwdFFT(Handle& handle,
                                       const ConvBwdTensors& tensors,
                                       Data_t workSpace,
                                       size_t workSpaceSize,
                                       const NetworkConfig& kcache_key) const
{
    assert(workSpaceSize >=
           BackwardGetWorkSpaceSizeFFT(tensors.wDesc, tensors.dyDesc, tensors.dxDesc));

    if(workSpace == nullptr || workSpaceSize == 0)
        MIOPEN_THROW("Error running FFT: none workspace");

    bool timed  = handle.IsProfilingEnabled();
    float timev = ExecuteBwdFFTKernel(handle,
                                      tensors.dyDesc,
                                      tensors.dy,
                                      tensors.wDesc,
                                      tensors.w,
                                      tensors.dxDesc,
                                      tensors.dx,
                                      workSpace,
                                      workSpaceSize,
                                      kcache_key,
                                      timed);

    if(timed)
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(timev);
    }
}

std::size_t ConvolutionDescriptor::GetBackwardSolutionCount(Handle& handle,
                                                            const TensorDescriptor& dyDesc,
                                                            const TensorDescriptor& wDesc,
                                                            const TensorDescriptor& dxDesc) const
{
    MIOPEN_LOG_I("");
    ValidateGroupCount(dxDesc, wDesc, *this);
    const auto problem =
        ProblemDescription{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
    const auto count = GetSolutionCount(handle, problem);
    if(count > 0)
        return count;
    return GetBwdSolutionCountFallback(dyDesc, wDesc, dxDesc);
}

void ConvolutionDescriptor::GetBackwardSolutions(Handle& handle,
                                                 const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& wDesc,
                                                 const TensorDescriptor& dxDesc,
                                                 size_t maxSolutionCount,
                                                 size_t* solutionCount,
                                                 miopenConvSolution_t* solutions) const
{
    MIOPEN_LOG_I("");
    if(solutionCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutionCount cannot be nullptr");
    if(solutions == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutions cannot be nullptr");

    const auto problem =
        ProblemDescription{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
    GetSolutions(handle,
                 problem,
                 maxSolutionCount,
                 solutionCount,
                 solutions,
                 StringToConvolutionBwdDataAlgo);

    if(*solutionCount == 0)
        GetBwdSolutionsFallback(
            handle, dyDesc, wDesc, dxDesc, maxSolutionCount, solutionCount, solutions);
}

void ConvolutionDescriptor::CompileBackwardSolution(Handle& handle,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dxDesc,
                                                    solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString());

    auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
    ctx.SetStream(&handle);
    ctx.disable_search_enforce = true;

    CompileSolution(handle, solver_id, ctx, conv::Direction::BackwardData, [&]() {
        const auto workspace_fft = BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc);
        std::vector<KernelInvoke> ignore0;
        const auto network_config = ctx.BuildConfKey();
        FindBwdFFTKernel(handle, dyDesc, wDesc, dxDesc, workspace_fft, ignore0, network_config);
    });
}

std::size_t ConvolutionDescriptor::GetBackwardSolutionWorkspaceSize(Handle& handle,
                                                                    const TensorDescriptor& dyDesc,
                                                                    const TensorDescriptor& wDesc,
                                                                    const TensorDescriptor& dxDesc,
                                                                    solver::Id solver_id) const
{
    MIOPEN_LOG_I2("solver_id = " << solver_id.ToString());
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm, "invalid solution id = " + solver_id.ToString());
    if(solver_id != solver::Id::gemm() && solver_id != solver::Id::fft())
    {
        auto sol = solver_id.GetSolver();
        auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        if(sol.IsApplicable(ctx))
            return sol.GetWorkspaceSize(ctx);
        else
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "The supplied solution id: " + solver_id.ToString() +
                             " is not applicable to the current problem");
        }
    }
    else if(solver_id == solver::Id::fft())
        return BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc);
    return GetBwdSolutionWorkspaceSizeFallback(dyDesc, wDesc, dxDesc, solver_id);
}

void ConvolutionDescriptor::ConvolutionBackwardImmediate(Handle& handle,
                                                         const TensorDescriptor& dyDesc,
                                                         ConstData_t dy,
                                                         const TensorDescriptor& wDesc,
                                                         ConstData_t w,
                                                         const TensorDescriptor& dxDesc,
                                                         Data_t dx,
                                                         Data_t workSpace,
                                                         std::size_t workSpaceSize,
                                                         solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString() << ", workspace = " << workSpaceSize);
    auto tensors = ConvBwdTensors{dyDesc, dy, wDesc, w, dxDesc, dx};

    ValidateConvTensors(tensors);

    if(wDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    static const float beta = 0.0f;
    ConvBwdCheckNumerics(handle, tensors, &beta, [&]() {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        ValidateGroupCount(dxDesc, wDesc, *this);

        auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};

        if(CheckInvokerSupport(solver_id, conv::Direction::BackwardData))
        {
            const auto invoker    = LoadOrPrepareInvoker(handle, ctx, solver_id);
            const auto invoke_ctx = conv::DataInvokeParams{tensors, workSpace, workSpaceSize};
            invoker(handle, invoke_ctx);
            return;
        }

        if(solver_id == solver::Id::gemm())
        {
            ConvBwdGemm(handle, tensors, workSpace, workSpaceSize);
            return;
        }

        ctx.SetStream(&handle);
        const auto network_config = ctx.BuildConfKey();
        const auto algo_name      = solver_id.GetAlgo(conv::Direction::BackwardData);
        const auto&& chk_kernels  = handle.GetKernels(algo_name, network_config);
        auto v_chk_kernels = std::vector<KernelInvoke>{chk_kernels.begin(), chk_kernels.end()};

        if(!v_chk_kernels.empty())
        {
            MIOPEN_LOG_I2(
                "Found previously compiled kernels for solution: " << solver_id.ToString());
            if(solver_id == solver::Id::fft())
                ConvBwdFFT(handle, tensors, workSpace, workSpaceSize, network_config);
            else if(algo_name == "miopenConvolutionBwdDataAlgoImplicitGEMM")
                ConvBwdImplicitGemm(
                    ctx, handle, tensors, workSpace, workSpaceSize, v_chk_kernels, lowp_quant);
            else
                MIOPEN_THROW("Invalid algorithm: " + algo_name);
            return;
        }

        const auto problem =
            ProblemDescription{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
        const FindDbRecord fdb_record{handle, problem};

        for(const auto& pair : fdb_record)
        {
            if(solver::Id{pair.second.solver_id} != solver_id)
                continue;

            const auto&& kernels = handle.GetKernels(pair.second.kcache_key.algorithm_name,
                                                     pair.second.kcache_key.network_config);
            auto v_kernels = std::vector<KernelInvoke>{kernels.begin(), kernels.end()};

            if(solver_id == solver::Id::fft())
            {
                if(v_kernels.empty())
                    FindBwdFFTKernel(
                        handle, dyDesc, wDesc, dxDesc, workSpaceSize, v_kernels, network_config);
                ConvBwdFFT(handle, tensors, workSpace, workSpaceSize, network_config);
                return;
            }

            if(v_kernels.empty())
                v_kernels = CompileSolver(handle, ctx, solver_id, pair.second.kcache_key);

            else if(pair.second.kcache_key.algorithm_name ==
                    "miopenConvolutionBwdDataAlgoImplicitGEMM")
                ConvBwdImplicitGemm(
                    ctx, handle, tensors, workSpace, workSpaceSize, v_kernels, lowp_quant);
            else
                MIOPEN_THROW("Invalid algorithm: " + pair.second.kcache_key.algorithm_name);
            return;
        }

        // Todo: solver not found in find-db.
        MIOPEN_THROW(miopenStatusNotImplemented);
    });
}

template <int WinoDataH, int WinoFilterH, typename T>
inline void EvaluateWinograd3x3MultipassWrW(Handle& handle,
                                            const ConvolutionContext& ctx,
                                            const ConvWrwTensors& tensors,
                                            Data_t workSpace,
                                            T kernels,
                                            int pad_H,
                                            int pad_W,
                                            float* elapsed = nullptr)
{
    EvaluateWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataH, WinoFilterH, T>(
        handle, ctx, tensors, workSpace, kernels, pad_H, pad_W, elapsed);
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW, typename T>
inline void EvaluateWinograd3x3MultipassWrW(Handle& handle,
                                            const ConvolutionContext& ctx,
                                            const ConvWrwTensors& tensors,
                                            Data_t workSpace,
                                            T kernels,
                                            int pad_H,
                                            int pad_W,
                                            float* elapsed = nullptr)

{
#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)
    int flags         = 0;
    int reserved      = 0;
    int* reserved_ptr = nullptr;
    int unused        = 0;
    int N, C, H, W, K, n_groups, out_H, out_W, R, S;

    GetCompiledInParameters(
        ctx, &C, &K, &R, &S, &N, &n_groups, &H, &W, &out_H, &out_W, &unused, &unused);
    // clang-format off
    BuffInfo
        in_buff_info(
            GetSwappedNCLayout(GetMemLayout_t(ctx.in_layout)),
            N, C, H, W, 1,
            GetTypeSize(ctx.in_data_type)),
        out_buff_info(
            GetSwappedNCLayout(GetMemLayout_t(ctx.out_layout)),
            N, K, out_H, out_W, 1,
            GetTypeSize(ctx.out_data_type)),
        weights_buff_info(
            // weights_layout unsupported ... GetSwappedNCLayout(GetMemLayout_t(ctx.weights_layout))
            GetSwappedNCLayout(MemLayout_t::NCHW),
            K, C, R, S, 1,
            GetTypeSize(ctx.weights_data_type));

    int wino_xform_h =
            solver::ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolverWinoXformHWSize(ctx,0),
        wino_xform_w =
            solver::ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolverWinoXformHWSize(ctx,1);
    WinogradBufferInfo <WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
        // cppcheck-suppress unreadVariable
        wino_in(N,K,C,out_H,out_W,R,S,
            MemLayout_t::HWNC,
            1,GetTypeSize(ctx.in_data_type),
            ConvWinoBuffType::Input,
            wino_xform_h,
            wino_xform_w),
        // cppcheck-suppress unreadVariable
        wino_out(N,K,C,out_H,out_W,R,S,
            MemLayout_t::HWNC,
            1,GetTypeSize(ctx.out_data_type),
            ConvWinoBuffType::Output,
            wino_xform_h,
            wino_xform_w),
        // cppcheck-suppress unreadVariable
        wino_wei(N,K,C,out_H,out_W,R,S,
            MemLayout_t::HWNC,
            1,GetTypeSize(ctx.weights_data_type),
            ConvWinoBuffType::Weight,
            wino_xform_h,
            wino_xform_w);
    float total_time = 0;
    // clang-format on
    for(const auto& cur_kernel : kernels)
    {
        BuffInfo* d_buf         = nullptr;
        BuffInfo* o_buf         = nullptr;
        Data_t buff_out_adr     = nullptr;
        auto f_buf              = &weights_buff_info;
        auto const_buff_in_adr  = tensors.x;
        auto buff_in_adr        = workSpace;
        bool const_input        = false;
        float cur_time          = 0;
        int flat_GroupCountMult = 1;

        size_t wino_in_offset = 0, wino_out_offset = wino_in.buff_info.total_byte_size,
               wino_wei_offset = wino_out_offset + wino_out.buff_info.total_byte_size;

        size_t buff_in_addr_offset = 0, buff_out_addr_offset = 0;

        if(cur_kernel.GetName() ==
           solver::ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
               GetSolverKernelNames(0)) // Input
                                        // Transform
        {
            d_buf               = &in_buff_info;
            o_buf               = &(wino_in.buff_info);
            const_buff_in_adr   = tensors.x;
            buff_out_adr        = workSpace;
            buff_in_addr_offset = wino_in_offset;
            const_input         = true;
            flat_GroupCountMult =
                solver::ConvWinograd3x3MultipassWrW<WinoDataH,
                                                    WinoFilterH,
                                                    WinoDataW,
                                                    WinoFilterW>::GetGroupCountMult();
        }
        else if(cur_kernel.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<WinoDataH,
                                                    WinoFilterH,
                                                    WinoDataW,
                                                    WinoFilterW>::GetSolverKernelNames(1)) // filter
        // Transform
        {
            d_buf                = &weights_buff_info;
            o_buf                = &(wino_wei.buff_info);
            const_buff_in_adr    = tensors.dy;
            buff_out_adr         = workSpace;
            buff_out_addr_offset = wino_wei_offset;
            const_input          = true;
            flat_GroupCountMult =
                solver::ConvWinograd3x3MultipassWrW<WinoDataH,
                                                    WinoFilterH,
                                                    WinoDataW,
                                                    WinoFilterW>::GetGroupCountMult();
        }
        else // Output
             // and GEMM
        {
            const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));
            int m = N, n = K, k = wino_in.wino_c;
            int lda = k, ldb = k, ldc = n;
            int batch_count       = wino_xform_h * wino_xform_w;
            long long int strideA = m * k * 1LL, strideB = k * n * 1LL, strideC = m * n * 1LL;
            float alpha = 1., beta = 0.0;
            // clang-format off
            GemmDescriptor wino_gemm_desc{false,false,true,m,n,k,
                lda,ldb,ldc,batch_count,strideA,strideB,
                strideC,alpha,beta,ctx.in_data_type};

            if(elapsed == nullptr)
                CallGemmStridedBatched(handle,
                            wino_gemm_desc,
                            workSpace,
                            static_cast<int>(wino_in_offset / GetTypeSize(ctx.in_data_type)),
                            workSpace,
                            static_cast<int>(wino_wei_offset / GetTypeSize(ctx.in_data_type)),
                            workSpace,
                            static_cast<int>(wino_out_offset / GetTypeSize(ctx.in_data_type)),
                            nullptr,
                            false,
                            GemmBackend_t::rocblas);
            else
                CallGemmTimeMeasure(handle,
                            wino_gemm_desc,
                            workSpace,
                            static_cast<int>(wino_in_offset / GetTypeSize(ctx.in_data_type)),
                            workSpace,
                            static_cast<int>(wino_wei_offset / GetTypeSize(ctx.in_data_type)),
                            workSpace,
                            static_cast<int>(wino_out_offset / GetTypeSize(ctx.in_data_type)),
                            nullptr,
                            time_precision,
                            CallGemmType_t::callGemmStridedBatched,
                            GemmBackend_t::rocblas);
            // clang-format on
            if(handle.IsProfilingEnabled() || elapsed != nullptr)
            {
                cur_time = handle.GetKernelTime();
                total_time += cur_time;
            }
            if(elapsed != nullptr)
            {
                *elapsed += cur_time;
                MIOPEN_LOG_I2("WRW_WINO_GEMM: " << cur_time);
            }

            d_buf               = &(wino_out.buff_info);
            o_buf               = &(out_buff_info);
            buff_in_adr         = workSpace;
            buff_in_addr_offset = wino_out_offset;
            buff_out_adr        = tensors.dw;
        }

        const auto input_ptr = static_cast<const void*>(
            static_cast<const char*>(const_input ? const_buff_in_adr : buff_in_adr) +
            buff_in_addr_offset);
        const auto output_ptr =
            static_cast<void*>(static_cast<char*>(buff_out_adr) + buff_out_addr_offset);

        cur_kernel(N,
                   C,
                   H,
                   W,
                   K,
                   n_groups * flat_GroupCountMult,
                   flags,
                   reserved,
                   input_ptr,
                   reserved_ptr,
                   output_ptr,
                   reserved_ptr,
                   R,
                   S,
                   pad_H,
                   pad_W,
                   out_H,
                   out_W,
                   reserved_ptr,
                   reserved,
                   d_buf->byte_stride.nk,
                   d_buf->byte_stride.c,
                   d_buf->byte_stride.h,
                   d_buf->byte_stride.w,
                   f_buf->byte_stride.nk,
                   f_buf->byte_stride.c,
                   f_buf->byte_stride.h,
                   f_buf->byte_stride.w,
                   o_buf->byte_stride.nk,
                   o_buf->byte_stride.c,
                   o_buf->byte_stride.h,
                   o_buf->byte_stride.w);

        if(elapsed != nullptr)
        {
            cur_time = handle.GetKernelTime();
            *elapsed += cur_time;
            MIOPEN_LOG_I2(cur_kernel.GetName() << ": " << cur_time);
        }
        else
        {
            if(handle.IsProfilingEnabled())
            {
                if(!(cur_kernel.GetName() ==
                     solver::ConvWinograd3x3MultipassWrW<WinoDataH,
                                                         WinoFilterH,
                                                         WinoDataW,
                                                         WinoFilterW>::GetSolverKernelNames(2)))
                {
                    total_time += handle.GetKernelTime();
                }
                else
                {
                    handle.AccumKernelTime(total_time);
                }
            }
        }
    }
#else
    (void)handle;
    (void)ctx;
    (void)tensors;
    (void)workSpace;
    (void)kernels;
    (void)pad_H;
    (void)pad_W;
    if(elapsed != nullptr)
    {
        *elapsed = 0;
    }
    MIOPEN_THROW(miopenStatusBadParm, "MixedWrW3x3Winograd Unsupported ");
#endif
}

// ConvolutionBackwardWeightsGetWorkSpaceSize
// FindBackwardWeightsAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdWeightsAlgorithm(Handle& handle,
                                                        const TensorDescriptor& dyDesc,
                                                        ConstData_t dy,
                                                        const TensorDescriptor& xDesc,
                                                        ConstData_t x,
                                                        const TensorDescriptor& dwDesc,
                                                        Data_t dw,
                                                        const int requestAlgoCount,
                                                        int* const returnedAlgoCount,
                                                        miopenConvAlgoPerf_t* perfResults,
                                                        Data_t workSpace,
                                                        size_t workSpaceSize,
                                                        bool exhaustiveSearch) const
{
    MIOPEN_LOG_I("requestAlgoCount = " << requestAlgoCount << ", workspace = " << workSpaceSize);
    if(x == nullptr || dw == nullptr || dy == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");
    if(xDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    *returnedAlgoCount = 0;

    AutoEnableProfiling enableProfiling{handle};

    auto problem =
        ProblemDescription{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};

    // < algorith_name, <time, workspace_size> >
    auto perf_db = UserFindDbRecord::TryLoad(handle, problem, [&](DbRecord& record) {

#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}) &&
           !(IsAnyBufferBF16(xDesc, dyDesc, dwDesc) && !IsUseRocBlas))
        {
            const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

            ValidateGroupCount(xDesc, dwDesc, *this);

            std::size_t in_n, in_c;
            std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

            auto in_spatial =
                boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + GetSpatialDimension());
            auto wei_spatial =
                boost::adaptors::slice(dwDesc.GetLengths(), 2, 2 + GetSpatialDimension());
            auto out_spatial =
                boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + GetSpatialDimension());

            size_t workspace_req =
                BackwardWeightsGetWorkSpaceSizeGEMM(dyDesc, dwDesc) * group_count;

            float time_gemm = 0;

            // if not 1x1
            if((miopen::any_of(wei_spatial, [](auto v) { return v != 1; }) ||
                miopen::any_of(GetConvPads(), [](auto v) { return v != 0; }) ||
                miopen::any_of(GetConvStrides(), [](auto v) { return v != 1; })) &&
               (workSpace != nullptr && workSpaceSize >= workspace_req))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                }
                float time_im2col = 0;
                int in_offset     = 0;
                time_im2col       = Im2ColGPU(handle,
                                        GetSpatialDimension(),
                                        x,
                                        in_offset,
                                        in_c,
                                        in_spatial,
                                        wei_spatial,
                                        out_spatial,
                                        GetConvPads(),
                                        GetConvStrides(),
                                        GetConvDilations(),
                                        workSpace,
                                        dyDesc.GetType());

                // dw = dy * transpose(Im2Col(x))
                GemmDescriptor gemm_desc =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count)
                        : CreateGemmDescriptorConvBwdWeight(dyDesc, xDesc, dwDesc);

                auto kcache_key = FindDbKCacheKey{};

                miopenStatus_t gemm_status = CallGemmTimeMeasure(
                    handle,
                    gemm_desc,
                    dy,
                    0,
                    workSpace,
                    0,
                    dw,
                    0,
                    &kcache_key,
                    time_precision,
                    group_count > 1 ? callGemmStridedBatched : callGemm,
                    group_count > 1 ? GemmBackend_t::rocblas : GemmBackend_t::miopengemm);

                time_gemm = in_n * (time_im2col + handle.GetKernelTime());

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues("miopenConvolutionBwdWeightsAlgoGEMM",
                                     FindDbData{
                                         "gemm", time_gemm, workspace_req, kcache_key,
                                     });
            }
            // 1x1 does not require im2col or workspace
            else if(miopen::any_of(wei_spatial, [](auto v) { return v == 1; }) &&
                    miopen::any_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                    miopen::any_of(GetConvStrides(), [](auto v) { return v == 1; }))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1");
                }

                // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                GemmDescriptor gemm_desc =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count)
                        : CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(dyDesc, xDesc, dwDesc);

                auto kcache_key = FindDbKCacheKey{};

                miopenStatus_t gemm_status = CallGemmTimeMeasure(
                    handle,
                    gemm_desc,
                    dy,
                    0,
                    x,
                    0,
                    dw,
                    0,
                    &kcache_key,
                    time_precision,
                    group_count > 1 ? callGemmStridedBatched : callGemmStridedBatchedSequential,
                    group_count > 1 ? GemmBackend_t::rocblas : GemmBackend_t::miopengemm);

                time_gemm = handle.GetKernelTime();
                if(group_count > 1)
                    time_gemm *= in_n;

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues("miopenConvolutionBwdWeightsAlgoGEMM",
                                     FindDbData{
                                         "gemm", time_gemm, 0, kcache_key,
                                     });
            }
        }
#endif
        ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
        bufs.SetWrW(x, dw, dy);
        auto ctx =
            ConvolutionContext{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
        ctx.do_search = exhaustiveSearch;
        ctx.SetStream(&handle);
        ctx.SetBufs(bufs);
        ctx.SetupFloats();
        ctx.DetectRocm();
        const auto network_config = ctx.BuildConfKey();
        const auto invoke_ctx =
            conv::WrWInvokeParams{{dyDesc, dy, xDesc, x, dwDesc, dw}, workSpace, workSpaceSize};
        // direct convolution
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
        {
            const auto all            = FindAllBwdWrW2DSolutions(ctx);
            const auto algorithm_name = AlgorithmName{"miopenConvolutionBwdWeightsAlgoDirect"};
            EvaluateInvokers(handle, all, algorithm_name, network_config, invoke_ctx, record);
        }

        try
        {
            const auto all = miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{})
                                 ? std::vector<miopen::solver::ConvSolution>()
                                 : FindWinogradWrWAllSolutions(ctx);

            float elapsed = 0.0f;
            if(!all.empty())
            {
                float best = std::numeric_limits<float>::max();
                miopen::solver::ConvSolution selected{miopenStatusUnknownError};
                for(const auto& sol : all)
                {
                    elapsed = 0.0f;
                    std::vector<KernelInvoke> kernels;

                    AddKernels(handle,
                               "miopenConvolutionBwdWeightsAlgoWinograd",
                               network_config,
                               sol,
                               &kernels);
                    auto tensors = ConvWrwTensors{dyDesc, dy, xDesc, x, dwDesc, dw};
                    if(workSpaceSize < sol.workspce_sz)
                        continue;
                    // clang-format off
                    if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<3, 2>()))
                        EvaluateWinograd3x3MultipassWrW<3,2>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<3, 3>()))
                        EvaluateWinograd3x3MultipassWrW<3,3>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<3, 4>()))
                        EvaluateWinograd3x3MultipassWrW<3,4>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<3, 5>()))
                        EvaluateWinograd3x3MultipassWrW<3,5>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<3, 6>()))
                        EvaluateWinograd3x3MultipassWrW<3,6>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<7, 2>()))
                        EvaluateWinograd3x3MultipassWrW<7,2>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<7, 3>()))
                        EvaluateWinograd3x3MultipassWrW<7,3>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>()))
                        EvaluateWinograd3x3MultipassWrW<7,2,1,1>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<7, 3, 1, 1>()))
                        EvaluateWinograd3x3MultipassWrW<7,3,1,1>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<1, 1, 7, 2>()))
                        EvaluateWinograd3x3MultipassWrW<1,1,7,2>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<1, 1, 7, 3>()))
                        EvaluateWinograd3x3MultipassWrW<1,1,7,3>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<5, 3>()))
                        EvaluateWinograd3x3MultipassWrW<5,3>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);
                    else if(sol.solver_id == SolverDbId(miopen::solver::ConvWinograd3x3MultipassWrW<5, 4>()))
                        EvaluateWinograd3x3MultipassWrW<5,4>(
                            handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1],&elapsed);

                    else // clang-format on
                    {    // single pass
                        int unused                       = 0;
                        using dataType                   = float;
                        static const int F_FLIP_K_C      = 1 << 2;
                        static const int F_NKC_STRIDES   = 1 << 9;
                        static const int F_GROUP_STRIDES = 1 << 10;
                        int reserved                     = 0;
                        int* reserved_ptr                = nullptr;
                        int pad_H                        = GetConvPads()[0];
                        int pad_W                        = GetConvPads()[1];
                        // clang-format off
                        int N, C, H, W, K, n_groups, out_H, out_W, R, S;
                        // clang-format on
                        if(kernels[0].GetName().rfind("miopenSp3AsmConv_group_20_5_23_M", 0) == 0)
                        {
                            GetCompiledInParameters(ctx,
                                                    &C,
                                                    &K,
                                                    &R,
                                                    &S,
                                                    &N,
                                                    &n_groups,
                                                    &H,
                                                    &W,
                                                    &out_H,
                                                    &out_W,
                                                    &unused,
                                                    &unused);
                            // GetCompiledInParameters(
                            // ctx, &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W, &R, &S, &pad_H,
                            // &pad_W);
                            int flags      = F_NKC_STRIDES + F_GROUP_STRIDES;
                            auto group_cnt = ctx.group_counts;
                            N              = N / group_cnt;
                            K              = K / group_cnt;

                            BuffInfo d_buf(
                                GetGroupConvLayout(
                                    GetSwappedNCLayout(GetMemLayout_t(ctx.in_layout)), true),
                                N,
                                C,
                                H,
                                W,
                                1,
                                group_cnt,
                                GetTypeSize(ctx.in_data_type)),
                                o_buf(
                                    GetGroupConvLayout(
                                        GetSwappedNCLayout(GetMemLayout_t(ctx.out_layout)), false),
                                    N,
                                    K,
                                    out_H,
                                    out_W,
                                    1,
                                    group_cnt,
                                    GetTypeSize(ctx.out_data_type)),
                                f_buf(
                                    GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true),
                                    K,
                                    C,
                                    R,
                                    S,
                                    1,
                                    group_cnt,
                                    GetTypeSize(ctx.weights_data_type));

                            if(GetKernelLocalWorkDim(kernels[0], 0) != 0)
                                n_groups = solver::ConvBinWinogradRxSf2x3::GetNGroups(
                                    ctx.group_counts,
                                    GetKernelGlobalWorkDim(kernels[0], 0) /
                                        GetKernelLocalWorkDim(kernels[0], 0));
                            else
                                n_groups = solver::ConvBinWinogradRxSf2x3::GetNGroups(
                                    ctx.group_counts,
                                    GetKernelGlobalWorkDim(kernels[0], 0) /
                                        512); // For OCL runtime. Issue #1724

                            // clang-format off
                            MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                                << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                                << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                                << " d_buf.byte_stride.nk=" << d_buf.byte_stride.nk << " d_buf.byte_stride.c=" << d_buf.byte_stride.c
                                << " d_buf.byte_stride.h=" << d_buf.byte_stride.h << " d_buf.byte_stride.w=" << d_buf.byte_stride.w
                                << " f_buf.byte_stride.nk=" << f_buf.byte_stride.nk << " f_buf.byte_stride.c=" << f_buf.byte_stride.c
                                << " f_buf.byte_stride.h=" << f_buf.byte_stride.h << " f_buf.byte_stride.w=" << f_buf.byte_stride.w
                                << " o_buf.byte_stride.nk=" << o_buf.byte_stride.nk << " o_buf.byte_stride.c=" << o_buf.byte_stride.c
                                << " o_buf.byte_stride.h="  << o_buf.byte_stride.h <<  " o_buf.byte_stride.w=" << o_buf.byte_stride.w
                                << " d_buf.byte_stride.g=" << d_buf.byte_stride.g  << " o_buf.byte_stride.g="  << o_buf.byte_stride.g
                                << " f_buf.byte_stride.g=" << f_buf.byte_stride.g); // clang-format on
                            MIOPEN_LOG_I2(" ctx.batch_sz=" << ctx.batch_sz << "ctx.n_inputs="
                                                           << ctx.n_inputs);
                            elapsed = 0;

                            kernels[0](N,
                                       C,
                                       H,
                                       W,
                                       K,
                                       n_groups,
                                       flags,
                                       reserved,
                                       x,
                                       dy,
                                       dw,
                                       reserved_ptr, // Unused return_addr.
                                       R,
                                       S,
                                       pad_H, // Like Fwd wino.
                                       pad_W,
                                       out_H,
                                       out_W,
                                       reserved_ptr, // Unused bias_addr.
                                       reserved,     // Unused relu_alpha.
                                       d_buf.byte_stride.nk,
                                       d_buf.byte_stride.c,
                                       d_buf.byte_stride.h,
                                       d_buf.byte_stride.w,
                                       f_buf.byte_stride.nk,
                                       f_buf.byte_stride.c,
                                       f_buf.byte_stride.h,
                                       f_buf.byte_stride.w,
                                       o_buf.byte_stride.nk,
                                       o_buf.byte_stride.c,
                                       o_buf.byte_stride.h,
                                       o_buf.byte_stride.w,
                                       group_cnt,
                                       d_buf.byte_stride.g,
                                       f_buf.byte_stride.g,
                                       o_buf.byte_stride.g);
                            elapsed = handle.GetKernelTime();
                        }
                        else // miopenSp3AsmConvRxSf3x2 and other
                        {
                            // clang-format off
                            GetCompiledInParameters(ctx, &N,&K,&out_H,&out_W,
                                &C,&n_groups,&H,&W,&R,&S,&unused,&unused);
                            // clang-format on
                            int flags      = F_FLIP_K_C + F_NKC_STRIDES;
                            int d_N_stride = H * W * static_cast<int>(sizeof(dataType));
                            int d_C_stride = C * d_N_stride;
                            int f_K_stride = out_H * out_W * static_cast<int>(sizeof(dataType));
                            int f_C_stride = K * f_K_stride;
                            int o_N_stride = R * S * static_cast<int>(sizeof(dataType));
                            int o_K_stride = C * o_N_stride;

                            // clang-format off
                            MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                                << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                                << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                                << " d_N_stride=" << d_N_stride << " d_C_stride=" << d_C_stride
                                << " f_K_stride=" << f_K_stride << " f_C_stride=" << f_C_stride
                                << " o_N_stride=" << o_N_stride << " o_K_stride=" << o_K_stride); // clang-format on
                            elapsed = 0;

                            kernels[0](C,
                                       N,
                                       H,
                                       W,
                                       K,
                                       n_groups,
                                       flags,
                                       reserved,
                                       x,
                                       dy,
                                       dw,
                                       reserved_ptr, // Unused return_addr.
                                       out_H,
                                       out_W,
                                       pad_H, // Like Fwd wino.
                                       pad_W,
                                       R,
                                       S,
                                       reserved_ptr, // Unused bias_addr.
                                       reserved,     // Unused relu_alpha.
                                       d_N_stride,
                                       d_C_stride,
                                       f_K_stride,
                                       f_C_stride,
                                       o_N_stride,
                                       o_K_stride);
                            elapsed = handle.GetKernelTime();
                        }
                    } ////single pass end
                    MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ")
                                     << best);
                    if(elapsed < best)
                    {
                        best     = elapsed;
                        selected = sol;
                    }
                }
                if(selected.Succeeded())
                {
                    const std::string algorithm_name = "miopenConvolutionBwdWeightsAlgoWinograd";
                    AddKernels(handle, algorithm_name, network_config, selected, nullptr);
                    MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                              << selected.workspce_sz);
                    record.SetValues(algorithm_name,
                                     FindDbData{
                                         selected.solver_id,
                                         best,
                                         selected.workspce_sz,
                                         {algorithm_name, network_config},
                                     });
                }
            }
        }
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_WE("Find Winograd WrW failed:" << ex.what());
        }

        // Implicit GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{}))
        {
            const auto all = FindImplicitGemmWrWAllSolutions(ctx);
            float best     = std::numeric_limits<float>::max();
            miopen::solver::ConvSolution selected{miopenStatusUnknownError};
            const auto algo_name = "miopenConvolutionBwdWeightsAlgoImplicitGEMM";
            float elapsed        = 0.0f;
            for(const auto& sol : all)
            {
                std::vector<KernelInvoke> kernels;
                AddKernels(handle, algo_name, network_config, sol, &kernels);
                if(!kernels.empty())
                {
                    auto kernel = kernels[0];

                    // For fp16/bfp16 backward data case, do zero init, bwd data with fp32 output
                    // and cast from fp32 to fp16/bfp16
                    // clang-format off
                    if((dwDesc.GetType() == miopenHalf || dwDesc.GetType() == miopenBFloat16) &&
                       (kernel.GetName() == "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer" ||
                        kernel.GetName() == "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer"))
                    // clang-format on
                    {
                        float zero = 0.f;
                        TensorDescriptor workSpaceDesc(
                            miopenFloat, dwDesc.GetLengths(), dwDesc.GetStrides());
                        SetTensor(handle, workSpaceDesc, workSpace, &zero);
                        elapsed = handle.GetKernelTime();

                        kernel(x, dy, workSpace);
                        elapsed += handle.GetKernelTime();

                        CastTensor(handle, &lowp_quant, workSpaceDesc, workSpace, dwDesc, dw, 0, 0);
                        elapsed += handle.GetKernelTime();
                    }
                    else
                    {
                        // this kernel may accumulate results into input tensor, therefore need to
                        // set zero
                        float zero = 0.f;
                        SetTensor(handle, dwDesc, dw, &zero);
                        elapsed = handle.GetKernelTime();

                        kernel(x, dy, dw);
                        elapsed += handle.GetKernelTime();
                    }
                }

                MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ") << best);

                if(elapsed < best)
                {
                    best     = elapsed;
                    selected = sol;
                }
            }
            if(selected.Succeeded())
            {
                AddKernels(handle, algo_name, network_config, selected, nullptr);
                MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                          << selected.workspce_sz);
                record.SetValues(algo_name,
                                 FindDbData{selected.solver_id,
                                            best,
                                            selected.workspce_sz,
                                            {algo_name, network_config}});
            }
        }
    });

    if(perf_db.empty())
        MIOPEN_THROW("Bwd Weights Convolution cannot be executed due to incorrect params");

    std::sort(begin(perf_db), end(perf_db));

    for(const auto& entry : perf_db)
        MIOPEN_LOG_I(entry.name << "\t" << entry.time << "\t" << entry.workspace);

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(perf_db.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].bwd_weights_algo = StringToConvolutionBwdWeightsAlgo(perf_db[i].name);
        perfResults[i].time             = perf_db[i].time;
        perfResults[i].memory           = perf_db[i].workspace;
    }
    MIOPEN_LOG_I("BWrW Chosen Algorithm: " << perf_db[0].solver_id << " , " << perf_db[0].workspace
                                           << ", "
                                           << perf_db[0].time);
}

static void ConvWrwCheckNumerics(Handle& handle,
                                 const ConvWrwTensors& tensors,
                                 const void* beta,
                                 std::function<void()>&& worker)
{
    if(!miopen::CheckNumericsEnabled())
    {
        worker();
        return;
    }

    miopen::checkNumericsInput(handle, tensors.dyDesc, tensors.dy);
    miopen::checkNumericsInput(handle, tensors.xDesc, tensors.x);
    if(!float_equal(*(static_cast<const float*>(beta)), 0))
        miopen::checkNumericsInput(handle, tensors.dwDesc, tensors.dw);

    worker();

    miopen::checkNumericsOutput(handle, tensors.dwDesc, tensors.dw);
}

// BackwardWeightsAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardWeights(Handle& handle,
                                                       const void* alpha,
                                                       const TensorDescriptor& dyDesc,
                                                       ConstData_t dy,
                                                       const TensorDescriptor& xDesc,
                                                       ConstData_t x,
                                                       miopenConvBwdWeightsAlgorithm_t algo,
                                                       const void* beta,
                                                       const TensorDescriptor& dwDesc,
                                                       Data_t dw,
                                                       Data_t workSpace,
                                                       size_t workSpaceSize) const
{
    MIOPEN_LOG_I("algo = " << algo << ", workspace = " << workSpaceSize);
    auto tensors = ConvWrwTensors{dyDesc, dy, xDesc, x, dwDesc, dw};
    ValidateConvTensors(tensors);
    ValidateAlphaBeta(alpha, beta);

    if(xDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    ConvWrwCheckNumerics(handle, tensors, beta, [&]() {
        ValidateGroupCount(xDesc, dwDesc, *this);

        const auto algorithm_name = [=]() {
            switch(algo)
            {
            case miopenConvolutionBwdWeightsAlgoGEMM:
                return AlgorithmName{"miopenConvolutionBwdWeightsAlgoGEMM"};
            case miopenConvolutionBwdWeightsAlgoDirect:
                return AlgorithmName{"miopenConvolutionBwdWeightsAlgoDirect"};
            case miopenConvolutionBwdWeightsAlgoWinograd:
                return AlgorithmName{"miopenConvolutionBwdWeightsAlgoWinograd"};
            case miopenConvolutionBwdWeightsAlgoImplicitGEMM:
                return AlgorithmName{"miopenConvolutionBwdWeightsAlgoImplicitGEMM"};
            }
            MIOPEN_THROW("Unknown conv wrw algorigthm: " + std::to_string(algo));
        }();

        auto ctx =
            ConvolutionContext{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
        ctx.SetStream(&handle);
        const auto network_config = ctx.BuildConfKey();
        const auto& invoker       = handle.GetInvoker(network_config, boost::none, algorithm_name);

        if(invoker)
        {
            const auto& invoke_ctx = conv::WrWInvokeParams{tensors, workSpace, workSpaceSize};
            (*invoker)(handle, invoke_ctx);
            return;
        }

        switch(algo)
        {
        case miopenConvolutionBwdWeightsAlgoDirect:
            MIOPEN_THROW("No invoker was registered for convolution weights. Was find executed?");
        case miopenConvolutionBwdWeightsAlgoGEMM:
            BackwardWeightsGemm(handle, tensors, workSpace, workSpaceSize);
            break;
        case miopenConvolutionBwdWeightsAlgoWinograd:
        {
            auto&& kernels = handle.GetKernels(algorithm_name, network_config);
            if(kernels.empty())
                MIOPEN_THROW("Error running Winograd WrW. Was Find() run previously?");
            BackwardWeightsWinograd(handle, ctx, tensors, workSpace, kernels);
        }
        break;
        case miopenConvolutionBwdWeightsAlgoImplicitGEMM:
        {
            auto&& kernels = handle.GetKernels(algorithm_name, network_config);
            if(kernels.empty())
                MIOPEN_THROW("Error running Implicit GEMM WrW. Was Find() run previously?");

            auto kernel = kernels[0];

            // For fp16/bfp16 backward data case, do zero init, bwd data with fp32 output
            // and cast from fp32 to fp16/bfp16
            // clang-format off
            if((dwDesc.GetType() == miopenHalf || dwDesc.GetType() == miopenBFloat16) &&
               (kernel.GetName() == "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer" ||
                kernel.GetName() == "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer"))
            // clang-format on
            {
                float zero = 0.f;
                TensorDescriptor workSpaceDesc(
                    miopenFloat, dwDesc.GetLengths(), dwDesc.GetStrides());
                SetTensor(handle, workSpaceDesc, workSpace, &zero);
                kernel(x, dy, workSpace);
                CastTensor(handle, &lowp_quant, workSpaceDesc, workSpace, dwDesc, dw, 0, 0);
            }
            else
            {
                float zero = 0.f;
                SetTensor(handle, dwDesc, dw, &zero);
                kernel(x, dy, dw);
            }
        }
        }
    });
}

void ConvolutionDescriptor::BackwardWeightsGemm(Handle& handle,
                                                const ConvWrwTensors& tensors,
                                                Data_t workSpace,
                                                std::size_t workSpaceSize) const
{
#if MIOPEN_USE_GEMM
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
    {
        MIOPEN_THROW("GEMM convolution is disabled");
    }
    if(IsAnyBufferBF16(tensors.xDesc, tensors.dyDesc, tensors.dwDesc) && !IsUseRocBlas)
    {
        MIOPEN_THROW("GEMM convolution is unsupported");
    }

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(tensors.xDesc.GetLengths());

    std::size_t wei_k = tensors.dwDesc.GetLengths()[0];

    auto in_spatial =
        boost::adaptors::slice(tensors.xDesc.GetLengths(), 2, 2 + GetSpatialDimension());
    auto wei_spatial =
        boost::adaptors::slice(tensors.dwDesc.GetLengths(), 2, 2 + GetSpatialDimension());
    auto out_spatial =
        boost::adaptors::slice(tensors.dyDesc.GetLengths(), 2, 2 + GetSpatialDimension());

    // Zeroing out the output buffer
    float zero = 0.0f;
    SetTensor(handle, tensors.dwDesc, tensors.dw, &zero);

    handle.ResetKernelTime();
    float time_0 = 0;
    if((miopen::any_of(wei_spatial, [](auto v) { return v != 1; }) ||
        miopen::any_of(GetConvPads(), [](auto v) { return v != 0; }) ||
        miopen::any_of(GetConvStrides(), [](auto v) { return v != 1; })))
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, non 1x1");
        }
        assert(workSpace != nullptr &&
               workSpaceSize >=
                   (BackwardWeightsGetWorkSpaceSizeGEMM(tensors.dyDesc, tensors.dwDesc) *
                    group_count));

        std::size_t out_spatial_size = std::accumulate(
            out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        std::size_t in_spatial_size = std::accumulate(
            in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

        float t1 = 0;

        for(std::size_t i = 0; i < in_n; i++)
        {
            std::size_t out_offset = i * wei_k * out_spatial_size;

            std::size_t in_offset = i * in_c * in_spatial_size;

            Im2ColGPU(handle,
                      GetSpatialDimension(),
                      tensors.x,
                      in_offset,
                      in_c,
                      in_spatial,
                      wei_spatial,
                      out_spatial,
                      GetConvPads(),
                      GetConvStrides(),
                      GetConvDilations(),
                      workSpace,
                      tensors.dyDesc.GetType());

            if(handle.IsProfilingEnabled())
                t1 = handle.GetKernelTime();

            if(group_count > 1)
            {
                GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvBwdWeight(
                    tensors.dyDesc, tensors.xDesc, tensors.dwDesc, group_count);
                CallGemmStridedBatched(handle,
                                       gemm_desc,
                                       tensors.dy,
                                       out_offset,
                                       workSpace,
                                       0,
                                       tensors.dw,
                                       0,
                                       nullptr,
                                       false);
            }
            else
            {
                // tensors.dw = tensors.dy * transpose(Im2Col(tensors.x))
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvBwdWeight(
                    tensors.dyDesc, tensors.xDesc, tensors.dwDesc);

                // dw = dy * transpose(Im2Col(x))
                CallGemm(handle,
                         gemm_desc,
                         tensors.dy,
                         out_offset,
                         workSpace,
                         0,
                         tensors.dw,
                         0,
                         nullptr,
                         false,
                         GemmBackend_t::miopengemm);
            }
            // Update times for both the kernels
            if(handle.IsProfilingEnabled())
            {
                if(i == in_n - 1)
                    handle.AccumKernelTime(t1 + time_0);
                else
                    handle.AccumKernelTime(t1);
                time_0 += handle.GetKernelTime();
            }
        }
    }
    else if(miopen::any_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::any_of(GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::any_of(GetConvStrides(), [](auto v) { return v == 1; }))
    {
        if(group_count > 1)
        {
            MIOPEN_LOG_FUNCTION("groupconv, 1x1");

            GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvBwdWeight(
                tensors.dyDesc, tensors.xDesc, tensors.dwDesc, group_count);

            std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                           out_spatial.end(),
                                                           std::size_t(1),
                                                           std::multiplies<std::size_t>());

            std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                          in_spatial.end(),
                                                          std::size_t(1),
                                                          std::multiplies<std::size_t>());

            for(std::size_t i = 0; i < in_n; i++)
            {
                std::size_t out_offset = i * wei_k * out_spatial_size;

                std::size_t in_offset = i * in_c * in_spatial_size;

                CallGemmStridedBatched(handle,
                                       gemm_desc,
                                       tensors.dy,
                                       out_offset,
                                       tensors.x,
                                       in_offset,
                                       tensors.dw,
                                       0,
                                       nullptr,
                                       false);

                if(handle.IsProfilingEnabled())
                {
                    if(i == in_n - 1)
                        handle.AccumKernelTime(time_0);
                    time_0 += handle.GetKernelTime();
                }
            }
        }
        else
        {
            MIOPEN_LOG_FUNCTION("convolution, 1x1");

            // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
            GemmDescriptor gemm_desc = CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(
                tensors.dyDesc, tensors.xDesc, tensors.dwDesc);

            // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
            CallGemmStridedBatchedSequential(handle,
                                             gemm_desc,
                                             tensors.dy,
                                             0,
                                             tensors.x,
                                             0,
                                             tensors.dw,
                                             0,
                                             nullptr,
                                             false,
                                             GemmBackend_t::miopengemm);
        }
    }

#ifdef NDEBUG
    std::ignore = workSpaceSize;
#endif
#else
    std::ignore = handle;
    std::ignore = tensors;
    std::ignore = workSpace;
    std::ignore = workSpaceSize;
    MIOPEN_THROW("GEMM is not supported");
#endif
}

template <class TKernels>
void ConvolutionDescriptor::BackwardWeightsWinograd(Handle& handle,
                                                    const ConvolutionContext& ctx,
                                                    const ConvWrwTensors& tensors,
                                                    Data_t workSpace,
                                                    const TKernels& kernels) const
{
    if(kernels.size() > 1)
    {
        auto kernel_1 = kernels.front();
        if(kernel_1.GetName() == solver::ConvWinograd3x3MultipassWrW<3, 2>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<3, 2>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<3, 3>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<3, 3>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<3, 4>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<3, 4>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<3, 5>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<3, 5>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<3, 6>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<3, 6>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<7, 2>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<7, 2>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<7, 3>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<7, 3>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<7, 2, 1, 1>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<7, 3, 1, 1>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<7, 3, 1, 1>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<1, 1, 7, 2>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<1, 1, 7, 2>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<1, 1, 7, 3>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<1, 1, 7, 3>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<5, 3>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<5, 3>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
        else if(kernel_1.GetName() ==
                solver::ConvWinograd3x3MultipassWrW<5, 4>::GetSolverKernelNames(0))
            EvaluateWinograd3x3MultipassWrW<5, 4>(
                handle, ctx, tensors, workSpace, kernels, GetConvPads()[0], GetConvPads()[1]);
    }
    else
    { // single pass
        static const int F_FLIP_K_C      = 1 << 2;
        static const int F_NKC_STRIDES   = 1 << 9;
        static const int F_GROUP_STRIDES = 1 << 10;
        int flags                        = F_FLIP_K_C + F_NKC_STRIDES;
        int reserved                     = 0;
        int* reserved_ptr                = nullptr;
        int pad_H                        = GetConvPads()[0];
        int pad_W                        = GetConvPads()[1];

        int N, C, H, W, K, n_groups, out_H, out_W, R, S, unused;

        if(kernels[0].GetName().rfind("miopenSp3AsmConv_group_20_5_23_M", 0) == 0)
        {
            GetCompiledInParameters(
                ctx, &C, &K, &R, &S, &N, &n_groups, &H, &W, &out_H, &out_W, &unused, &unused);
            // GetCompiledInParameters(
            //  ctx, &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W, &R, &S, &pad_H, &pad_W);

            flags          = F_NKC_STRIDES + F_GROUP_STRIDES;
            auto group_cnt = ctx.group_counts;
            N              = N / group_cnt;
            K              = K / group_cnt;

            // cppcheck-suppress unreadVariable
            BuffInfo d_buf(
                GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(ctx.in_layout)), true),
                N,
                C,
                H,
                W,
                1,
                group_cnt,
                GetTypeSize(ctx.in_data_type)),
                // cppcheck-suppress unreadVariable
                o_buf(GetGroupConvLayout(GetSwappedNCLayout(GetMemLayout_t(ctx.out_layout)), false),
                      N,
                      K,
                      out_H,
                      out_W,
                      1,
                      group_cnt,
                      GetTypeSize(ctx.out_data_type)),
                // cppcheck-suppress unreadVariable
                f_buf(GetGroupConvLayout(GetSwappedNCLayout(MemLayout_t::NCHW), true),
                      K,
                      C,
                      R,
                      S,
                      1,
                      group_cnt,
                      GetTypeSize(ctx.weights_data_type));

            if(GetKernelLocalWorkDim(kernels[0], 0) != 0)
                n_groups = solver::ConvBinWinogradRxSf2x3::GetNGroups(
                    ctx.group_counts,
                    GetKernelGlobalWorkDim(kernels[0], 0) / GetKernelLocalWorkDim(kernels[0], 0));
            else
                n_groups = solver::ConvBinWinogradRxSf2x3::GetNGroups(
                    ctx.group_counts,
                    GetKernelGlobalWorkDim(kernels[0], 0) / 512); // For OCL runtime. Issue #1724

            // clang-format off
            MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                << " d_buf.byte_stride.nk=" << d_buf.byte_stride.nk << " d_buf.byte_stride.c=" << d_buf.byte_stride.c
                << " d_buf.byte_stride.h=" << d_buf.byte_stride.h << " d_buf.byte_stride.w=" << d_buf.byte_stride.w
                << " f_buf.byte_stride.nk=" << f_buf.byte_stride.nk << " f_buf.byte_stride.c=" << f_buf.byte_stride.c
                << " f_buf.byte_stride.h=" << f_buf.byte_stride.h << " f_buf.byte_stride.w=" << f_buf.byte_stride.w
                << " o_buf.byte_stride.nk=" << o_buf.byte_stride.nk << " o_buf.byte_stride.c=" << o_buf.byte_stride.c
                << " o_buf.byte_stride.h="  << o_buf.byte_stride.h <<  " o_buf.byte_stride.w=" << o_buf.byte_stride.w
                << " d_buf.byte_stride.g=" << d_buf.byte_stride.g
                << " f_buf.byte_stride.g=" << f_buf.byte_stride.g
                << " o_buf.byte_stride.g=" << o_buf.byte_stride.g); // clang-format on

            kernels[0](N,
                       C,
                       H,
                       W,
                       K,
                       n_groups,
                       flags,
                       reserved,
                       tensors.x,
                       tensors.dy,
                       tensors.dw,
                       reserved_ptr, // Unused return_addr.
                       R,
                       S,
                       pad_H, // Like Fwd wino.
                       pad_W,
                       out_H,
                       out_W,
                       reserved_ptr, // Unused bias_addr.
                       reserved,     // Unused relu_alpha.
                       d_buf.byte_stride.nk,
                       d_buf.byte_stride.c,
                       d_buf.byte_stride.h,
                       d_buf.byte_stride.w,
                       f_buf.byte_stride.nk,
                       f_buf.byte_stride.c,
                       f_buf.byte_stride.h,
                       f_buf.byte_stride.w,
                       o_buf.byte_stride.nk,
                       o_buf.byte_stride.c,
                       o_buf.byte_stride.h,
                       o_buf.byte_stride.w,
                       group_cnt,
                       d_buf.byte_stride.g,
                       f_buf.byte_stride.g,
                       o_buf.byte_stride.g);
        }
        else
        {
            auto kernel = kernels.front();
            // For bwd & wrw inputs and outputs reside in k_p in reversed order.
            GetCompiledInParameters(
                ctx, &N, &K, &out_H, &out_W, &C, &n_groups, &H, &W, &R, &S, &unused, &unused);
            using dataType = float;
            int d_N_stride = H * W * static_cast<int>(sizeof(dataType));
            int d_C_stride = C * d_N_stride;
            int f_K_stride = out_H * out_W * static_cast<int>(sizeof(dataType));
            int f_C_stride = K * f_K_stride;
            int o_N_stride = R * S * static_cast<int>(sizeof(dataType));
            int o_K_stride = C * o_N_stride;
            // clang-format off
            MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                << " d_N_stride=" << d_N_stride << " d_C_stride=" << d_C_stride
                << " f_K_stride=" << f_K_stride << " f_C_stride=" << f_C_stride
                << " o_N_stride=" << o_N_stride << " o_K_stride=" << o_K_stride ); // clang-format on
            kernel(C,
                   N,
                   H,
                   W,
                   K,
                   n_groups,
                   flags,
                   reserved,
                   tensors.x,
                   tensors.dy,
                   tensors.dw,
                   reserved_ptr,
                   out_H,
                   out_W,
                   pad_H,
                   pad_W,
                   R,
                   S,
                   reserved_ptr,
                   reserved,
                   d_N_stride,
                   d_C_stride,
                   f_K_stride,
                   f_C_stride,
                   o_N_stride,
                   o_K_stride);
        }
    }
    ////single pass end
}

ProblemDescription ConvolutionDescriptor::MakeWrwProblem(const TensorDescriptor& dyDesc,
                                                         const TensorDescriptor& xDesc,
                                                         const TensorDescriptor& dwDesc) const
{
    auto problem =
        ProblemDescription{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
    return problem;
}

std::size_t ConvolutionDescriptor::GetWrwSolutionCount(Handle& handle,
                                                       const TensorDescriptor& dyDesc,
                                                       const TensorDescriptor& xDesc,
                                                       const TensorDescriptor& dwDesc) const
{
    MIOPEN_LOG_I("");
    const auto problem = MakeWrwProblem(dyDesc, xDesc, dwDesc);
    const auto count   = GetSolutionCount(handle, problem);
    if(count > 0)
        return count;
    return GetWrwSolutionCountFallback(dyDesc, xDesc, dwDesc);
}

void ConvolutionDescriptor::GetWrwSolutions(Handle& handle,
                                            const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& xDesc,
                                            const TensorDescriptor& dwDesc,
                                            size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions) const
{
    MIOPEN_LOG_I("");
    if(solutionCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutionCount cannot be nullptr");
    if(solutions == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutions cannot be nullptr");

    const auto problem = MakeWrwProblem(dyDesc, xDesc, dwDesc);
    GetSolutions(handle,
                 problem,
                 maxSolutionCount,
                 solutionCount,
                 solutions,
                 StringToConvolutionBwdWeightsAlgo);

    if(*solutionCount == 0)
        GetWrwSolutionsFallback(
            handle, dyDesc, xDesc, dwDesc, maxSolutionCount, solutionCount, solutions);
}

void ConvolutionDescriptor::CompileWrwSolution(Handle& handle,
                                               const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& dwDesc,
                                               solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString());
    auto ctx = ConvolutionContext{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
    ctx.SetStream(&handle);
    ctx.disable_search_enforce = true;

    CompileSolution(handle, solver_id, ctx, conv::Direction::BackwardWeights, [&]() {
        MIOPEN_THROW("FFT is not supported in WrW");
    });
}

std::size_t ConvolutionDescriptor::GetWrwSolutionWorkspaceSize(Handle& handle,
                                                               const TensorDescriptor& dyDesc,
                                                               const TensorDescriptor& xDesc,
                                                               const TensorDescriptor& dwDesc,
                                                               solver::Id solver_id) const
{
    MIOPEN_LOG_I2("solver_id = " << solver_id.ToString());
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm, "invalid solution id = " + solver_id.ToString());
    if(solver_id != solver::Id::gemm() && solver_id != solver::Id::fft())
    {
        auto sol = solver_id.GetSolver();
        auto problem =
            ProblemDescription{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
        auto ctx = ConvolutionContext{problem};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        if(sol.IsApplicable(ctx))
            return sol.GetWorkspaceSize(ctx);
        else
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "The supplied solution id: " + solver_id.ToString() +
                             " is not applicable to the current problem");
        }
    }
    return GetWrwSolutionWorkspaceSizeFallback(handle, dyDesc, xDesc, dwDesc, solver_id);
}

void ConvolutionDescriptor::ConvolutionWrwImmediate(Handle& handle,
                                                    const TensorDescriptor& dyDesc,
                                                    ConstData_t dy,
                                                    const TensorDescriptor& xDesc,
                                                    ConstData_t x,
                                                    const TensorDescriptor& dwDesc,
                                                    Data_t dw,
                                                    Data_t workSpace,
                                                    std::size_t workSpaceSize,
                                                    solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString() << ", workspace = " << workSpaceSize);
    auto tensors = ConvWrwTensors{dyDesc, dy, xDesc, x, dwDesc, dw};
    ValidateConvTensors(tensors);

    if(xDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    float beta = 0;
    ConvWrwCheckNumerics(handle, tensors, &beta, [&]() {
        ValidateGroupCount(xDesc, dwDesc, *this);

        auto ctx =
            ConvolutionContext{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
        ctx.SetStream(&handle);

        if(CheckInvokerSupport(solver_id, conv::Direction::BackwardWeights))
        {
            const auto invoker    = LoadOrPrepareInvoker(handle, ctx, solver_id);
            const auto invoke_ctx = conv::WrWInvokeParams{tensors, workSpace, workSpaceSize};
            invoker(handle, invoke_ctx);
            return;
        }

        if(solver_id == solver::Id::gemm())
        {
            BackwardWeightsGemm(handle, tensors, workSpace, workSpaceSize);
            return;
        }

        const auto network_config = ctx.BuildConfKey();
        auto algo_name            = solver_id.GetAlgo(conv::Direction::BackwardWeights);
        const auto&& chk_kernels  = handle.GetKernels(algo_name, network_config);
        auto v_chk_kernels = std::vector<KernelInvoke>{chk_kernels.begin(), chk_kernels.end()};
        if(!v_chk_kernels.empty())
        {
            MIOPEN_LOG_I2(
                "Found previously compiled kernels for solution: " << solver_id.ToString());

            if(algo_name == "miopenConvolutionBwdWeightsAlgoWinograd")
                BackwardWeightsWinograd(handle, ctx, tensors, workSpace, v_chk_kernels);
            else if(algo_name == "miopenConvolutionBwdWeightsAlgoImplicitGEMM")
                v_chk_kernels[0](x, dy, dw);
            else
                MIOPEN_THROW("Invalid algorithm: " + algo_name);
            return;
        }

        const FindDbRecord fdb_record{handle, ctx};

        for(const auto& pair : fdb_record)
        {
            if(solver::Id{pair.second.solver_id} != solver_id)
                continue;

            const auto&& kernels = handle.GetKernels(pair.second.kcache_key.algorithm_name,
                                                     pair.second.kcache_key.network_config);
            auto v_kernels = std::vector<KernelInvoke>{kernels.begin(), kernels.end()};

            if(v_kernels.empty())
                v_kernels = CompileSolver(handle, ctx, solver_id, pair.second.kcache_key);

            if(pair.second.kcache_key.algorithm_name == "miopenConvolutionBwdWeightsAlgoWinograd")
                BackwardWeightsWinograd(handle, ctx, tensors, workSpace, v_kernels);
            else if(pair.second.kcache_key.algorithm_name ==
                    "miopenConvolutionBwdWeightsAlgoImplicitGEMM")
                v_kernels[0](x, dy, dw);
            else
                MIOPEN_THROW("Invalid algorithm: " + pair.second.kcache_key.algorithm_name);
            return;
        }

        // Todo: solver not found in find-db.
        MIOPEN_THROW(miopenStatusNotImplemented);
    });
}

void ConvolutionBackwardBias(Handle& handle,
                             const void* alpha,
                             const TensorDescriptor& dyDesc,
                             ConstData_t dy,
                             const void* beta,
                             const TensorDescriptor& dbDesc,
                             Data_t db)
{
    if(dy == nullptr || db == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetLengths()[1] != dbDesc.GetLengths()[1])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, dyDesc, dy);
    }

    std::size_t out_n, out_k, stride_n, stride_k;
    std::tie(out_n, out_k)       = tie_pick<0, 1>()(dyDesc.GetLengths());
    std::tie(stride_n, stride_k) = tie_pick<0, 1>()(dyDesc.GetStrides());
    std::string program_name = "MIOpenConvBwdBias.cl";
    std::string kernel_name  = "MIOpenConvBwdB";

    std::string params;
    std::size_t lcl_grp_size0 = 256;
    std::size_t lcl_grp_size1 = 1;
    std::size_t local_mem_sz  = 256;

    std::size_t map_size = std::accumulate(dyDesc.GetLengths().begin() + 2,
                                           dyDesc.GetLengths().end(),
                                           std::size_t(1),
                                           std::multiplies<std::size_t>());
    std::size_t read_unit        = 4;
    std::size_t map_size_aligned = (map_size + (read_unit - 1)) / read_unit;
    std::size_t off_pix          = map_size - (map_size / read_unit) * read_unit;

    params = " -DMLO_CONVBWD_GROUP_SZ0=" + std::to_string(lcl_grp_size0);
    params += " -DMLO_CONVBWD_GROUP_SZ1=" + std::to_string(lcl_grp_size1);
    params += " -DMLO_CONVBWDB_LCL_MEMSZ=" + std::to_string(local_mem_sz);
    params += " -DMLO_CONVBWDB_UNITSIZE=" + std::to_string(read_unit);
    params += " -DMLO_OUT_BATCH_SZ=" + std::to_string(out_n);
    params += " -DMLO_OUT_CHANNEL_STRIDE=" + std::to_string(stride_k);
    params += " -DMLO_OUT_BATCH_STRIDE=" + std::to_string(stride_n);
    params += " -DMLO_WK_SIZE=" + std::to_string(map_size_aligned);
    params += " -DMLO_N_PIX_OFF=" + std::to_string(off_pix);

    params += GetDataTypeKernelParams(dyDesc.GetType());

    const std::vector<size_t> vld = {lcl_grp_size0, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {lcl_grp_size0, static_cast<size_t>(out_k), size_t{1}};

    handle.AddKernel("miopenConvolutionBwdBias", "", program_name, kernel_name, vld, vgd, params)(
        dy, db);

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dbDesc, db);
    }
}

} // namespace miopen
