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
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/env.hpp>
#include <miopen/find_db.hpp>
#include <miopen/finddb_kernel_cache_key.hpp>
#include <miopen/find_controls.hpp>
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
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMMED_FALLBACK)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_COMPILE_ONLY)

size_t GetKernelGlobalWorkDim(const KernelInvoke& kernel, int dim) { return kernel.gdims[dim]; }

size_t GetKernelLocalWorkDim(const KernelInvoke& kernel, int dim) { return kernel.ldims[dim]; }

static inline void AddKernels(const Handle& handle,
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
    ///\todo How make these validation clearly
    if(conv.group_count == 1)
    {
        if((((wDesc.GetLayout_t() == miopenTensorNCHW) ||
             (wDesc.GetLayout_t() == miopenTensorNCHWc4) ||
             (wDesc.GetLayout_t() == miopenTensorNCHWc8)) &&
            (xDesc.GetLengths()[1] != wDesc.GetLengths()[1])) ||
           ((wDesc.GetLayout_t() == miopenTensorCHWNc4 ||
             wDesc.GetLayout_t() == miopenTensorCHWNc8) &&
            (xDesc.GetLengths()[1] != wDesc.GetLengths()[0])))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
    if(conv.group_count > 1)
    {
        if(xDesc.GetLengths()[1] % conv.group_count != 0 ||
           conv.group_count > xDesc.GetLengths()[1] ||
           (((wDesc.GetLayout_t() == miopenTensorNCHW) ||
             (wDesc.GetLayout_t() == miopenTensorNCHWc4) ||
             (wDesc.GetLayout_t() == miopenTensorNCHWc8)) &&
            (wDesc.GetLengths()[0] % conv.group_count != 0 ||
             conv.group_count > wDesc.GetLengths()[0])) ||
           ((wDesc.GetLayout_t() == miopenTensorCHWNc4 ||
             wDesc.GetLayout_t() == miopenTensorCHWNc8) &&
            (wDesc.GetLengths()[3] % conv.group_count != 0 ||
             conv.group_count > wDesc.GetLengths()[3])))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if((((wDesc.GetLayout_t() == miopenTensorNCHW) ||
             (wDesc.GetLayout_t() == miopenTensorNCHWc4) ||
             (wDesc.GetLayout_t() == miopenTensorNCHWc8)) &&
            (xDesc.GetLengths()[1] / conv.group_count != wDesc.GetLengths()[1])) ||
           ((wDesc.GetLayout_t() == miopenTensorCHWNc4 ||
             wDesc.GetLayout_t() == miopenTensorCHWNc8) &&
            (xDesc.GetLengths()[1] / conv.group_count != wDesc.GetLengths()[0])))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindWinogradSolutions(const ConvolutionContext& ctx,
                                             const AnyInvokeParams& invoke_ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{}))
        return {};
    try
    {
        return FindAllWinogradSolutions(ctx, invoke_ctx);
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return {};
    }
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindDataGemmSolutions(const ConvolutionContext& ctx,
                                             const AnyInvokeParams& invoke_ctx) const
{
#if MIOPEN_USE_GEMM
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        return {};
    try
    {
        return FindAllGemmSolutions(ctx, invoke_ctx);
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return {};
    }
#else
    std::ignore = ctx;
    std::ignore = invoke_ctx;
    return {};
#endif
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindDataDirectSolutions(Handle& handle,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& wDesc,
                                               const TensorDescriptor& yDesc,
                                               bool exhaustiveSearch,
                                               bool isForward,
                                               const ConvolutionUserBuffers& bufs,
                                               const AnyInvokeParams& invoke_ctx) const
{

    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
        return {};

    const auto dir = isForward ? conv::Direction::Forward : conv::Direction::BackwardData;
    auto ctx       = ConvolutionContext{xDesc, wDesc, yDesc, *this, dir};
    ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
    ctx.do_search                  = exhaustiveSearch;
    ctx.save_srch_req              = true;
    ctx.general_compile_options    = "";
    ctx.SetStream(&handle);
    ctx.SetBufs(bufs);
    ctx.DetectRocm();
    ctx.SetupFloats();

    try
    {
        return FindAllDirectSolutions(ctx, invoke_ctx);
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
                                                     const ConvolutionUserBuffers& bufs,
                                                     const AnyInvokeParams& invoke_ctx) const
{

    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{}))
        return {};

    const auto dir = isForward ? conv::Direction::Forward : conv::Direction::BackwardData;
    auto ctx       = ConvolutionContext{xDesc, wDesc, yDesc, *this, dir};

    ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
    ctx.do_search                  = exhaustiveSearch;
    ctx.save_srch_req              = true;
    ctx.general_compile_options    = "";
    ctx.SetStream(&handle);
    ctx.SetBufs(bufs);
    ctx.DetectRocm();
    ctx.SetupFloats();

    try
    {
        return FindAllImplicitGemmSolutions(ctx, invoke_ctx);
    }
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return {};
    }
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindFftSolutions(const ConvolutionContext& ctx,
                                        const AnyInvokeParams& invoke_ctx) const
{

    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_FFT{}))
        return {};

    try
    {
        return FindAllFFTSolutions(ctx, invoke_ctx);
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

    const char* const arch = miopen::GetStringEnv(MIOPEN_DEVICE_ARCH{});
    if(arch != nullptr && strlen(arch) > 0)
    {
        return;
    }
    miopen::solver::ConvSolution selected{miopenStatusUnknownError};
    float best = std::numeric_limits<float>::max();
    Invoker best_invoker;

    for(const auto& sol : solutions)
    {
        if(sol.workspace_sz > 0)
        {
            if(invoke_ctx.workSpace == nullptr)
            {
                MIOPEN_LOG_I("Warning: skipping solver <" << sol.solver_id
                                                          << "> due to no workspace provided ("
                                                          << sol.workspace_sz << " required)");
                continue;
            }
            if(invoke_ctx.workSpaceSize < sol.workspace_sz)
            {
                MIOPEN_LOG_I("Warning: skipping solver <"
                             << sol.solver_id << "> due to insufficient workspace ("
                             << invoke_ctx.workSpaceSize << " < " << sol.workspace_sz << ")");
                continue;
            }
        }

        if(!sol.invoker_factory)
            MIOPEN_THROW("Invoker is not provided by solver " + sol.solver_id);

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        try
        {
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
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_E(ex.what());
        }
    }

    if(selected.Succeeded())
    {
        handle.RegisterInvoker(best_invoker, network_config, selected.solver_id, algorithm_name);
        MIOPEN_LOG_I("Selected: " << selected << ": " << best
                                  << ", workspace_sz = " << selected.workspace_sz);
        record.SetValues(algorithm_name,
                         FindDbData{selected.solver_id,
                                    best,
                                    selected.workspace_sz,
                                    FindDbKCacheKey::MakeUnused(algorithm_name)});
    }
}

static inline void AppendPointersToElements(const std::vector<miopen::solver::ConvSolution>& from,
                                            std::vector<const miopen::solver::ConvSolution*>& to)
{
    std::transform(from.begin(),
                   from.end(),
                   std::back_inserter(to),
                   [](const miopen::solver::ConvSolution& s) { return &s; });
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
                            DbRecord& record,
                            ConvolutionContext& ctx, // non-const only for use_winograd_only hack.
                            bool use_winograd_only)
{
    AutoEnableProfiling enableProfiling{handle};
    ValidateGroupCount(xDesc, wDesc, conv);

    const auto network_config = ctx.BuildConfKey();
    const auto invoke_ctx     = conv::DataInvokeParams{InvokeType::Evaluate,
                                                   {xDesc, x, wDesc, w, yDesc, y},
                                                   workSpace,
                                                   workSpaceSize,
                                                   conv.attribute.gfx90aFp16alt.GetFwd()};

    // Find solutions
    const auto winograd = !use_winograd_only ? conv.FindWinogradSolutions(ctx, invoke_ctx) : [&]() {
        AutoUseFastDynamicSolutions tmp{ctx};
        return conv.FindWinogradSolutions(ctx, invoke_ctx);
    }();
    ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
    bufs.SetFwd(x, w, y);
    const auto gemm = !use_winograd_only ? conv.FindDataGemmSolutions(ctx, invoke_ctx)
                                         : std::vector<miopen::solver::ConvSolution>{};
    const auto direct =
        !use_winograd_only
            ? conv.FindDataDirectSolutions(
                  handle, xDesc, wDesc, yDesc, exhaustiveSearch, true, bufs, invoke_ctx)
            : std::vector<miopen::solver::ConvSolution>{};
    const auto igemm =
        !use_winograd_only
            ? conv.FindDataImplicitGemmSolutions(
                  handle, xDesc, wDesc, yDesc, exhaustiveSearch, true, bufs, invoke_ctx)
            : std::vector<miopen::solver::ConvSolution>{};
    const auto fft = !use_winograd_only ? conv.FindFftSolutions(ctx, invoke_ctx)
                                        : std::vector<miopen::solver::ConvSolution>{};

    // Precompile
    {
        std::vector<const miopen::solver::ConvSolution*> all;
        all.reserve(gemm.size() + winograd.size() + direct.size() + igemm.size() + fft.size());
        AppendPointersToElements(gemm, all);
        AppendPointersToElements(winograd, all);
        AppendPointersToElements(direct, all);
        AppendPointersToElements(igemm, all);
        AppendPointersToElements(fft, all);
        PrecompileSolutions(handle, all);
    }

    // Evaluate Invokers
    EvaluateInvokers(handle,
                     gemm,
                     AlgorithmName{"miopenConvolutionFwdAlgoGEMM"},
                     network_config,
                     invoke_ctx,
                     record);
    EvaluateInvokers(handle,
                     winograd,
                     AlgorithmName{"miopenConvolutionFwdAlgoWinograd"},
                     network_config,
                     invoke_ctx,
                     record);
    EvaluateInvokers(handle,
                     direct,
                     AlgorithmName{"miopenConvolutionFwdAlgoDirect"},
                     network_config,
                     invoke_ctx,
                     record);
    EvaluateInvokers(handle,
                     igemm,
                     AlgorithmName{"miopenConvolutionFwdAlgoImplicitGEMM"},
                     network_config,
                     invoke_ctx,
                     record);
    EvaluateInvokers(handle,
                     fft,
                     AlgorithmName{"miopenConvolutionFwdAlgoFFT"},
                     network_config,
                     invoke_ctx,
                     record);
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
    auto ctx = ConvolutionContext{problem};
    ctx.SetStream(&handle);

    std::vector<PerfField> perf_db;

    bool use_immediate_solution = false;
    miopenConvSolution_t sol;
    if(findMode.IsFast(ctx) || findMode.IsHybrid(ctx))
    {
        size_t count;
        bool fallback;
        GetForwardSolutions(handle, wDesc, xDesc, yDesc, 1, &count, &sol, &fallback);
        use_immediate_solution = (count > 0) && !(findMode.IsHybrid(ctx) && fallback);
        // In Hybrid Find mode, we use Normal Find instead of Immediate fallback kernels.
    }

    if(use_immediate_solution)
    {
        CompileForwardSolution(handle, wDesc, xDesc, yDesc, sol.solution_id);
        /// It is possible to measure actual execution time and return it to the caller.
        /// \todo Consider if we need (and want to spend time) for this.
        const auto id = solver::Id(sol.solution_id);
        perf_db.push_back(
            {id.GetAlgo(conv::Direction::Forward), id.ToString(), sol.time, sol.workspace_size});
    }
    else
    {
        ctx.DetectRocm();
        ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
        bufs.SetFwd(x, w, y);
        ctx.SetBufs(bufs);
        ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
        perf_db = UserFindDbRecord::TryLoad(handle, problem, [&](DbRecord& record) {
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
                            record,
                            ctx,
                            IsWinograd3x3SupportedAndFast(ctx));
        });
    }

    if(IsEnabled(MIOPEN_DEBUG_COMPILE_ONLY{}))
        MIOPEN_THROW(
            miopenStatusGpuOperationsSkipped,
            "MIOPEN_DEBUG_COMPILE_ONLY is enabled, escaping forward convolution. Search skipped.");

    if(perf_db.empty())
        MIOPEN_THROW("Forward Convolution cannot be executed due to incorrect params");

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
                                         << ", " << perf_db[0].time);
}

void ValidateConvTensors(const ConvTensors& tensors)
{
    const auto invalid_buffers =
        tensors.x == nullptr || tensors.w == nullptr || tensors.y == nullptr;

    const auto tensor_sizes_not_matched = tensors.xDesc.GetSize() != tensors.yDesc.GetSize() ||
                                          tensors.xDesc.GetSize() != tensors.wDesc.GetSize();

    const auto trivial_tensor_types_not_matched =
        tensors.xDesc.GetType() != tensors.yDesc.GetType() &&
        tensors.xDesc.GetType() != miopenInt8 && tensors.xDesc.GetType() != miopenInt8x4;
    const auto int8_in8x4_tensor_not_matched =
        (tensors.xDesc.GetType() == miopenInt8 && tensors.yDesc.GetType() != miopenInt32 &&
         tensors.yDesc.GetType() != miopenFloat) ||
        (tensors.xDesc.GetType() == miopenInt8x4 && tensors.yDesc.GetType() != miopenInt32);

    // if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
    //    MIOPEN_THROW(miopenStatusBadParm);
    //}

    const auto x_tensor_invalid = tensors.xDesc.GetSize() < 3;

    const auto bad_parameters = invalid_buffers || tensor_sizes_not_matched ||
                                trivial_tensor_types_not_matched || int8_in8x4_tensor_not_matched ||
                                x_tensor_invalid;

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

static void ConvForwardCheckNumerics(const Handle& handle,
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

    if(algo != miopenConvolutionFwdAlgoGEMM && xDesc.GetType() == miopenInt8x4)
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
        const auto& invoker       = handle.GetInvoker(network_config, {}, algorithm_name);

        if(invoker)
        {
            const auto& invoke_ctx = conv::DataInvokeParams{
                tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetFwd()};
            (*invoker)(handle, invoke_ctx);
            return;
        }

        MIOPEN_THROW("No invoker was registered for convolution forward. Was find executed?");
    });
}
static std::size_t GetSolutionCount(Handle& handle, const ProblemDescription& problem)
{
    const FindDbRecord fdb_record{handle, problem};
    if(fdb_record.empty())
        return 0;
    return std::distance(fdb_record.begin(), fdb_record.end());
}

static const char immFallbackFailed[] =
    "Requested convolution is not supported or Immediate mode Fallback unsuccessful.";

std::size_t ConvolutionDescriptor::GetSolutionCountFallback(Handle& handle,
                                                            const ProblemDescription& problem) const
{
    size_t n                    = 0;
    const auto maxSolutionCount = solver::GetSolversByPrimitive(solver::Primitive::Convolution)
                                      .size(); // Simple and guarantees to provide enough space.
    GetSolutionsFallback(handle, problem, maxSolutionCount, &n, nullptr);
    if(n > 0)
        return n;
    MIOPEN_LOG_I(immFallbackFailed);
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
    MIOPEN_THROW(miopenStatusNotImplemented, immFallbackFailed);
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
    return GetSolutionCountFallback(handle, problem);
}

static inline bool IsAlgorithmDisabled(const miopenConvAlgorithm_t algo)
{
    switch(algo)
    { // clang-format off
    case miopenConvolutionAlgoGEMM:
        return !MIOPEN_USE_GEMM || miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{});
    case miopenConvolutionAlgoDirect:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{});
    case miopenConvolutionAlgoFFT:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_FFT{});
    case miopenConvolutionAlgoWinograd:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{});
    case miopenConvolutionAlgoImplicitGEMM:
        return miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{});
    default: // Disable future algos by default to enforce explicit handling:
        return true;
    } // clang-format on
}

// Helper class used for emplace and sort.
struct SolutionSortWrapper : miopenConvSolution_t
{
    SolutionSortWrapper(const float& t,
                        const size_t& ws,
                        const uint64_t& id,
                        const miopenConvAlgorithm_t& algo)
        : miopenConvSolution_t{t, ws, id, algo}
    {
    }
    bool operator<(const SolutionSortWrapper& other) const
    {
        // Negative values are very coarse estimations.
        // The more modulus, the "worse" (slower) is solution.
        if(time < 0 && other.time < 0)
            return !(time < other.time);
        // Positive values are always "better" than negative (coarse) estimations.
        if(time > 0 && other.time < 0)
            return true;
        if(time < 0 && other.time > 0)
            return false;
        // Both values are positive. The less is the better.
        return (time < other.time);
    }
};

void ConvolutionDescriptor::GetSolutionsFallback(Handle& handle,
                                                 const ProblemDescription& problem,
                                                 const size_t maxSolutionCount,
                                                 size_t* const solutionCount,
                                                 miopenConvSolution_t* const solutions) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMMED_FALLBACK{}))
    {
        MIOPEN_LOG_I("Disabled via environment");
        *solutionCount = 0;
        return;
    }

    /// \todo This is terrible. Should do away when we converge to
    /// single conv::ProblemDescription type.
    const auto& inDesc = problem.direction.IsForward() ? problem.conv_problem.GetIn()
                                                       : problem.conv_problem.GetOut();
    const auto& weightsDesc = problem.conv_problem.GetWeights();
    // This check is needed on fallback path only.
    // On regular path (find-db hit) this was checked during Find().
    ValidateGroupCount(inDesc, weightsDesc, *this);

    std::vector<SolutionSortWrapper> interim;
    interim.reserve(maxSolutionCount); // For speed. In most cases we have less entries than asked.

    auto ctx = ConvolutionContext{problem};
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    const auto wti2time = [](const float& wti) {
        assert(wti != 0.0f);
        if(wti <= 0.0f) // Return negative values as is, avoid DIV/0.
            return wti;
        return 10.0f / wti; // Assume WTI == 1.0 (100%) is 10 ms.
    };

    for(const auto& solver_id : solver::GetSolversByPrimitive(solver::Primitive::Convolution))
    {
        // solver_id is always valid here, because taken from registry.
        // Validity check is not required.
        const auto algo = solver_id.GetAlgo();
        if(IsAlgorithmDisabled(algo)) // Algos can be disabled globally.
            continue;
        const auto& s = solver_id.GetSolver();
        if(s.IsEmpty())
            continue;
        if(!s.IsDynamic()) // Let's allow non-dynamic later, if necessary.
            continue;
        if(!s.IsApplicable(ctx))
            continue;

        const auto wti = s.GetWti(ctx);
        MIOPEN_LOG_I2(solver_id.ToString() << " Estimated WTI = " << wti);
        if(wti < 0.0f) // Skip unknown WTIs.
            continue;

        interim.emplace_back(wti2time(wti), s.GetWorkspaceSize(ctx), solver_id.Value(), algo);
    }

    MIOPEN_LOG_I2("maxSolutionCount = " << maxSolutionCount << ", available = " << interim.size());
    for(const auto& s : interim)
        MIOPEN_LOG_I2("id: " << s.solution_id << " algo: " << s.algorithm << ", time: " << s.time
                             << " ms, ws: " << s.workspace_size
                             << ", name: " << miopen::solver::Id(s.solution_id).ToString());
    // Dual purpose variable:
    // * Used as index for writing into output array (solutions).
    // * Counts the number of entries written, yielding value for solutionsCount.
    auto i = std::size_t{0};
    std::sort(begin(interim), end(interim));
    for(const auto& entry : interim)
    {
        if(i >= maxSolutionCount)
            break;
        if(solutions != nullptr)
            solutions[i] = entry;
        ++i;
    }
    *solutionCount = i;
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

    std::vector<SolutionSortWrapper> interim;
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

        if(solver_id.GetSolver().IsApplicable(ctx))
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

/// \todo Extend miopenConvSolution_t with an attribute indicating
/// how the solution was obtained (benchmarked on the current system,
/// taken from the System find-db, heuristically estimated, produced by
/// MLP classifier...) and then remove the fallbackPathTaken out param.
void ConvolutionDescriptor::GetForwardSolutions(Handle& handle,
                                                const TensorDescriptor& wDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& yDesc,
                                                const size_t maxSolutionCount,
                                                size_t* const solutionCount,
                                                miopenConvSolution_t* const solutions,
                                                bool* const fallbackPathTaken) const
{
    MIOPEN_LOG_I("");
    if(solutionCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutionCount cannot be nullptr");
    if(solutions == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "solutions cannot be nullptr");

    auto problem = ConvolutionContext{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
    problem.SetStream(&handle);

    GetSolutions(
        handle, problem, maxSolutionCount, solutionCount, solutions, StringToConvolutionFwdAlgo);

    if(fallbackPathTaken != nullptr)
        *fallbackPathTaken = (*solutionCount == 0);
    if(*solutionCount == 0)
        GetSolutionsFallback(handle, problem, maxSolutionCount, solutionCount, solutions);
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
    auto sol = solver_id.GetSolver();
    if(!sol.MayNeedWorkspace())
        return 0;
    auto ctx = ConvolutionContext{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    if(sol.IsApplicable(ctx))
        return sol.GetWorkspaceSize(ctx);
    MIOPEN_THROW(miopenStatusBadParm,
                 "The supplied solution id: " + solver_id.ToString() +
                     " is not applicable to the current problem");
}

// Todo: remove when all immediate mode calls will support invokers
static std::vector<KernelInvoke> CompileSolver(const Handle& handle,
                                               ConvolutionContext& ctx,
                                               solver::Id solver_id,
                                               const FindDbKCacheKey& key)
{
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto solver   = solver_id.GetSolver();
    auto db             = GetDb(ctx);
    const auto solution = solver.FindSolution(ctx, db, {}); // auto tune is not expected here

    std::vector<KernelInvoke> kernels;
    AddKernels(handle, key.algorithm_name, key.network_config, solution, &kernels);
    return kernels;
}

static Invoker PrepareInvoker(Handle& handle,
                              ConvolutionContext& ctx,
                              const NetworkConfig& config,
                              solver::Id solver_id,
                              conv::Direction dir)
{
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto solver = solver_id.GetSolver();
    auto db           = GetDb(ctx);
    auto solution     = solver.FindSolution(ctx, db, {}); // auto tune is not expected here
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);

    handle.RegisterInvoker(
        invoker, config, solver_id.ToString(), AlgorithmName(solver_id.GetAlgo(dir)));
    return invoker; // NOLINT (performance-no-automatic-move)
}

Invoker LoadOrPrepareInvoker(Handle& handle,
                             ConvolutionContext& ctx,
                             solver::Id solver_id,
                             conv::Direction dir)
{
    const auto config = ctx.BuildConfKey();
    auto invoker      = handle.GetInvoker(config, solver_id);
    if(invoker)
        return *invoker;
    return PrepareInvoker(handle, ctx, config, solver_id, dir);
}

static bool CheckInvokerSupport(const solver::Id solver_id, conv::Direction dir)
{
    const auto& algo = solver_id.GetAlgo(dir);
    return CheckInvokerSupport(algo);
}

static void CompileSolution(Handle& handle,
                            const solver::Id solver_id,
                            ConvolutionContext& ctx,
                            conv::Direction dir)
{
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm, "solver_id = " + solver_id.ToString());

    if(CheckInvokerSupport(solver_id, dir))
    {
        LoadOrPrepareInvoker(handle, ctx, solver_id, dir);
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

    CompileSolution(handle, solver_id, ctx, conv::Direction::Forward);
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

        if(!CheckInvokerSupport(solver_id, conv::Direction::Forward))
        {
            const auto algo_name = solver_id.GetAlgo(conv::Direction::Forward);
            MIOPEN_THROW("Conv forward algorithm " + algo_name + " must implement invokers.");
        }

        const auto invoker = LoadOrPrepareInvoker(handle, ctx, solver_id, conv::Direction::Forward);
        const auto invoke_ctx = conv::DataInvokeParams{
            tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetFwd()};
        invoker(handle, invoke_ctx);
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

    *returnedAlgoCount = 0;

    AutoEnableProfiling enableProfiling{handle};
    ValidateGroupCount(dxDesc, wDesc, *this);

    const ProblemDescription problem(dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData);
    std::vector<PerfField> perf_db;

    bool use_immediate_solution = false;
    miopenConvSolution_t imm_sol;
    auto ctx = ConvolutionContext{problem};
    if(findMode.IsFast(ctx) || findMode.IsHybrid(ctx))
    {
        size_t count;
        bool fallback;
        GetBackwardSolutions(handle, dyDesc, wDesc, dxDesc, 1, &count, &imm_sol, &fallback);
        use_immediate_solution = (count > 0) && !(findMode.IsHybrid(ctx) && fallback);
    }

    if(use_immediate_solution)
    {
        CompileBackwardSolution(handle, dyDesc, wDesc, dxDesc, imm_sol.solution_id);
        const auto id = solver::Id(imm_sol.solution_id);
        perf_db.push_back({id.GetAlgo(conv::Direction::BackwardData),
                           id.ToString(),
                           imm_sol.time,
                           imm_sol.workspace_size});
    }
    else
    {
        const auto use_winograd_only = [&]() {
            ctx.SetStream(&handle);
            ctx.DetectRocm();
            return IsWinograd3x3SupportedAndFast(ctx);
        }();

        perf_db = UserFindDbRecord::TryLoad(handle, problem, [&](DbRecord& record) {
            const auto network_config      = problem.BuildConfKey();
            const auto invoke_ctx          = conv::DataInvokeParams{InvokeType::Evaluate,
                                                           {dyDesc, dy, wDesc, w, dxDesc, dx},
                                                           workSpace,
                                                           workSpaceSize,
                                                           this->attribute.gfx90aFp16alt.GetBwd()};
            ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);

            // Find solutions
            const auto winograd =
                !use_winograd_only ? FindWinogradSolutions(ctx, invoke_ctx) : [&]() {
                    AutoUseFastDynamicSolutions tmp{ctx};
                    return FindWinogradSolutions(ctx, invoke_ctx);
                }();
            ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
            bufs.SetBwd(dx, w, dy);
            const auto gemm = !use_winograd_only ? FindDataGemmSolutions(ctx, invoke_ctx)
                                                 : std::vector<miopen::solver::ConvSolution>{};
            const auto direct =
                !use_winograd_only
                    ? FindDataDirectSolutions(
                          handle, dxDesc, wDesc, dyDesc, exhaustiveSearch, false, bufs, invoke_ctx)
                    : std::vector<miopen::solver::ConvSolution>{};
            const auto igemm =
                !use_winograd_only
                    ? FindDataImplicitGemmSolutions(
                          handle, dxDesc, wDesc, dyDesc, exhaustiveSearch, false, bufs, invoke_ctx)
                    : std::vector<miopen::solver::ConvSolution>{};
            const auto fft = !use_winograd_only ? FindFftSolutions(ctx, invoke_ctx)
                                                : std::vector<miopen::solver::ConvSolution>{};

            // Precompile
            {
                std::vector<const miopen::solver::ConvSolution*> all;
                all.reserve(gemm.size() + winograd.size() + direct.size() + igemm.size() +
                            fft.size());
                AppendPointersToElements(gemm, all);
                AppendPointersToElements(winograd, all);
                AppendPointersToElements(direct, all);
                AppendPointersToElements(igemm, all);
                AppendPointersToElements(fft, all);
                PrecompileSolutions(handle, all);
            }

            // Evaluate Invokers
            EvaluateInvokers(handle,
                             gemm,
                             AlgorithmName{"miopenConvolutionBwdDataAlgoGEMM"},
                             network_config,
                             invoke_ctx,
                             record);
            EvaluateInvokers(handle,
                             winograd,
                             AlgorithmName{"miopenConvolutionBwdDataAlgoWinograd"},
                             network_config,
                             invoke_ctx,
                             record);
            EvaluateInvokers(handle,
                             direct,
                             AlgorithmName{"miopenConvolutionBwdDataAlgoDirect"},
                             network_config,
                             invoke_ctx,
                             record);
            EvaluateInvokers(handle,
                             igemm,
                             AlgorithmName{"miopenConvolutionBwdDataAlgoImplicitGEMM"},
                             network_config,
                             invoke_ctx,
                             record);
            EvaluateInvokers(handle,
                             fft,
                             AlgorithmName{"miopenConvolutionBwdDataAlgoFFT"},
                             network_config,
                             invoke_ctx,
                             record);
        });
    }

    if(IsEnabled(MIOPEN_DEBUG_COMPILE_ONLY{}))
        MIOPEN_THROW(
            miopenStatusGpuOperationsSkipped,
            "MIOPEN_DEBUG_COMPILE_ONLY is enabled, escaping bwd convolution. Search skipped.");

    if(perf_db.empty())
        MIOPEN_THROW(miopenStatusUnknownError,
                     "Backward Data Convolution cannot be executed due to incorrect params");

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
                                          << ", " << perf_db[0].time);
}
static void ConvBwdCheckNumerics(const Handle& handle,
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
        const auto& invoker       = handle.GetInvoker(network_config, {}, algorithm_name);

        if(!invoker)
            MIOPEN_THROW("No invoker was registered for convolution backward. Was find executed?");

        const auto& invoke_ctx = conv::DataInvokeParams{
            tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetBwd()};
        (*invoker)(handle, invoke_ctx);
    });
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
    return GetSolutionCountFallback(handle, problem);
}

void ConvolutionDescriptor::GetBackwardSolutions(Handle& handle,
                                                 const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& wDesc,
                                                 const TensorDescriptor& dxDesc,
                                                 size_t maxSolutionCount,
                                                 size_t* solutionCount,
                                                 miopenConvSolution_t* solutions,
                                                 bool* const fallbackPathTaken) const
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

    if(fallbackPathTaken != nullptr)
        *fallbackPathTaken = (*solutionCount == 0);
    if(*solutionCount == 0)
        GetSolutionsFallback(handle, problem, maxSolutionCount, solutionCount, solutions);
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

    CompileSolution(handle, solver_id, ctx, conv::Direction::BackwardData);
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

    auto sol = solver_id.GetSolver();
    if(!sol.MayNeedWorkspace())
        return 0;
    auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    if(sol.IsApplicable(ctx))
        return sol.GetWorkspaceSize(ctx);
    else
        MIOPEN_THROW(miopenStatusBadParm,
                     "The supplied solution id: " + solver_id.ToString() +
                         " is not applicable to the current problem");
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

    static const float beta = 0.0f;
    ConvBwdCheckNumerics(handle, tensors, &beta, [&]() {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        ValidateGroupCount(dxDesc, wDesc, *this);

        auto ctx = ConvolutionContext{dxDesc, wDesc, dyDesc, *this, conv::Direction::BackwardData};
        ctx.SetStream(&handle);

        if(!CheckInvokerSupport(solver_id, conv::Direction::BackwardData))
        {
            const auto algo_name = solver_id.GetAlgo(conv::Direction::BackwardData);
            MIOPEN_THROW("Conv backward algorithm " + algo_name + " must implement invokers.");
        }

        const auto invoker =
            LoadOrPrepareInvoker(handle, ctx, solver_id, conv::Direction::BackwardData);
        const auto invoke_ctx = conv::DataInvokeParams{
            tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetBwd()};
        invoker(handle, invoke_ctx);
    });
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
    auto ctx = ConvolutionContext{problem};

    std::vector<PerfField> perf_db;
    bool use_immediate_solution = false;
    miopenConvSolution_t imm_sol;
    if(findMode.IsFast(ctx) || findMode.IsHybrid(ctx))
    {
        size_t count;
        bool fallback;
        GetWrwSolutions(handle, dyDesc, xDesc, dwDesc, 1, &count, &imm_sol, &fallback);
        use_immediate_solution = (count > 0) && !(findMode.IsHybrid(ctx) && fallback);
    }

    if(use_immediate_solution)
    {
        CompileWrwSolution(handle, dyDesc, xDesc, dwDesc, imm_sol.solution_id);
        const auto id = solver::Id(imm_sol.solution_id);
        perf_db.push_back({id.GetAlgo(conv::Direction::BackwardWeights),
                           id.ToString(),
                           imm_sol.time,
                           imm_sol.workspace_size});
    }
    else
    {
        perf_db = UserFindDbRecord::TryLoad(handle, problem, [&](DbRecord& record) {
            ConvolutionUserBuffers bufs(workSpace, workSpaceSize);
            bufs.SetWrW(x, dw, dy);
            ctx.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
            ctx.do_search                  = exhaustiveSearch;
            ctx.SetStream(&handle);
            ctx.SetBufs(bufs);
            ctx.SetupFloats();
            ctx.DetectRocm();
            const auto network_config = ctx.BuildConfKey();
            const auto invoke_ctx     = conv::WrWInvokeParams{InvokeType::Evaluate,
                                                          {dyDesc, dy, xDesc, x, dwDesc, dw},
                                                          workSpace,
                                                          workSpaceSize,
                                                          this->attribute.gfx90aFp16alt.GetWrW()};

            // Find solutions
            const auto gemm = !miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{})
                                  ? FindAllGemmSolutions(ctx, invoke_ctx)
                                  : std::vector<miopen::solver::ConvSolution>{};
            const auto direct = !miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{})
                                    ? FindAllBwdWrW2DSolutions(ctx, invoke_ctx)
                                    : std::vector<miopen::solver::ConvSolution>{};
            const auto winograd = !miopen::IsDisabled(MIOPEN_DEBUG_CONV_WINOGRAD{})
                                      ? FindWinogradWrWAllSolutions(ctx, invoke_ctx)
                                      : std::vector<miopen::solver::ConvSolution>{};
            const auto implictgemm = !miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM{})
                                         ? FindImplicitGemmWrWAllSolutions(ctx, invoke_ctx)
                                         : std::vector<miopen::solver::ConvSolution>{};

            // Precompile Solutions
            {
                std::vector<const miopen::solver::ConvSolution*> all;
                all.reserve(gemm.size() + direct.size() + winograd.size() + implictgemm.size());
                AppendPointersToElements(gemm, all);
                AppendPointersToElements(direct, all);
                AppendPointersToElements(winograd, all);
                AppendPointersToElements(implictgemm, all);
                PrecompileSolutions(handle, all);
            }

            // Evaluate Invokers
            EvaluateInvokers(handle,
                             gemm,
                             AlgorithmName{"miopenConvolutionBwdWeightsAlgoGEMM"},
                             network_config,
                             invoke_ctx,
                             record);
            EvaluateInvokers(handle,
                             direct,
                             AlgorithmName{"miopenConvolutionBwdWeightsAlgoDirect"},
                             network_config,
                             invoke_ctx,
                             record);
            EvaluateInvokers(handle,
                             winograd,
                             AlgorithmName{"miopenConvolutionBwdWeightsAlgoWinograd"},
                             network_config,
                             invoke_ctx,
                             record);
            EvaluateInvokers(handle,
                             implictgemm,
                             AlgorithmName{"miopenConvolutionBwdWeightsAlgoImplicitGEMM"},
                             network_config,
                             invoke_ctx,
                             record);
        });
    }

    if(IsEnabled(MIOPEN_DEBUG_COMPILE_ONLY{}))
        MIOPEN_THROW(miopenStatusGpuOperationsSkipped,
                     "MIOPEN_DEBUG_COMPILE_ONLY is enabled, "
                     "escaping backwards convolution. Search "
                     "skipped.");

    if(perf_db.empty())
        MIOPEN_THROW("Backward Weights Convolution cannot be executed due to incorrect params");

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
                                           << ", " << perf_db[0].time);
}

static void ConvWrwCheckNumerics(const Handle& handle,
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
void ConvolutionDescriptor::ConvolutionBackwardWeights(const Handle& handle,
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
    decltype(auto) tensors = ConvWrwTensors{dyDesc, dy, xDesc, x, dwDesc, dw};
    ValidateConvTensors(tensors);
    ValidateAlphaBeta(alpha, beta);

    if(xDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    ConvWrwCheckNumerics(handle, tensors, beta, [&]() {
        ValidateGroupCount(xDesc, dwDesc, *this);

        decltype(auto) direction      = conv::Direction::BackwardWeights;
        decltype(auto) algorithm_name = AlgorithmName{ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(algo), direction)};
        decltype(auto) ctx = conv::ProblemDescription{dyDesc, dwDesc, xDesc, *this, direction};
        decltype(auto) network_config = ctx.BuildConfKey();
        decltype(auto) invoker = handle.GetInvoker(network_config, boost::none, algorithm_name);

        if(!invoker)
            MIOPEN_THROW("No invoker was registered for convolution weights. Was find executed?");

        const auto invoke_ctx = conv::WrWInvokeParams{
            tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetWrW()};
        (*invoker)(handle, invoke_ctx);
    });
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
    return GetSolutionCountFallback(handle, problem);
}

void ConvolutionDescriptor::GetWrwSolutions(Handle& handle,
                                            const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& xDesc,
                                            const TensorDescriptor& dwDesc,
                                            size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions,
                                            bool* const fallbackPathTaken) const
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

    if(fallbackPathTaken != nullptr)
        *fallbackPathTaken = (*solutionCount == 0);
    if(*solutionCount == 0)
        GetSolutionsFallback(handle, problem, maxSolutionCount, solutionCount, solutions);
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

    CompileSolution(handle, solver_id, ctx, conv::Direction::BackwardWeights);
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

    auto sol = solver_id.GetSolver();
    if(!sol.MayNeedWorkspace())
        return 0;
    auto problem =
        ProblemDescription{xDesc, dwDesc, dyDesc, *this, conv::Direction::BackwardWeights};
    auto ctx = ConvolutionContext{problem};
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    if(sol.IsApplicable(ctx))
        return sol.GetWorkspaceSize(ctx);
    else
        MIOPEN_THROW(miopenStatusBadParm,
                     "The supplied solution id: " + solver_id.ToString() +
                         " is not applicable to the current problem");
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

        if(!CheckInvokerSupport(solver_id, conv::Direction::BackwardWeights))
        {
            MIOPEN_THROW("Solver " + solver_id.ToString() +
                         " requested in immediate WrW, which is not supported.");
        }

        const auto invoker =
            LoadOrPrepareInvoker(handle, ctx, solver_id, conv::Direction::BackwardWeights);
        const auto invoke_ctx = conv::WrWInvokeParams{
            tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetWrW()};
        invoker(handle, invoke_ctx);
    });
}

void ConvolutionBackwardBias(const Handle& handle,
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
    std::string algo_name        = "miopenConvolutionBwdBias";
    std::string program_name     = "MIOpenConvBwdBias.cl";
    std::string kernel_name      = "MIOpenConvBwdB";
    std::string network_config =
        "convbwdbias-" +
        std::string(dyDesc.GetType() == miopenFloat
                        ? "fp32"
                        : (dyDesc.GetType() == miopenHalf
                               ? "fp16"
                               : (dyDesc.GetType() == miopenBFloat16 ? "bfloat16" : "int32")));

    std::string params;
    std::size_t lcl_grp_size0 = 256;
    std::size_t lcl_grp_size1 = 1;
    std::size_t local_mem_sz  = 256;

    std::size_t map_size         = std::accumulate(dyDesc.GetLengths().begin() + 2,
                                           dyDesc.GetLengths().end(),
                                           std::size_t(1),
                                           std::multiplies<std::size_t>());
    std::size_t read_unit        = 4;
    std::size_t map_size_aligned = (map_size + (read_unit - 1)) / read_unit;
    std::size_t off_pix          = map_size - (map_size / read_unit) * read_unit;
    std::size_t total_work       = map_size_aligned * out_n;

    params = " -DMLO_CONVBWD_GROUP_SZ0=" + std::to_string(lcl_grp_size0);
    params += " -DMLO_CONVBWD_GROUP_SZ1=" + std::to_string(lcl_grp_size1);
    params += " -DMLO_CONVBWDB_LCL_MEMSZ=" + std::to_string(local_mem_sz);
    params += " -DMLO_CONVBWDB_UNITSIZE=" + std::to_string(read_unit);

    params += GetDataTypeKernelParams(dyDesc.GetType());

    const std::vector<size_t> vld = {lcl_grp_size0, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {lcl_grp_size0, size_t{256}, size_t{1}};

    auto&& kernels = handle.GetKernels(algo_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(dy,
                        db,
                        uint(out_k),
                        uint(stride_k),
                        uint(stride_n),
                        uint(map_size_aligned),
                        uint(off_pix),
                        uint(total_work));
    }
    else
    {
        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, params)(
            dy,
            db,
            uint(out_k),
            uint(stride_k),
            uint(stride_n),
            uint(map_size_aligned),
            uint(off_pix),
            uint(total_work));
    }

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dbDesc, db);
    }
}

} // namespace miopen
