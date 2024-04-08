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
#include <miopen/conv/solver_finders.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/env.hpp>
#include <miopen/find_db.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/generic_search_controls.hpp>
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
#include <miopen/conv/heuristics/ai_heuristics.hpp>

#include <cassert>
#include <functional>
#include <type_traits>

#include <boost/range/adaptors.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMMED_FALLBACK)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DUMP_TENSOR_PATH)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK)

namespace miopen {

static inline bool IsValidFilterChannelNumber(const TensorDescriptor& x,
                                              const TensorDescriptor& w,
                                              const miopenTensorLayout_t layout,
                                              const int groups)
{
    if(layout == miopenTensorNCHW      //
       || layout == miopenTensorNCHWc4 //
       || layout == miopenTensorNCHWc8)
    {
        return x.GetLengths()[1] / groups == w.GetLengths()[1];
    }

    if(layout == miopenTensorCHWNc4 //
       || layout == miopenTensorCHWNc8)
    {
        return x.GetLengths()[1] / groups == w.GetLengths()[0];
    }

    return true;
}

static inline bool IsValidGroupCount(const TensorDescriptor& x,
                                     const TensorDescriptor& w,
                                     const miopenTensorLayout_t layout,
                                     const int groups)
{
    if(groups > 1) // Optimize for speed
    {
        if(x.GetLengths()[1] % groups != 0)
            return false;

        if(layout == miopenTensorNCHW      //
           || layout == miopenTensorNCHWc4 //
           || layout == miopenTensorNCHWc8)
            return w.GetLengths()[0] % groups == 0;

        if(layout == miopenTensorCHWNc4 //
           || layout == miopenTensorCHWNc8)
            return w.GetLengths()[3] % groups == 0;
    }
    return true;
}

static inline void ValidateGroupCount(const TensorDescriptor& x,
                                      const TensorDescriptor& w,
                                      const ConvolutionDescriptor& conv)
{
    const auto layout = w.GetLayout_t();
    const auto groups = conv.group_count;
    assert(groups > 0);

    const auto ok_c = IsValidFilterChannelNumber(x, w, layout, groups);
    const auto ok_g = IsValidGroupCount(x, w, layout, groups);

    if(ok_c && ok_g)
        return;

    MIOPEN_LOG_W(w.GetLayout_str() << "w {" << w.ToString() << "}, " //
                                   << "x {" << x.ToString() << "}, " //
                                   << "groups = " << conv.group_count);
    if(!ok_c)
        MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    if(!ok_g)
        MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
}

static inline void ValidateWorkspace(Data_t workSpace, const size_t workSpaceSize)
{

    [[maybe_unused]] bool x = (workSpace != nullptr);
    [[maybe_unused]] bool y = (workSpaceSize != 0);

    assert(((x && y) || (!x && !y)) && "workspace pointer and size don't match. Either both should "
                                       "be zero or both should be non-zero");

    /// \todo could add a check here that workSpace points to GPU memory
}

static Invoker PrepareInvoker(ExecutionContext ctx,
                              const conv::ProblemDescription& problem,
                              const NetworkConfig& config,
                              solver::Id solver_id)
{
    problem.SetupFloats(ctx);
    ctx.do_search              = false;
    ctx.disable_search_enforce = true;

    const auto solver = solver_id.GetSolver();
    auto db           = GetDb(ctx);
    auto solution     = solver.FindSolution(ctx, problem, db, {}); // auto tune is not expected here
    auto& handle      = ctx.GetStream();
    auto invoker = handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    const auto algo = AlgorithmName{solver_id.GetAlgo(problem.GetDirection())};

    handle.RegisterInvoker(invoker, config, solver_id.ToString(), algo);
    return invoker;
}

Invoker LoadOrPrepareInvoker(const ExecutionContext& ctx,
                             const conv::ProblemDescription& problem,
                             solver::Id solver_id)
{
    const auto& handle = ctx.GetStream();
    const auto config  = problem.MakeNetworkConfig();
    auto invoker       = handle.GetInvoker(config, solver_id);
    if(invoker)
        return *invoker;
    return PrepareInvoker(ctx, problem, config, solver_id);
}

static void
CompileSolution(solver::Id solver_id, ExecutionContext ctx, const conv::ProblemDescription& problem)
{
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm, "solver_id = " + solver_id.ToString());

    ctx.disable_search_enforce = true;
    LoadOrPrepareInvoker(ctx, problem, solver_id);
}

/// Keep only the best within algorithm, remove all others.
static void ShrinkToFind10Results(std::vector<PerfField>& found)
{
    std::vector<PerfField> out;
    std::sort(begin(found), end(found));
    for(const auto& f : found)
    {
        // If an algo already resides in out, then skip solver.
        if(std::find_if(out.begin(), out.end(), [&](const auto& o) {
               return o.algorithm == f.algorithm;
           }) != out.end())
            continue;
        out.emplace_back(f);
    }
    found = out;
}

static inline std::vector<PerfField> FindConvolution(const ExecutionContext& ctx,
                                                     const conv::ProblemDescription& problem,
                                                     const AnyInvokeParams& invoke_ctx)
{
    auto results         = std::vector<PerfField>{};
    auto sol             = boost::optional<miopenConvSolution_t>{};
    const auto& conv     = problem.GetConv();
    const auto& findMode = conv.findMode;

    if(findMode.IsFast(ctx) || findMode.IsHybrid(ctx))
    {
        auto fallback = bool{};
        auto sols     = conv.GetSolutions(ctx, problem, 1, &fallback);
        // override the normal find with immed mode with env var
        if(!sols.empty() && (!(findMode.IsHybrid(ctx) && fallback) ||
                             miopen::IsEnabled(ENV(MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK))))
            sol = sols.front();
        // In Hybrid Find mode, we use Normal Find instead of Immediate fallback kernels.
    }

    if(sol.has_value())
    {
        /// It is possible to measure actual execution time and return it to the caller.
        /// \todo Consider if we need (and want to spend time) for this.
        const auto id = solver::Id{sol->solution_id};
        const auto& s = id.GetSolver();
        CompileSolution(id, ctx, problem);
        results.push_back({id.GetAlgo(problem.GetDirection()),
                           id.ToString(),
                           sol->time,
                           s.GetWorkspaceSize(ctx, problem)});
    }
    else
    {
        results = UserFindDbRecord::TryLoad(ctx.GetStream(), problem, [&](DbRecord& record) {
            auto ctx_copy                       = ctx;
            ctx_copy.use_dynamic_solutions_only = findMode.IsDynamicHybrid(ctx);
            const auto params =
                conv::ConvFindParameters{conv.IsWinograd3x3SupportedAndFast(ctx_copy, problem)};

            FindCore(invoke_ctx, record, ctx_copy, problem, params, conv::GetConvSolverFinders());
        });
    }

    if(IsEnabled(ENV(MIOPEN_DEBUG_COMPILE_ONLY)))
    {
        MIOPEN_THROW(
            miopenStatusGpuOperationsSkipped,
            "MIOPEN_DEBUG_COMPILE_ONLY is enabled, escaping forward convolution. Search skipped.");
    }

    ShrinkToFind10Results(results);

    for(const auto& entry : results)
        MIOPEN_LOG_I(entry.algorithm << "\t" << entry.time << "\t" << entry.workspace);

    return results;
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
    ValidateWorkspace(workSpace, workSpaceSize);
    if(x == nullptr || w == nullptr || y == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");

    *returnedAlgoCount = 0;

    const auto problem =
        conv::ProblemDescription(xDesc, wDesc, yDesc, *this, conv::Direction::Forward);
    const auto ctx = [&] {
        auto tmp = ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        tmp.do_search = exhaustiveSearch;
        return tmp;
    }();

    const auto invoke_ctx = conv::DataInvokeParams{InvokeType::Evaluate,
                                                   {xDesc, x, wDesc, w, yDesc, y},
                                                   workSpace,
                                                   workSpaceSize,
                                                   attribute.gfx90aFp16alt.GetFwd()};

    const auto results = FindConvolution(ctx, problem, invoke_ctx);

    if(results.empty())
    {
        // Changes to this message lead to failures in test_conv_for_implicit_gemm
        // To fix them check the test
        // Two similar messages are in other convolution find methods
        MIOPEN_THROW("No suitable algorithm was found to execute the required convolution");
    }

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(results.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].fwd_algo = StringToConvolutionFwdAlgo(results[i].algorithm);
        perfResults[i].time     = results[i].time;
        perfResults[i].memory   = results[i].workspace;
    }

    MIOPEN_LOG_I("FW Chosen Algorithm: " << results[0].solver_id << " , " << results[0].workspace
                                         << ", " << results[0].time);
}

namespace {

void ValidateAlphaBeta(const void* alpha, const void* beta)
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Only alpha=1 and beta=0 is supported");
    }
}

} // namespace

void DumpTensorToFileFromDevice(const miopen::Handle& handle,
                                const miopen::TensorDescriptor& tDesc,
                                ConstData_t dData,
                                const std::string& filename)
{
    if(dData == nullptr)
    {
        MIOPEN_LOG_E("Dereferencing nullptr when trying to dump tensor from gpu");
        return;
    }

    fs::path file_name_with_path(filename);
    fs::path path = file_name_with_path.parent_path();

    // dump to current folder if full path not provided.
    if(path.empty())
    {
        path                = fs::current_path();
        file_name_with_path = path / file_name_with_path; // append paths
    }
    if(!fs::exists(path))
    {
        MIOPEN_LOG_E("Directory does not exists : " << path);
        return;
    }

    std::ofstream file_stream{file_name_with_path};
    if(!file_stream.is_open())
    {
        MIOPEN_LOG_E("Cannot write to file : " << file_name_with_path);
        return;
    }

    // read tensor data from gpu
    size_t num_bytes = tDesc.GetNumBytes();
    MIOPEN_LOG_I2("Start bringing tensor from device to host");
    std::vector<char> hdata(num_bytes);
    handle.ReadTo(hdata.data(), dData, num_bytes);
    MIOPEN_LOG_I2("Done bringing tensor from device to host");
    // write tensor data to file
    const char* pointer = hdata.data();
    file_stream.write(pointer, num_bytes);
    file_stream.close();
    MIOPEN_LOG_I("Dumping tensor to file : " << file_name_with_path);
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

    bool flag = false;

    flag |= miopen::checkNumericsInput(handle, tensors.xDesc, tensors.x);
    flag |= miopen::checkNumericsInput(handle, tensors.wDesc, tensors.w);

    worker();

    flag |= miopen::checkNumericsOutput(handle, tensors.yDesc, tensors.y);

    const auto& file_name = miopen::GetStringEnv(ENV(MIOPEN_DUMP_TENSOR_PATH));
    if(flag && !file_name.empty())
    {
        DumpTensorToFileFromDevice(handle, tensors.xDesc, tensors.x, file_name + "_x.bin");
        DumpTensorToFileFromDevice(handle, tensors.wDesc, tensors.w, file_name + "_w.bin");
        DumpTensorToFileFromDevice(handle, tensors.yDesc, tensors.y, file_name + "_y.bin");
    }
}

void ConvolutionDescriptor::ValidateTensors(const ConvTensors& tensors) const
{

    // Group stride in current TensorDescriptor is implicit. When invoking kernels,
    // we need to add the group dimension G and compute its stride. We want the stride
    // left of C to be a multiple of group count G. e.g. for NCHW, the stride for N
    // should be a multiple of G so that we can compute the strides for NGCHW
    auto bad_group_stride = [this](const TensorDescriptor& td) {
        auto l             = td.GetLayout_t();
        int g_stride_index = -1;
        if(l == miopenTensorNCHW || l == miopenTensorNCDHW)
        {
            g_stride_index = 0; // stride index for N;
        }
        else if(l == miopenTensorNHWC || l == miopenTensorNDHWC)
        {
            // stride index for W. Normally this would be 2nd-last stride but we store
            // strides in NCHW order for some weird reason.
            g_stride_index = td.GetStrides().size() - 1;
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError, "Layout not supported for grouped convolution");
        }

        if(g_stride_index != -1)
        {
            return (td.GetStrides()[g_stride_index] % this->group_count) != 0;
        }

        return false;
    };

    // invalid_buffers
    if(tensors.x == nullptr || tensors.w == nullptr || tensors.y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "One of the convolution tensors is null");
    }

    // x_tensor_invalid =
    if(tensors.xDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm, "input tensor's number of dimensions is wrong");
    }

    // tensor_sizes_not_matched =
    if(tensors.xDesc.GetSize() != tensors.yDesc.GetSize() ||
       tensors.xDesc.GetSize() != tensors.wDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "number of dimensions mismatch between input, output and weights tensors");
    }

    // trivial_tensor_types_not_matched =
    if(tensors.xDesc.GetType() != tensors.yDesc.GetType() && tensors.xDesc.GetType() != miopenInt8)
    {
        MIOPEN_THROW(miopenStatusBadParm, "input/output tensor data types do not match");
    }

    // check for bad_group_stride. This applies for input and output only. There
    // is no check for weight tensor currently.
    // no need to check for group_count == 1

    if((this->group_count > 1) && bad_group_stride(tensors.xDesc))
    {
        MIOPEN_THROW(
            miopenStatusBadParm,
            "Invalid input tensor strides. Channel stride must be a multiple of group count");
    }
    if((this->group_count > 1) && bad_group_stride(tensors.yDesc))
    {
        MIOPEN_THROW(
            miopenStatusBadParm,
            "Invalid output tensor strides. Channel stride must be a multiple of group count");
    }

    // if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
    //    MIOPEN_THROW(miopenStatusBadParm);
    //}
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
    ValidateWorkspace(workSpace, workSpaceSize);

    const auto tensors = ConvFwdTensors{xDesc, x, wDesc, w, yDesc, y};
    ValidateTensors(tensors);
    ValidateAlphaBeta(alpha, beta);

    ConvForwardCheckNumerics(handle, tensors, [&]() {
        ValidateGroupCount(xDesc, wDesc, *this);

        const auto algorithm_name = AlgorithmName{ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(algo), conv::Direction::Forward)};

        const auto problem =
            conv::ProblemDescription{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
        const auto network_config = problem.MakeNetworkConfig();
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

static std::size_t GetSolutionCount(Handle& handle, const conv::ProblemDescription& problem)
{
    const FindDbRecord fdb_record{handle, problem};
    if(fdb_record.empty())
        return 0;
    return std::distance(fdb_record.begin(), fdb_record.end());
}

static const char immFallbackFailed[] =
    "Requested convolution is not supported or Immediate mode Fallback unsuccessful.";

std::size_t
ConvolutionDescriptor::GetSolutionCountFallback(const ExecutionContext& ctx,
                                                const conv::ProblemDescription& problem) const
{
    const auto maxSolutionCount = solver::GetSolversByPrimitive(solver::Primitive::Convolution)
                                      .size(); // Simple and guarantees to provide enough space.
    const auto n = GetSolutionsFallback(ctx, problem, maxSolutionCount).size();
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

std::size_t ConvolutionDescriptor::GetSolutionCount(const ExecutionContext& ctx,
                                                    const conv::ProblemDescription& problem) const
{
    MIOPEN_LOG_I("");
    const auto n = miopen::GetSolutionCount(ctx.GetStream(), problem);
    if(n > 0)
        return n;
    return GetSolutionCountFallback(ctx, problem);
}

struct SolutionTimeComparator
{
    bool operator()(const miopenConvSolution_t& lhs, const miopenConvSolution_t& rhs) const
    {
        // Negative values are very coarse estimations.
        // The more modulus, the "worse" (slower) is solution.
        if(lhs.time < 0 && rhs.time < 0)
            return !(lhs.time < rhs.time);
        // Positive values are always "better" than negative (coarse) estimations.
        if(lhs.time > 0 && rhs.time < 0)
            return true;
        if(lhs.time < 0 && rhs.time > 0)
            return false;
        // Both values are positive. The less is the better.
        return (lhs.time < rhs.time);
    }
};

std::vector<miopenConvSolution_t>
ConvolutionDescriptor::GetSolutionsFallback(const ExecutionContext& ctx,
                                            const conv::ProblemDescription& problem,
                                            const size_t maxSolutionCount) const
{
    if(miopen::IsDisabled(ENV(MIOPEN_DEBUG_CONV_IMMED_FALLBACK)))
    {
        MIOPEN_LOG_I("Disabled via environment");
        return {};
    }

    const auto& xDesc =
        (problem.GetDirection() == conv::Direction::Forward) ? problem.GetIn() : problem.GetOut();
    const auto& weightsDesc = problem.GetWeights();
    // This check is needed on fallback path only.
    // On regular path (find-db hit) this was checked during Find().
    ValidateGroupCount(xDesc, weightsDesc, *this);

    auto interim = std::vector<miopenConvSolution_t>{};
    interim.reserve(maxSolutionCount); // For speed. In most cases we have less entries than asked.

    // TunaNet Fallback
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
    if(!miopen::IsDisabled(ENV(MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK)))
    {
        const static std::string arch = ctx.GetStream().GetDeviceName();
        auto solvers                  = ai::immed_mode::PredictSolver(problem, ctx, arch);
        if(!solvers.empty())
        {
            MIOPEN_LOG_I2("Using TunaNet Fallback");
            const auto ai_time = [](const int& idx) {
                return 10.0f * static_cast<float>(idx); // Assume idx == 1 (best solver) is 10 ms.
            };
            int idx = 1;
            for(const auto kinder : solvers)
            {
                const auto solver_id = solver::Id{kinder};
                const auto sol       = solver_id.GetSolver();
                const auto algo      = solver_id.GetAlgo();
                if(conv::IsAlgorithmDisabled(algo))
                    continue;
                if(!sol.IsDynamic())
                    continue; // branch should never be taken
                if(!sol.IsApplicable(ctx, problem))
                    continue;
                interim.emplace_back(miopenConvSolution_t{
                    ai_time(idx), sol.GetWorkspaceSize(ctx, problem), solver_id.Value(), algo});
                ++idx;
            }
        }
    }
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

    // WTI Fallback
    // if TunaNet is not enabled or produces no applicable solvers then fallback to WTI
    if(interim.empty())
    {
        MIOPEN_LOG_I2("Using WTI Fallback");
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
            if(conv::IsAlgorithmDisabled(algo)) // Algos can be disabled globally.
                continue;
            const auto& s = solver_id.GetSolver();
            // Let's allow non-dynamic later, if necessary.
            if(s.IsEmpty() || !s.IsDynamic() || !s.IsApplicable(ctx, problem))
                continue;

            const auto wti = s.GetWti(ctx, problem);
            MIOPEN_LOG_I2(solver_id.ToString() << " Estimated WTI = " << wti);
            if(wti < 0.0f) // Skip unknown WTIs.
                continue;
            interim.emplace_back(miopenConvSolution_t{
                wti2time(wti), s.GetWorkspaceSize(ctx, problem), solver_id.Value(), algo});
        }
    }
    MIOPEN_LOG_I2("maxSolutionCount = " << maxSolutionCount << ", available = " << interim.size());
    for(const auto& s : interim)
    {
        MIOPEN_LOG_I2("id: " << s.solution_id << " algo: " << s.algorithm << ", time: " << s.time
                             << " ms, ws: " << s.workspace_size
                             << ", name: " << miopen::solver::Id(s.solution_id).ToString());
    }
    std::sort(begin(interim), end(interim), SolutionTimeComparator{});
    interim.resize(std::min(maxSolutionCount, interim.size()));

    return interim;
}

namespace {

std::vector<miopenConvSolution_t> GetSolutions(const ExecutionContext& ctx,
                                               const conv::ProblemDescription& problem,
                                               const size_t maxSolutionCount)
{
    auto algo_resolver = std::function<int(const std::string&)>{};

    switch(problem.GetDirection())
    {
    case conv::Direction::Forward: algo_resolver = &StringToConvolutionFwdAlgo; break;
    case conv::Direction::BackwardData: algo_resolver = &StringToConvolutionBwdDataAlgo; break;
    case conv::Direction::BackwardWeights:
        algo_resolver = &StringToConvolutionBwdWeightsAlgo;
        break;
    }

    const FindDbRecord fdb_record{ctx.GetStream(), problem};

    if(fdb_record.empty())
        return {};

    auto interim = std::vector<miopenConvSolution_t>{};
    interim.reserve(20); // Heuristic for speed.

    for(const auto& pair : fdb_record)
    {
        const auto algo = static_cast<miopenConvAlgorithm_t>(algo_resolver(pair.second.algorithm));
        if(conv::IsAlgorithmDisabled(algo))
            continue;

        const auto solver_id = solver::Id{pair.first};
        // Wrong IDs can't be used to call IsApplicable(), so let's
        // ignore obsolete or invalid IDs read from find-db first.
        if(!solver_id.IsValid())
        {
            // Do not disturb users with warnings unless detailed log is enabled.
            MIOPEN_LOG_I("[Warning] incorrect solver_id: " << pair.first);
            continue;
        }

        interim.emplace_back(
            miopenConvSolution_t{pair.second.time, pair.second.workspace, solver_id.Value(), algo});
    }

    std::sort(begin(interim), end(interim), SolutionTimeComparator{});

    // Let's avoid checks of solvers that reside beyond maxSolutionCount,
    // i.e. those that unnecessary anyway. This optimization is important
    // because applicability check may involve running MIIR compiler
    // (for MLIR solvers), which can be very slow.
    interim.resize(std::min(interim.size(), maxSolutionCount));
    const auto to_erase_from = std::remove_if(interim.begin(), interim.end(), [&](auto&& entry) {
        const auto solver_id = solver::Id{entry.solution_id};
        return !solver_id.GetSolver().IsApplicable(ctx, problem);
    });
    interim.erase(to_erase_from, interim.end());

    return interim;
}

} // namespace

/// \todo Extend miopenConvSolution_t with an attribute indicating
/// how the solution was obtained (benchmarked on the current system,
/// taken from the System find-db, heuristically estimated, produced by
/// MLP classifier...) and then remove the fallbackPathTaken out param.
std::vector<miopenConvSolution_t>
ConvolutionDescriptor::GetSolutions(const ExecutionContext& ctx,
                                    const conv::ProblemDescription& problem,
                                    size_t maxSolutionCount,
                                    bool* fallbackPathTaken) const
{
    MIOPEN_LOG_I("");
    auto solutions = miopen::GetSolutions(ctx, problem, maxSolutionCount);

    if(fallbackPathTaken != nullptr)
        *fallbackPathTaken = solutions.empty();

    if(!solutions.empty())
        return solutions;

    return GetSolutionsFallback(ctx, problem, maxSolutionCount);
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
    const auto problem =
        conv::ProblemDescription{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
    auto ctx = ExecutionContext{};
    ctx.SetStream(&handle);
    if(sol.IsApplicable(ctx, problem))
        return sol.GetWorkspaceSize(ctx, problem);
    MIOPEN_THROW(miopenStatusBadParm,
                 "The supplied solution id: " + solver_id.ToString() +
                     " is not applicable to the current problem");
}

void ConvolutionDescriptor::CompileSolution(const ExecutionContext& ctx,
                                            const conv::ProblemDescription& problem,
                                            solver::Id solver_id) const
{
    MIOPEN_LOG_I("solver_id = " << solver_id.ToString());
    miopen::CompileSolution(solver_id, ctx, problem);
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
    ValidateWorkspace(workSpace, workSpaceSize);
    const auto tensors = ConvFwdTensors{xDesc, x, wDesc, w, yDesc, y};

    ValidateTensors(tensors);
    if(!solver_id.IsValid())
        MIOPEN_THROW(miopenStatusBadParm);

    ConvForwardCheckNumerics(handle, tensors, [&]() {
        const auto problem =
            conv::ProblemDescription{xDesc, wDesc, yDesc, *this, conv::Direction::Forward};
        const auto ctx        = ExecutionContext{&handle};
        const auto invoker    = LoadOrPrepareInvoker(ctx, problem, solver_id);
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
    ValidateWorkspace(workSpace, workSpaceSize);
    if(dx == nullptr || w == nullptr || dy == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");

    *returnedAlgoCount = 0;

    ValidateGroupCount(dxDesc, wDesc, *this);

    const auto problem =
        conv::ProblemDescription{dyDesc, wDesc, dxDesc, *this, conv::Direction::BackwardData};

    const auto ctx = [&] {
        auto tmp = ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        tmp.do_search = exhaustiveSearch;
        return tmp;
    }();

    const auto invoke_ctx = conv::DataInvokeParams{InvokeType::Evaluate,
                                                   {dyDesc, dy, wDesc, w, dxDesc, dx},
                                                   workSpace,
                                                   workSpaceSize,
                                                   this->attribute.gfx90aFp16alt.GetBwd()};

    const auto results = FindConvolution(ctx, problem, invoke_ctx);

    if(results.empty())
    {
        // Changes to this message lead to failures in test_conv_for_implicit_gemm
        // To fix them check the test
        // Two similar messages are in other convolution find methods
        MIOPEN_THROW("No suitable algorithm was found to execute the required convolution");
    }

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(results.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].bwd_data_algo = StringToConvolutionBwdDataAlgo(results[i].algorithm);
        perfResults[i].time          = results[i].time;
        perfResults[i].memory        = results[i].workspace;
    }

    MIOPEN_LOG_I("BWD Chosen Algorithm: " << results[0].solver_id << " , " << results[0].workspace
                                          << ", " << results[0].time);
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

    bool flag = false;

    flag |= miopen::checkNumericsInput(handle, tensors.dyDesc, tensors.dy);
    flag |= miopen::checkNumericsInput(handle, tensors.wDesc, tensors.w);
    if(!float_equal(*(static_cast<const float*>(beta)), 0))
        flag |= miopen::checkNumericsInput(handle, tensors.dxDesc, tensors.dx);

    worker();

    flag |= miopen::checkNumericsOutput(handle, tensors.dxDesc, tensors.dx);

    const auto& file_name = miopen::GetStringEnv(ENV(MIOPEN_DUMP_TENSOR_PATH));
    if(flag && !file_name.empty())
    {
        DumpTensorToFileFromDevice(handle, tensors.dyDesc, tensors.dy, file_name + "_dy.bin");
        DumpTensorToFileFromDevice(handle, tensors.wDesc, tensors.w, file_name + "_w.bin");
        DumpTensorToFileFromDevice(handle, tensors.dxDesc, tensors.dx, file_name + "_dx.bin");
    }
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
    ValidateWorkspace(workSpace, workSpaceSize);

    auto tensors = ConvBwdTensors{dyDesc, dy, wDesc, w, dxDesc, dx};

    ValidateTensors(tensors);
    ValidateAlphaBeta(alpha, beta);

    ConvBwdCheckNumerics(handle, tensors, beta, [&]() {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        ValidateGroupCount(dxDesc, wDesc, *this);

        const auto algorithm_name = AlgorithmName{ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(algo), conv::Direction::BackwardData)};

        const auto problem =
            conv::ProblemDescription{dyDesc, wDesc, dxDesc, *this, conv::Direction::BackwardData};
        const auto network_config = problem.MakeNetworkConfig();
        const auto& invoker       = handle.GetInvoker(network_config, {}, algorithm_name);

        if(!invoker)
            MIOPEN_THROW("No invoker was registered for convolution backward. Was find executed?");

        const auto& invoke_ctx = conv::DataInvokeParams{
            tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetBwd()};
        (*invoker)(handle, invoke_ctx);
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

    auto sol = solver_id.GetSolver();
    if(!sol.MayNeedWorkspace())
        return 0;
    const auto problem =
        conv::ProblemDescription{dyDesc, wDesc, dxDesc, *this, conv::Direction::BackwardData};
    auto ctx = ExecutionContext{};
    ctx.SetStream(&handle);
    if(sol.IsApplicable(ctx, problem))
    {
        return sol.GetWorkspaceSize(ctx, problem);
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The supplied solution id: " + solver_id.ToString() +
                         " is not applicable to the current problem");
    }
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
    ValidateWorkspace(workSpace, workSpaceSize);
    auto tensors = ConvBwdTensors{dyDesc, dy, wDesc, w, dxDesc, dx};

    ValidateTensors(tensors);

    static const float beta = 0.0f;
    ConvBwdCheckNumerics(handle, tensors, &beta, [&]() {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        ValidateGroupCount(dxDesc, wDesc, *this);

        const auto problem =
            conv::ProblemDescription{dyDesc, wDesc, dxDesc, *this, conv::Direction::BackwardData};
        const auto ctx        = ExecutionContext{&handle};
        const auto invoker    = LoadOrPrepareInvoker(ctx, problem, solver_id);
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
    ValidateWorkspace(workSpace, workSpaceSize);
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

    const auto problem =
        conv::ProblemDescription{dyDesc, dwDesc, xDesc, *this, conv::Direction::BackwardWeights};
    const auto ctx = [&] {
        auto tmp = ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        tmp.do_search = exhaustiveSearch;
        return tmp;
    }();

    const auto invoke_ctx = conv::WrWInvokeParams{InvokeType::Evaluate,
                                                  {dyDesc, dy, xDesc, x, dwDesc, dw},
                                                  workSpace,
                                                  workSpaceSize,
                                                  attribute.gfx90aFp16alt.GetWrW()};

    const auto results = FindConvolution(ctx, problem, invoke_ctx);

    if(results.empty())
    {
        // Changes to this message lead to failures in test_conv_for_implicit_gemm
        // To fix them check the test
        // Two similar messages are in other convolution find methods
        MIOPEN_THROW("No suitable algorithm was found to execute the required convolution");
    }

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(results.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].bwd_weights_algo = StringToConvolutionBwdWeightsAlgo(results[i].algorithm);
        perfResults[i].time             = results[i].time;
        perfResults[i].memory           = results[i].workspace;
    }
    MIOPEN_LOG_I("BWrW Chosen Algorithm: " << results[0].solver_id << " , " << results[0].workspace
                                           << ", " << results[0].time);
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

    bool flag = false;

    flag |= miopen::checkNumericsInput(handle, tensors.dyDesc, tensors.dy);
    flag |= miopen::checkNumericsInput(handle, tensors.xDesc, tensors.x);
    if(!float_equal(*(static_cast<const float*>(beta)), 0))
        flag |= miopen::checkNumericsInput(handle, tensors.dwDesc, tensors.dw);

    worker();

    flag |= miopen::checkNumericsOutput(handle, tensors.dwDesc, tensors.dw);

    const auto& file_name = miopen::GetStringEnv(ENV(MIOPEN_DUMP_TENSOR_PATH));
    if(flag && !file_name.empty())
    {
        DumpTensorToFileFromDevice(handle, tensors.dyDesc, tensors.dy, file_name + "_dy.bin");
        DumpTensorToFileFromDevice(handle, tensors.xDesc, tensors.x, file_name + "_x.bin");
        DumpTensorToFileFromDevice(handle, tensors.dwDesc, tensors.dw, file_name + "_dw.bin");
    }
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
    ValidateWorkspace(workSpace, workSpaceSize);
    decltype(auto) tensors = ConvWrwTensors{dyDesc, dy, xDesc, x, dwDesc, dw};
    ValidateTensors(tensors);
    ValidateAlphaBeta(alpha, beta);

    if(xDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    ConvWrwCheckNumerics(handle, tensors, beta, [&]() {
        ValidateGroupCount(xDesc, dwDesc, *this);

        decltype(auto) direction      = conv::Direction::BackwardWeights;
        decltype(auto) algorithm_name = AlgorithmName{ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(algo), direction)};
        decltype(auto) problem = conv::ProblemDescription{dyDesc, dwDesc, xDesc, *this, direction};
        decltype(auto) network_config = problem.MakeNetworkConfig();
        decltype(auto) invoker = handle.GetInvoker(network_config, boost::none, algorithm_name);

        if(!invoker)
            MIOPEN_THROW("No invoker was registered for convolution weights. Was find executed?");

        const auto invoke_ctx = conv::WrWInvokeParams{
            tensors, workSpace, workSpaceSize, this->attribute.gfx90aFp16alt.GetWrW()};
        (*invoker)(handle, invoke_ctx);
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

    auto sol = solver_id.GetSolver();
    if(!sol.MayNeedWorkspace())
        return 0;
    const auto problem =
        conv::ProblemDescription{dyDesc, dwDesc, xDesc, *this, conv::Direction::BackwardWeights};
    auto ctx = ExecutionContext{};
    ctx.SetStream(&handle);
    if(sol.IsApplicable(ctx, problem))
    {
        return sol.GetWorkspaceSize(ctx, problem);
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The supplied solution id: " + solver_id.ToString() +
                         " is not applicable to the current problem");
    }
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
    ValidateWorkspace(workSpace, workSpaceSize);
    auto tensors = ConvWrwTensors{dyDesc, dy, xDesc, x, dwDesc, dw};
    ValidateTensors(tensors);

    if(xDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    float beta = 0;
    ConvWrwCheckNumerics(handle, tensors, &beta, [&]() {
        ValidateGroupCount(xDesc, dwDesc, *this);

        const auto problem = conv::ProblemDescription{
            dyDesc, dwDesc, xDesc, *this, conv::Direction::BackwardWeights};
        const auto ctx        = ExecutionContext{&handle};
        const auto invoker    = LoadOrPrepareInvoker(ctx, problem, solver_id);
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
                        static_cast<unsigned>(out_k),
                        static_cast<unsigned>(stride_k),
                        static_cast<unsigned>(stride_n),
                        static_cast<unsigned>(map_size_aligned),
                        static_cast<unsigned>(off_pix),
                        static_cast<unsigned>(total_work));
    }
    else
    {
        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, params)(
            dy,
            db,
            static_cast<unsigned>(out_k),
            static_cast<unsigned>(stride_k),
            static_cast<unsigned>(stride_n),
            static_cast<unsigned>(map_size_aligned),
            static_cast<unsigned>(off_pix),
            static_cast<unsigned>(total_work));
    }

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dbDesc, db);
    }
}

} // namespace miopen
