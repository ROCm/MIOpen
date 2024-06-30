/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_SOLVER_HPP_
#define GUARD_MIOPEN_SOLVER_HPP_

#include <miopen/config.hpp>

#include <miopen/buffer_info.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/performance_config.hpp>
#include <miopen/type_name.hpp>

#include <boost/any.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <ostream>
#include <algorithm>
#include <initializer_list>

namespace miopen {

namespace debug {

/// If set to true, then always enable ConvDirectNaive* solver, regardless of environment value
/// MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_* that control enable/disable of these solvers.
/// Currently used during driver using naive kernel as gpu reference.
MIOPEN_EXPORT extern bool
    AlwaysEnableConvDirectNaive; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug

struct AnyInvokeParams;

namespace solver {
/// \todo Move wave_size into abstraction wich represent GPU information
const int wave_size = 64;

/// Base class for problem solvers.
///
/// Solvers are to be instantiated as const objects and shall not have any variable
/// internal state. Any non-const state information, if required, to be stored in the
/// solver-specific context objects.
///
/// There could be multiple solvers of the same algorithm for a problem config.
struct SolverBase
{
    virtual ~SolverBase() = default;

    /// This will retrieve the id of the solver to write to the database. By
    /// default it uses the class name. If the class is renamed, this function can
    /// overriden to keep the name to avoid DB corruption.
    virtual const std::string& SolverDbId() const = 0;

    /// In some instances (particularly fusions) the fused solver might like to
    /// fallback to the non-fused variant for performance parameters, this information
    /// is returned via AltSolverDbId
    virtual const std::string& AltSolverDbId() const
    {
        static const std::string null_id = "";
        return null_id;
    }

    /// Returns true if solution can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    ///
    /// Every SolverBase which IsApplicable() for some problem config must be able to
    /// GetDefaultPerformanceConfig() so that GetSolution() would return valid
    /// solution for a problem (i.e. convolution). In other words, if a Solution
    /// says "I'm suitable" for a problem, it agrees to solve that problem correctly.
    virtual bool IsApplicable(const ExecutionContext& ctx, const boost::any& problem) const = 0;

    /// [Informative as of Sep 2020] The minimum requirement for Dynamic Solvers:
    /// Batch size and input picture size (N, W, H) must NOT be compiled into the
    /// kernel(s) that consist a Solution. These must go into the kernel as a
    /// run-time parameters.
    virtual bool IsDynamic() const { return false; }

    /// [Informative as of Sep 2020] Returns an approximated value of the expected
    /// WTI or -2.0 when this value can't be computed. Tips:
    /// * Value 1.0 corresponds to the 100% utilization of HW capabilities as
    ///   if Direct computational algorithm is used.
    /// * [Notice] WTI may exceed 1.0 for highly optimized algorithms like Winograd.
    /// * @see https://github.com/ROCm/MIOpen/issues/410
    virtual float GetWti(const ExecutionContext& ctx, const boost::any& problem) const = 0;

    // Returns the workspace size required by the solver for a given ExecutionContext
    virtual size_t GetWorkspaceSize(const ExecutionContext& ctx,
                                    const boost::any& problem) const = 0;

    // Must return true if a Solver has its own implementation of GetWorkspaceSize().
    virtual bool MayNeedWorkspace() const { return false; }

protected:
    template <class Solver>
    static const std::string& GetSolverDbId()
    {
        static const auto result = ComputeSolverDbId(get_type_name<Solver>());
        return result;
    }
    SolverBase()                  = default;
    SolverBase(const SolverBase&) = default;

private:
    static std::string ComputeSolverDbId(const std::string& type_name)
    {
        auto idx  = type_name.find_last_of(':');
        auto name = type_name.substr(idx + 1);
        std::replace(name.begin(), name.end(), ',', '-');
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());

        return name;
    }
};

template <class Context, class Problem>
struct SolverMixin : SolverBase
{
    static_assert(std::is_base_of<ExecutionContext, Context>{},
                  "Context must be derived of ExecutionContext");

    virtual bool IsApplicable(const Context&, const Problem&) const = 0;
    virtual float GetWti(const Context&, const Problem&) const { return -2.0f; };
    virtual size_t GetWorkspaceSize(const Context&, const Problem&) const { return 0; };

    bool IsApplicable(const ExecutionContext& ctx, const boost::any& problem) const final
    {
        return IsApplicable(dynamic_cast<const Context&>(ctx),
                            boost::any_cast<const Problem&>(problem));
    }

    float GetWti(const ExecutionContext& ctx, const boost::any& problem) const final
    {
        return GetWti(dynamic_cast<const Context&>(ctx), boost::any_cast<const Problem&>(problem));
    }

    size_t GetWorkspaceSize(const ExecutionContext& ctx, const boost::any& problem) const final
    {
        return GetWorkspaceSize(dynamic_cast<const Context&>(ctx),
                                boost::any_cast<const Problem&>(problem));
    }
};

/// Base class for non tunable solvers
template <class Context, class Problem>
struct NonTunableSolverBase : SolverMixin<Context, Problem>
{
    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    virtual ConvSolution GetSolution(const Context&, const Problem&) const = 0;
    virtual InvokerFactory GetInvokerFactory(const Context& ctx, const Problem& problem) const
    {
        return *GetSolution(ctx, problem).invoker_factory;
    }
};

struct TunableSolverTrait
{
};

/// Base class for tunable solvers
template <class Context, class Problem>
struct TunableSolverBase : SolverMixin<Context, Problem>, TunableSolverTrait
{
    /// Initializes performance config to the default values.
    /// The function may involve some heuristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any
    GetDefaultPerformanceConfig(const Context& ctx, const Problem& problem, int) const = 0;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const Context& ctx,
                                          const Problem& problem,
                                          const PerfConfig& config) const = 0;

    /// Search
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any Search(const Context& ctx,
                              const Problem& problem,
                              const AnyInvokeParams& invoke_ctx,
                              int) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution
    GetSolution(const Context& ctx, const Problem& problem, const PerfConfig& config) const = 0;
    virtual InvokerFactory
    GetInvokerFactory(const Context& ctx, const Problem& problem, const PerfConfig& config) const
    {
        return *GetSolution(ctx, problem, config).invoker_factory;
    }
};

template <class Context, class Problem, class PerformanceConfig>
struct TunableSolverMixin : TunableSolverBase<Context, Problem>
{
    static_assert(std::is_base_of<PerfConfig, PerformanceConfig>{},
                  "PerformanceConfig must be derived of PerfConfig");

    virtual PerformanceConfig GetDefaultPerformanceConfig(const Context&, const Problem&) const = 0;
    virtual bool
    IsValidPerformanceConfig(const Context&, const Problem&, const PerformanceConfig&) const = 0;
    virtual PerformanceConfig
    Search(const Context&, const Problem&, const AnyInvokeParams&) const = 0;
    virtual ConvSolution
    GetSolution(const Context&, const Problem&, const PerformanceConfig&) const = 0;

    boost::any
    GetDefaultPerformanceConfig(const Context& ctx, const Problem& problem, int) const final
    {
        return GetDefaultPerformanceConfig(ctx, problem);
    }

    bool IsValidPerformanceConfig(const Context& ctx,
                                  const Problem& problem,
                                  const PerfConfig& config) const final
    {
        return IsValidPerformanceConfig(
            ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }

    boost::any Search(const Context& ctx,
                      const Problem& problem,
                      const AnyInvokeParams& invoke_ctx,
                      int) const final
    {
        return Search(ctx, problem, invoke_ctx);
    }

    ConvSolution
    GetSolution(const Context& ctx, const Problem& problem, const PerfConfig& config) const final
    {
        return GetSolution(ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }
};

template <class Solver>
struct IsTunable : std::is_base_of<TunableSolverTrait, Solver>
{
    static_assert(!std::is_same_v<Solver, TunableSolverTrait>,
                  "Raw trait shouldn't be passed, explicit type is needed");
};

namespace conv {

/// Typedef for convolution non-tunable solvers
using ConvSolver = NonTunableSolverBase<ExecutionContext, miopen::conv::ProblemDescription>;

/// Typedef for convolution tunable solvers
template <class PerformanceConfig>
using ConvTunableSolver =
    TunableSolverMixin<ExecutionContext, miopen::conv::ProblemDescription, PerformanceConfig>;

struct PerformanceConfigConvAsm3x3U : PerfConfigBase<PerformanceConfigConvAsm3x3U>
{
    int limit_wave_cnt;        // [0..9]
    int filters_per_wave;      // [1..8]
    int output_lines_per_wave; // [1..8]

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm3x3U(int lwc, int fpw, int olpw);
    PerformanceConfigConvAsm3x3U() : PerformanceConfigConvAsm3x3U(-1, -1, -1) {}
    PerformanceConfigConvAsm3x3U(bool) : PerformanceConfigConvAsm3x3U(0, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.filters_per_wave, "filters_per_wave");
        f(self.output_lines_per_wave, "output_lines_per_wave");
    }

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigConvAsm3x3U& other) const;
};

struct ConvAsm3x3U final : ConvTunableSolver<PerformanceConfigConvAsm3x3U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm3x3U>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm3x3U GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsm3x3U&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm3x3U
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigConvAsm3x3U&) const override;
};

struct PerformanceConfigConvAsm1x1U : PerfConfigBase<PerformanceConfigConvAsm1x1U>
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int read_size;        // [1..4]            <same>          <same>
    int k_mult;           // 1,[4,8,12..32]    2^n[8..32]      1,4
    int chunks_per_wave;  // [1..16]           [1..8]          <same>
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     1,4
    int n_mult;           // [1..8]            [1..4]          <same>
    int c_mult;           // 2^n[1..32]        2^n[1..4]       <same>
    int waves_c_in_group; // [1..8]            [1..4]          <same>
    int waves_k_in_group; // 1,[2,4,8]         1,[2,4,8]       <same>
    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigConvAsm1x1U(int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm1x1U(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.read_size, "read_size");
        f(self.k_mult, "k_mult");
        f(self.chunks_per_wave, "chunks_per_wave");
        f(self.chunk_size, "chunk_size");
        f(self.n_mult, "n_mult");
        f(self.c_mult, "c_mult");
        f(self.waves_c_in_group, "waves_c_in_group");
        f(self.waves_k_in_group, "waves_k_in_group");
    }

    // clang-format off
    int GetReadSize() const { return read_size; }
    int GetKMult() const { return k_mult; }
    int GetChunksPerWave() const { return chunks_per_wave; }
    int GetChunkSize() const { return chunk_size; }
    int GetNMult() const { return n_mult; }
    int GetCMult() const { return c_mult; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const { assert(chunk_size); return 64 / chunk_size; }
    // clang-format on

    MIOPEN_INTERNALS_EXPORT void StaticHeuristic(const miopen::conv::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;
    bool IsValidValue() const { return IsValidValueImpl(8); }
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription& problem) const
    {
        return IsValidImpl(problem, 8);
    }
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigConvAsm1x1U& other) const;

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    bool IsPartiallyValid(const miopen::conv::ProblemDescription& problem,
                          int sequence_length) const
    {
        return IsValidImpl(problem, sequence_length);
    }
    bool IsPartiallyValidValue(int sequence_length) const
    {
        return IsValidValueImpl(sequence_length);
    }
    bool RunParameterPredictionModel(const ExecutionContext&,
                                     const miopen::conv::ProblemDescription&);
    bool ModelApplyToken(int index, std::string value, const miopen::conv::ProblemDescription&);
#endif
    bool IsValidImpl(const miopen::conv::ProblemDescription& problem, int sequence_length) const;
    bool IsValidValueImpl(int sequence_length) const;
};

struct ConvAsm1x1U final : ConvTunableSolver<PerformanceConfigConvAsm1x1U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm1x1U>(); }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm1x1U GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsm1x1U&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm1x1U
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigConvAsm1x1U&) const override;
};

struct PerformanceConfigConvAsm1x1UV2 : PerfConfigBase<PerformanceConfigConvAsm1x1UV2>
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     <same>
    int dwords_per_ld;    // [1..4]            1,2,3           <same>
    int k_mult;           // [1..32]           8,16            1,2,3,4
    int c_mult;           // [1..32]           2^n[1..4]       <same>
    int n_mult;           // [1..32]           1,2             <same>
    int w_mult;           // [1..32]           1,2             <same>
    int h_mult;           // [1..32]           1,2             <same>
    int h_per_chunk;      // 2^n[1..64]        [2,4,8]         <same>
    int waves_k_in_group; // [1..8]            2,4             <same>
    int waves_c_in_group; // [1..8]            1,2             <same>
    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigConvAsm1x1UV2(int, int, int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1UV2()
        : PerformanceConfigConvAsm1x1UV2(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm1x1UV2(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.dwords_per_ld, "dwords_per_ld");
        f(self.k_mult, "k_mult");
        f(self.c_mult, "c_mult");
        f(self.n_mult, "n_mult");
        f(self.w_mult, "w_mult");
        f(self.h_mult, "h_mult");
        f(self.h_per_chunk, "h_per_chunk");
        f(self.waves_k_in_group, "waves_k_in_group");
        f(self.waves_c_in_group, "waves_c_in_group");
    }

    // clang-format off
    int GetChunkSize() const { return chunk_size; }
    int GetDwordsPerLd() const { return dwords_per_ld; }
    int GetCMult() const { return c_mult; }
    int GetKMult() const { return k_mult; }
    int GetNMult() const { return n_mult; }
    int GetWMult() const { return w_mult; }
    int GetHMult() const { return h_mult; }
    int GetHPerChunk() const { return h_per_chunk; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const { assert(chunk_size); return 64 / chunk_size; }
    // clang-format on

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigConvAsm1x1UV2& other) const;
};

struct ConvAsm1x1UV2 final : ConvTunableSolver<PerformanceConfigConvAsm1x1UV2>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm1x1UV2>(); }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm1x1UV2 GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsm1x1UV2&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsm1x1UV2
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigConvAsm1x1UV2&) const override;
};

struct ConvAsm5x10u2v2f1 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm5x10u2v2f1>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsm5x10u2v2b1 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm5x10u2v2b1>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsm7x7c3h224w224k64u2v2p3q3f1 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsm7x7c3h224w224k64u2v2p3q3f1>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvOclDirectFwd11x11 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclDirectFwd11x11>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvOclDirectFwdGen final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclDirectFwdGen>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct PerformanceImplicitGemm : PerfConfigBase<PerformanceImplicitGemm>
{
    int BPerBlock; // 2^n[8..16]
    int KPerBlock; // 2^n[32..128]
    int EPerBlock; // 2^n[4..16]

    int GemmNRepeat; // == 2

    int GemmMPerThreadSubC; // 2^n[2..4]
    int GemmNPerThreadSubC; // 2^n[2..4]

    int GemmMLevel0Cluster; // 2^n[1..4]
    int GemmNLevel0Cluster; // 2^n[1..4]
    int GemmMLevel1Cluster; // 2^n[1..4]
    int GemmNLevel1Cluster; // 2^n[1..4]

    int InBlockCopyClusterLengths_E;  // 2^n[4..16]
    int InBlockCopyClusterLengths_B;  // 2^n[8..16]
    int InBlockCopyClusterLengths_N1; // 2^n[1..2]
    int InBlockCopyClusterLengths_N2; // 2^n[1..4]

    int WeiBlockCopyClusterLengths_E; // 2^n[1..4]
    int WeiBlockCopyClusterLengths_K; // 2^n[16..128]

    bool use_spare_set;

    PerformanceImplicitGemm(
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, bool);

    PerformanceImplicitGemm()
        : PerformanceImplicitGemm(
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemm(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BPerBlock, "BPerBlock");
        f(self.KPerBlock, "KPerBlock");
        f(self.EPerBlock, "EPerBlock");
        f(self.GemmNRepeat, "GemmNRepeat");
        f(self.GemmMPerThreadSubC, "GemmMPerThreadSubC");
        f(self.GemmNPerThreadSubC, "GemmNPerThreadSubC");
        f(self.GemmMLevel0Cluster, "GemmMLevel0Cluster");
        f(self.GemmNLevel0Cluster, "GemmNLevel0Cluster");
        f(self.GemmMLevel1Cluster, "GemmMLevel1Cluster");
        f(self.GemmNLevel1Cluster, "GemmNLevel1Cluster");
        f(self.InBlockCopyClusterLengths_E, "InBlockCopyClusterLengths_E");
        f(self.InBlockCopyClusterLengths_N1, "InBlockCopyClusterLengths_N1");
        f(self.InBlockCopyClusterLengths_B, "InBlockCopyClusterLengths_B");
        f(self.InBlockCopyClusterLengths_N2, "InBlockCopyClusterLengths_N2");
        f(self.WeiBlockCopyClusterLengths_E, "WeiBlockCopyClusterLengths_E");
        f(self.WeiBlockCopyClusterLengths_K, "WeiBlockCopyClusterLengths_K");
    }

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceImplicitGemm& other) const;
};

struct PerformanceImplicitGemmV4R1 : public PerformanceImplicitGemm
{
    PerformanceImplicitGemmV4R1(int a,
                                int b,
                                int c,
                                int d,
                                int e,
                                int f,
                                int g,
                                int h,
                                int i,
                                int j,
                                int k,
                                int l,
                                int m,
                                int n,
                                int o,
                                int p,
                                bool q)
        : PerformanceImplicitGemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
    {
    }

    PerformanceImplicitGemmV4R1()
        : PerformanceImplicitGemmV4R1(
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R1(bool spare) : PerformanceImplicitGemm(spare) {}

    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmV4R4Fwd : PerfConfigBase<PerformanceImplicitGemmV4R4Fwd>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4Fwd(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4Fwd(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4Fwd(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4Fwd() : PerformanceImplicitGemmV4R4Fwd(-1, -1, -1, -1, -1, -1, false)
    {
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4Fwd(bool spare);

    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceImplicitGemmV4R4Fwd& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmV4R4WrW : PerfConfigBase<PerformanceImplicitGemmV4R4WrW>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4WrW(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4WrW(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4WrW(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4WrW() : PerformanceImplicitGemmV4R4WrW(-1, -1, -1, -1, -1, -1, false)
    {
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4WrW(bool spare);

    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceImplicitGemmV4R4WrW& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmBwdDataV1R1 : PerfConfigBase<PerformanceImplicitGemmBwdDataV1R1>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV1R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV1R1()
        : PerformanceImplicitGemmBwdDataV1R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV1R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV1R1(a, b, c, d, e, f, false)
    {
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV1R1(bool spare);

    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceImplicitGemmBwdDataV1R1& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ExecutionContext&,
                                                 const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ExecutionContext&,
                                                 const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmBwdDataV4R1 : PerfConfigBase<PerformanceImplicitGemmBwdDataV4R1>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV4R1()
        : PerformanceImplicitGemmBwdDataV4R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV4R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV4R1(a, b, c, d, e, f, false)
    {
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1(bool spare);

    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceImplicitGemmBwdDataV4R1& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateBlockGemmPerformanceParameters() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool> MIOPEN_INTERNALS_EXPORT
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmBwdDataV4R1Xdlops
    : PerfConfigBase<PerformanceImplicitGemmBwdDataV4R1Xdlops>
{
    int GemmNPerBlock; // 2^n[8..16]
    int GemmMPerBlock; // 2^n[32..128]
    int GemmKPerBlock; // 2^n[4..16]

    int GemmKPACKSize; // 2^[1..4]

    int GemmMPerWave;
    int GemmNPerWave;

    // GemmAThreadCopyMoreGemmK is currently a fix value, is untunable
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    bool use_spare_set;
    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmBwdDataV4R1Xdlops(int, int, int, int, int, int, bool, bool, bool);

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1Xdlops();
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1Xdlops(bool spare);
    PerformanceImplicitGemmBwdDataV4R1Xdlops(
        int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceImplicitGemmBwdDataV4R1Xdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceImplicitGemmBwdDataV4R1Xdlops& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmKPACKSize, "GemmKPACKSize");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsFastToBeUsedForTuning(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct ConvHipImplicitGemmV4R1Fwd final : ConvTunableSolver<PerformanceImplicitGemmV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R1Fwd>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R1 GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R1&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmV4R1&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R1
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct ConvHipImplicitGemmV4R4Fwd final : ConvTunableSolver<PerformanceImplicitGemmV4R4Fwd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R4Fwd>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4Fwd GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R4Fwd&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4Fwd
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmV4R4Fwd&) const override;

private:
    static std::tuple<int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmV4R4Fwd;
};

struct PerformanceConvMlirIgemm : PerfConfigBase<PerformanceConvMlirIgemm>
{
    int BlockSize;
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerThread;
    int GemmNPerThread;
    bool use_spare_set;

    /// \ref https://github.com/ROCm/MIOpen/issues/1154
    static PerformanceConvMlirIgemm& MlirHeuristicInitRequest()
    {
        static PerformanceConvMlirIgemm heur;
        heur.SetMlirHeuristicInitRequest();
        return heur;
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm(int, int, int, int, int, int, bool);

    PerformanceConvMlirIgemm(int a, int b, int c, int d, int e, int f)
        : PerformanceConvMlirIgemm(a, b, c, d, e, f, false)
    {
    }

    PerformanceConvMlirIgemm() : PerformanceConvMlirIgemm(-1, -1, -1, -1, -1, -1, false) {}

    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm(bool spare);

    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConvMlirIgemm& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);

private:
    void SetMlirHeuristicInitRequest();
};

struct ConvMlirIgemmFwd final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmFwd>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConvMlirIgemm&) const override;
};

struct PerformanceConvMlirIgemmXdlops : PerfConfigBase<PerformanceConvMlirIgemmXdlops>
{
    int GemmMPerBlock; // 2^n[32..128]
    int GemmNPerBlock; // 2^n[8..16]
    int GemmKPerBlock; // 2^n[4..16]
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPACKSize; // 2^[1..4]

    // GemmAThreadCopyMoreGemmK is currently a fix value, is untunable
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    bool use_spare_set;

    /// \ref https://github.com/ROCm/MIOpen/issues/1154
    static PerformanceConvMlirIgemmXdlops& MlirHeuristicInitRequest()
    {
        static PerformanceConvMlirIgemmXdlops heur;
        heur.SetMlirHeuristicInitRequest();
        return heur;
    }

    MIOPEN_INTERNALS_EXPORT
    PerformanceConvMlirIgemmXdlops(int, int, int, int, int, int, bool, bool, bool);

    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops();
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops(bool spare);
    PerformanceConvMlirIgemmXdlops(int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceConvMlirIgemmXdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConvMlirIgemmXdlops& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPACKSize, "GemmKPACKSize");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);

private:
    void SetMlirHeuristicInitRequest();
};

struct ConvMlirIgemmFwdXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmFwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConvMlirIgemmXdlops&) const override;
};

struct ConvHipImplicitGemmV4R4WrW final : ConvTunableSolver<PerformanceImplicitGemmV4R4WrW>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R4WrW>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4WrW GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R4WrW&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R4WrW
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmV4R4WrW&) const override;

private:
    static std::tuple<int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmV4R4WrW;
};

struct ConvMlirIgemmWrW final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmWrW>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConvMlirIgemm&) const override;
};

struct ConvMlirIgemmWrWXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmWrWXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConvMlirIgemmXdlops&) const override;
};

struct PerformanceImplicitGemmForwardV4R4Xdlops
    : PerfConfigBase<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmForwardV4R4Xdlops(int, int, int, int, int, int, bool, bool, int);
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops();
    PerformanceImplicitGemmForwardV4R4Xdlops(bool) : PerformanceImplicitGemmForwardV4R4Xdlops() {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceImplicitGemmForwardV4R4Xdlops& other) const;

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsFastToBeUsedForTuning(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool> CalculateBlockSize() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmForwardV4R5Xdlops
    : PerfConfigBase<PerformanceImplicitGemmForwardV4R5Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmForwardV4R5Xdlops(int, int, int, int, int, int, bool, bool, int, bool);
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R5Xdlops();
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R5Xdlops(bool spare);

    PerformanceImplicitGemmForwardV4R5Xdlops(
        int a, int b, int c, int d, int e, int f, bool g, bool h, int i)
        : PerformanceImplicitGemmForwardV4R5Xdlops(a, b, c, d, e, f, g, h, i, false)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceImplicitGemmForwardV4R5Xdlops& other) const;

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsFastToBeUsedForTuning(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool> CalculateBlockSize() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    : PerfConfigBase<PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    int GemmMFactor;
    int GemmNFactor;
    int GemmKFactor;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool, int);
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm();
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm(bool)
        : PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm()
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmMFactor, "GemmMFactor");
        f(self.GemmNFactor, "GemmNFactor");
        f(self.GemmKFactor, "GemmKFactor");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm& other) const;

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsFastToBeUsedForTuning(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool> CalculateBlockSize() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmBwdV1R1Xdlops : PerfConfigBase<PerformanceImplicitGemmBwdV1R1Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmBwdV1R1Xdlops(int, int, int, int, int, int, bool, bool);
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdV1R1Xdlops();
    PerformanceImplicitGemmBwdV1R1Xdlops(bool) : PerformanceImplicitGemmBwdV1R1Xdlops() {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceImplicitGemmBwdV1R1Xdlops& other) const;

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsFastToBeUsedForTuning(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;

    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool> CalculateBlockSize() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmForwardV4R4Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R4Xdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static std::tuple<int, int, int, int>
    CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmForwardV4R4Xdlops;
};

struct ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static std::tuple<int, int, int, int, int, int, int> CalculateGemmSize(
        const miopen::conv::ProblemDescription&, int GemmMFactor, int GemmNFactor, int GemmKFactor);

    friend struct PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm;
};

struct ConvHipImplicitGemmForwardV4R5Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R5Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R5Xdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R5Xdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R5Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmForwardV4R5Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R5Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct ConvHipImplicitGemmV4R1WrW final : ConvTunableSolver<PerformanceImplicitGemmV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R1WrW>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R1 GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R1&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmV4R1&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmV4R1
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct ConvHipImplicitGemmBwdDataV1R1 final : ConvTunableSolver<PerformanceImplicitGemmBwdDataV1R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV1R1>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV1R1 GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV1R1&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV1R1
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmBwdDataV1R1&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool MayNeedWorkspace() const override { return true; }

private:
    static std::tuple<int, int, int> CalculateGemmSize(const ExecutionContext&,
                                                       const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmBwdDataV1R1;
};

struct ConvMlirIgemmBwd final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmBwd>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemm
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConvMlirIgemm&) const override;
};

struct ConvMlirIgemmBwdXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmBwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvMlirIgemmXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConvMlirIgemmXdlops&) const override;
};

struct ConvHipImplicitGemmBwdDataV4R1 final : ConvTunableSolver<PerformanceImplicitGemmBwdDataV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV4R1>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1 GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV4R1&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmBwdDataV4R1&) const override;

private:
    static int CalculateNumberOfGemm(const miopen::conv::ProblemDescription&);
    static std::tuple<int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&,
                                                       int gemm_id);

    friend struct PerformanceImplicitGemmBwdDataV4R1;
};

struct ConvHipImplicitGemmBwdDataV4R1Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmBwdDataV4R1Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV4R1Xdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1Xdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV4R1Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmBwdDataV4R1Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdDataV4R1Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static int CalculateNumberOfGemm(const miopen::conv::ProblemDescription&);
    static std::tuple<int, int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&,
                                                            int gemm_id);

    friend struct PerformanceImplicitGemmBwdDataV4R1Xdlops;
};

struct ConvHipImplicitGemmBwdDataV1R1Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmBwdV1R1Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV1R1Xdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdV1R1Xdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdV1R1Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmBwdV1R1Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmBwdV1R1Xdlops&) const override;

private:
    static std::tuple<int, int, int, int>
    CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmBwdV1R1Xdlops;
};

struct ConvAsmImplicitGemmV4R1DynamicFwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicFwd>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsmImplicitGemmV4R1DynamicFwd_1x1 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicFwd_1x1>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsmImplicitGemmV4R1DynamicWrw final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicWrw>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsmImplicitGemmGTCDynamicWrwXdlops final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicWrwXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsmImplicitGemmV4R1DynamicBwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicBwd>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsmImplicitGemmGTCDynamicFwdXdlops final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvAsmImplicitGemmGTCDynamicBwdXdlops final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicBwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
struct ConvOclDirectFwdLegacyExhaustiveSearch : ConvTunableSolver<LegacyPerformanceConfig>
{
    MIOPEN_INTERNALS_EXPORT LegacyPerformanceConfig GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT LegacyPerformanceConfig
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    template <typename Tgpu>
    LegacyPerformanceConfig SearchImpl(const ExecutionContext&,
                                       const miopen::conv::ProblemDescription&,
                                       const AnyInvokeParams& invoke_ctx) const;
};

struct ConvOclDirectFwd : ConvOclDirectFwdLegacyExhaustiveSearch
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclDirectFwd>(); }

    MIOPEN_INTERNALS_EXPORT static ConvSolution
    BaseGetSolution(const ExecutionContext&,
                    const miopen::conv::ProblemDescription&,
                    const LegacyPerformanceConfig&);

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution GetSolution(const ExecutionContext&,
                                                     const miopen::conv::ProblemDescription&,
                                                     const LegacyPerformanceConfig&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const LegacyPerformanceConfig&) const override;
};

struct ConvOclDirectFwd1x1 final : ConvOclDirectFwdLegacyExhaustiveSearch
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclDirectFwd1x1>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution GetSolution(const ExecutionContext&,
                                                     const miopen::conv::ProblemDescription&,
                                                     const LegacyPerformanceConfig&) const override;

    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const LegacyPerformanceConfig&) const override
    {
        return true;
    }
};

struct ConvBinWinograd3x3U final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBinWinograd3x3U>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvBinWinogradRxS final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBinWinogradRxS>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct PerformanceConfigConvBinWinogradRxS : PerfConfigBase<PerformanceConfigConvBinWinogradRxS>
{
    int n_groups;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvBinWinogradRxS(int n_groups_);
    PerformanceConfigConvBinWinogradRxS() : PerformanceConfigConvBinWinogradRxS(-1) {}
    PerformanceConfigConvBinWinogradRxS(bool) : PerformanceConfigConvBinWinogradRxS(1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_groups, "n_groups");
    }
    int GetNGroups() const { return n_groups; }

    template <int Winodata, int Winofilter>
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext& ctx, const miopen::conv::ProblemDescription&) const
    {
        return IsValid(ctx);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&) const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigConvBinWinogradRxS& other) const;
};

template <int Winodata, int Winofilter>
struct ConvBinWinoRxS final : ConvTunableSolver<PerformanceConfigConvBinWinogradRxS>
{
    const std::string& SolverDbId() const override { return GetSolverDbId(); }

    static const std::string& GetSolverDbId()
    {
        static const std::string dbId = std::string("ConvBinWinogradRxSf")
                                            .append(std::to_string(Winodata))
                                            .append("x")
                                            .append(std::to_string(Winofilter));
        return dbId;
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvBinWinogradRxS GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvBinWinogradRxS&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvBinWinogradRxS
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigConvBinWinogradRxS&) const override;

private:
    static size_t GetNGroups(const size_t group_conv, const size_t grid_group_size)
    {
        assert(group_conv != 0);
        return grid_group_size / group_conv;
    }
};

// Suppress misleading clang warnings
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif

extern template struct ConvBinWinoRxS<2, 3>;
extern template struct ConvBinWinoRxS<3, 2>;

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

struct ConvBinWinogradRxSf2x3g1 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSf2x3g1>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT float GetWti(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvMPBidirectWinograd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    // kernel_file_name for solver identification
    static fs::path GetSolverFileNames(int id)
    {
        static const fs::path names[3] = {"xform_bidirect_winograd_data.s",
                                          "xform_bidirect_winograd_filter.s",
                                          "xform_bidirect_winograd_out.s"};
        return names[id];
    }

    static std::string GetSolverKernelNames(int id)
    {
        static const std::string name_suffix =
            '_' + std::to_string(WinoDataH) + '_' + std::to_string(WinoDataW) + '_' +
            std::to_string(WinoFilterH) + '_' + std::to_string(WinoFilterW);
        static const std::string names[3] = {
            "miopenGcnAsmMPBidirectWinogradXformData" + name_suffix,
            "miopenGcnAsmMPBidirectWinogradXformFilter" + name_suffix,
            "miopenGcnAsmMPBidirectWinogradXformOut" + name_suffix};
        return names[id];
    }

    static int GetSolverWinoXformHWSize() { return WinoDataH + WinoFilterH - 1; }
};

// To suppress misleading clang warnings
#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif

extern template struct ConvMPBidirectWinograd<2, 3>;
extern template struct ConvMPBidirectWinograd<3, 3>;
extern template struct ConvMPBidirectWinograd<4, 3>;
extern template struct ConvMPBidirectWinograd<5, 3>;
extern template struct ConvMPBidirectWinograd<6, 3>;

#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic pop
#endif

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvMPBidirectWinograd_xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvMPBidirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override
    {
        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsDynamic() &&
               ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}
                   .IsDynamic() &&
               IsThisSolverDynamic();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops
    GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                const miopen::conv::ProblemDescription& problem) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvHipImplicitGemmForwardV4R4Xdlops{}.GetDefaultPerformanceConfig(xdlops_ctx,
                                                                                  xdlops_problem);
    }

    bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::conv::ProblemDescription& problem,
                             const PerformanceImplicitGemmForwardV4R4Xdlops& config) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsValidPerformanceConfig(
            xdlops_ctx, xdlops_problem, config);
    }

    size_t GetWorkspaceSize(const ExecutionContext& ctx,
                            const miopen::conv::ProblemDescription& problem) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>()
                   .GetWorkspaceSize(ctx, problem) +
               ConvHipImplicitGemmForwardV4R4Xdlops{}.GetWorkspaceSize(xdlops_ctx, xdlops_problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmForwardV4R4Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;

private:
    ExecutionContext
    GetTransformedConvContext(const ExecutionContext& ctx,
                              const miopen::conv::ProblemDescription& transformed_problem) const;
    miopen::conv::ProblemDescription
    GetTransformedProblem(const miopen::conv::ProblemDescription& problem) const;

    // kernel_file_name for solver identification
    static fs::path GetSolverFileNames(int id)
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverFileNames(id);
    }

    static std::string GetSolverKernelNames(int id)
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverKernelNames(id);
    }

    static int GetSolverWinoXformHWSize()
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverWinoXformHWSize();
    }

    bool IsThisSolverDynamic() const { return true; }
};

// To suppress misleading clang warnings
#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif

extern template struct ConvMPBidirectWinograd_xdlops<2, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<3, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<4, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<5, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<6, 3>;

#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic pop
#endif

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvWinograd3x3MultipassWrW final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    // kernel_file_name for solver identification
    static fs::path GetSolverFileNames(int id)
    {
        static const fs::path names[3] = {"xform_data.s", "xform_filter.s", "xform_out.s"};
        return names[id];
    }

    static std::string GetSolverKernelNames(int id)
    {
        static const std::string name_suffix =
            '_' + std::to_string(WinoDataH) + '_' + std::to_string(WinoDataW) + '_' +
            std::to_string(WinoFilterH) + '_' + std::to_string(WinoFilterW);
        static const std::string names[3] = {"miopenGcnAsmWinogradXformData" + name_suffix,
                                             "miopenGcnAsmWinogradXformFilter" + name_suffix,
                                             "miopenGcnAsmWinogradXformOut" + name_suffix};

        return names[id];
    }

    static int GetGroupCountMult() { return 4; }

    static int GetSolverWinoXformHWSize(const miopen::conv::ProblemDescription& problem, int id)
    {
        if(id == 0)
        {
            return WinoDataH +
                   (WinoFilterH - 1) * (WinoDataH == 7 ? 2 : problem.GetKernelStrideH());
        }
        else
        {
            return WinoDataW +
                   (WinoFilterW - 1) * (WinoDataW == 7 ? 2 : problem.GetKernelStrideW());
        }
    }

private:
    InvokerFactory PrepareInvokerFactory(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&,
                                         std::size_t ws_sz) const;
};

// To suppress misleading clang warnings
#if defined(__clang__) && defined(CONV_MULTIPASS_WINO3X3WRW_CPP)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif

extern template struct ConvWinograd3x3MultipassWrW<3, 2>;
extern template struct ConvWinograd3x3MultipassWrW<3, 3>;
extern template struct ConvWinograd3x3MultipassWrW<3, 4>;
extern template struct ConvWinograd3x3MultipassWrW<3, 5>;
extern template struct ConvWinograd3x3MultipassWrW<3, 6>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2, 1, 1>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3, 1, 1>;
extern template struct ConvWinograd3x3MultipassWrW<5, 3>;
extern template struct ConvWinograd3x3MultipassWrW<5, 4>;

#if defined(__clang__) && defined(CONV_MULTIPASS_WINO3X3WRW_CPP)
#pragma clang diagnostic pop
#endif

struct PerformanceConfigAsmDirect3x3WrW : PerfConfigBase<PerformanceConfigAsmDirect3x3WrW>
{
    int limit_wave_cnt;   // [0..9]
    int reverse_inout;    // [0..1], 1 is allowed for stride=1x1 only.
    int chunk_size;       // {16,8}, Smaller values increase register pressure.
    int k_per_wave;       // {1,2,4,8} && ((chunk_size * k_per_wave) <= 64).
                          // Higher values increase register pressure.
    int pipe_lines_depth; // [1..16] && (pipe_lines_depth <= img_h).
                          // Higher values increase register pressure.
    int n_per_group;      // [1..8] && (n_per_group <= batch_size).

    PerformanceConfigAsmDirect3x3WrW(int lwc, int rio, int csz, int kpw, int pld, int npg);
    PerformanceConfigAsmDirect3x3WrW() : PerformanceConfigAsmDirect3x3WrW(-1, -1, -1, -1, -1, -1) {}
    PerformanceConfigAsmDirect3x3WrW(bool) : PerformanceConfigAsmDirect3x3WrW(0, 0, 8, 1, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.reverse_inout, "reverse_inout");
        f(self.chunk_size, "chunk_size");
        f(self.k_per_wave, "k_per_wave");
        f(self.pipe_lines_depth, "pipe_lines_depth");
        f(self.n_per_group, "n_per_group");
    }

    // clang-format off
    int GetLimitWaveCnt() const { return limit_wave_cnt; }
    int GetReverseInout() const { return reverse_inout; }
    int GetChunkSize() const { return chunk_size; }
    int GetKPerWave() const { return k_per_wave; }
    int GetPipeLinesDepth() const { return pipe_lines_depth; }
    int GetNPerGroup() const { return n_per_group; }
    int GetCPerWave() const { assert(chunk_size); return 64 / chunk_size; } // clang-format on

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigAsmDirect3x3WrW& other) const;
};

struct ConvAsmBwdWrW3x3 final : ConvTunableSolver<PerformanceConfigAsmDirect3x3WrW>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsmBwdWrW3x3>(); }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmDirect3x3WrW GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigAsmDirect3x3WrW&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmDirect3x3WrW
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmDirect3x3WrW& config) const override;
};

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoFuryRxS final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvWinoFuryRxS<Winodata, Winofilter>>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT float GetWti(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

// Suppress misleading clang warnings
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif

extern template struct ConvWinoFuryRxS<2, 3>;
// extern template struct ConvWinoFuryRxS<3, 2>;

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

struct PerformanceConfigConvAsmBwdWrW1x1 : PerfConfigBase<PerformanceConfigConvAsmBwdWrW1x1>
{

    int chunk_size;    // {1,2,4,8,16}
    int c_per_gpr;     // {1,2,4,8,16}
    int c_mult;        // {1,2,4,8,16}
    int k_per_gpr;     // {1,2,4,8,16}
    int k_mult;        // {1,2,4,8,16}
    int n_per_gpr;     // {1,2,4}
    int n_part_cnt;    // [1..8]
    int read_size;     // [1..4]
    int short_store;   // {0,1}
    int data_prefetch; // [0..4]
    bool use_spare_set;

    /// The following conditions must be met.
    ///
    /// Shader design-related constraints:
    /// - (A) (chunk_size * c_per_gpr) == 16
    /// - (B) k_per_gpr <= c_per_gpr
    /// - (C) (c_mult > 1 || k_mult > 1)
    ///         ? ((fwd_C % (c_per_gpr * c_mult) == 0) && (fwd_K % (k_per_gpr * k_mult) == 0))
    ///         : (true)
    ///
    /// Resource-related constraints:
    /// - (D) c_mult * k_mult * k_per_gpr + 9 + (c_mult + k_mult) * read_size * pipe_depth <= 256
    ///
    /// Where:
    /// - fwd_C := Num input channels for forward convolution (-c).
    ///   For backward, this is actually n_outputs.
    /// - fwd_K := Num output channels for forward convolution (-k).
    ///   For backward, this is actually n_inputs.

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsmBwdWrW1x1(int chunk_size_,
                                                              int c_per_gpr_,
                                                              int c_mult_,
                                                              int k_per_gpr_,
                                                              int k_mult_,
                                                              int n_per_gpr_,
                                                              int n_part_cnt_,
                                                              int read_size_,
                                                              int short_store_,
                                                              int data_prefetch_,
                                                              bool);
    PerformanceConfigConvAsmBwdWrW1x1()
        : PerformanceConfigConvAsmBwdWrW1x1(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsmBwdWrW1x1(bool spare)
        : PerformanceConfigConvAsmBwdWrW1x1(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.c_per_gpr, "c_per_gpr");
        f(self.c_mult, "c_mult");
        f(self.k_per_gpr, "k_per_gpr");
        f(self.k_mult, "k_mult");
        f(self.n_per_gpr, "n_per_gpr");
        f(self.n_part_cnt, "n_part_cnt");
        f(self.read_size, "read_size");
        f(self.short_store, "short_store");
        f(self.data_prefetch, "data_prefetch");
    }

    // clang-format off
    int GetChunkSize() const { return chunk_size; }
    int GetCPerGpr() const { return c_per_gpr; }
    int GetCMult() const { return c_mult; }
    int GetKPerGpr() const { return k_per_gpr; }
    int GetKMult() const { return k_mult; }
    int GetNPerGpr() const { return n_per_gpr; }
    int GetNPartCnt() const { return n_part_cnt; }
    int GetHWPerGpr() const {   assert(c_per_gpr); assert(n_per_gpr); assert(chunk_size);
                                return wave_size / (c_per_gpr * n_per_gpr * chunk_size); } // "hw" stands for "height-and-width".
    int GetReadSize() const { return read_size; }
    int GetShortStore() const {return short_store; }
    int GetDataPrefetch() const { return data_prefetch; }
    // clang-format on

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigConvAsmBwdWrW1x1& other) const;
};

struct ConvAsmBwdWrW1x1 final : ConvTunableSolver<PerformanceConfigConvAsmBwdWrW1x1>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsmBwdWrW1x1>(); }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsmBwdWrW1x1 MIOPEN_INTERNALS_EXPORT
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsmBwdWrW1x1&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvAsmBwdWrW1x1
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigConvAsmBwdWrW1x1&) const override;
};

/// N_BATCH_LOOPS - {1,2,4,8,16} Num batches processed in single workitem.
///     Required workspace size depends on it. However there is a restriction in the internal
///     Solver API that this shouldn't be so. Therefore the family of Solvers created.
///     Each Solver in the family has constant value of this parameter.
template <int N_BATCH_LOOPS>
struct PerformanceConfigConvOclBwdWrw2
    : PerfConfigBase<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>
{
    // Num waves involved a workgroup.
    int n_waves = -1; // {1,2,4,8}
    // Num values to read in a workitem (read_unit).
    int read_size = -1; // [6..12]
    // Num of output channels (top/bottom layer in forward/backward direction)
    // that share the same input channel in single workgroup.
    // Also represents number of output channels in single tile.
    int n_out_channels_per_tile = -1; // {1,2,4,8}
    // How many tiles of output channels are processed in a single workgroup?
    // n_out_channels_in_lcl * n_out_channels_tiles = total number of
    // output channels processed in single workgroup.
    int n_out_channels_tiles = -1; // {1,2,4,8}
    // Num of output rows processed in a single iteration of loop in a workitem
    // (N_ALIGNED_OUT_SCAN_BLK).
    int n_out_rows_in_lcl = -1; // [2..11]

    PerformanceConfigConvOclBwdWrw2(int nw, int rs, int nocpt, int noct, int noril)
        : n_waves(nw),
          read_size(rs),
          n_out_channels_per_tile(nocpt),
          n_out_channels_tiles(noct),
          n_out_rows_in_lcl(noril)
    {
    }
    PerformanceConfigConvOclBwdWrw2() {}
    PerformanceConfigConvOclBwdWrw2(bool) : PerformanceConfigConvOclBwdWrw2(1, 6, 1, 1, 2) {}
    // spare_set is not used in this solver.

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_waves, "n_waves");
        f(self.read_size, "read_size");
        f(self.n_out_channels_per_tile, "n_out_channels_per_tile");
        f(self.n_out_channels_tiles, "n_out_channels_tiles");
        f(self.n_out_rows_in_lcl, "n_out_rows_in_lcl");
    }

    // clang-format off
    int GetNumWaves() const { return n_waves; }
    int GetReadSize() const { return read_size; }
    int GetNumOutChannelsPerTile() const { return n_out_channels_per_tile; }
    int GetNumOutChannelTiles() const { return n_out_channels_tiles; }
    int GetNumOutRowsPerIterPerWork() const { return n_out_rows_in_lcl; } // clang-format on

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& other) const;
};

template <int N_BATCH_LOOPS>
struct ConvOclBwdWrW2 : ConvTunableSolver<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>
{
    const std::string& SolverDbId() const override
    {
        return this->template GetSolverDbId<ConvOclBwdWrW2<N_BATCH_LOOPS>>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&) const override;

protected:
    bool IsApplicableBase(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
};

// To suppress misleading clang warnings
#if defined(__clang__) && defined(CONV_OCL_DIR2D_BWDWRW_2_CPP)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif

extern template struct PerformanceConfigConvOclBwdWrw2<1>;
extern template struct PerformanceConfigConvOclBwdWrw2<2>;
extern template struct PerformanceConfigConvOclBwdWrw2<4>;
extern template struct PerformanceConfigConvOclBwdWrw2<8>;
extern template struct PerformanceConfigConvOclBwdWrw2<16>;

extern template struct ConvOclBwdWrW2<1>;
extern template struct ConvOclBwdWrW2<2>;
extern template struct ConvOclBwdWrW2<4>;
extern template struct ConvOclBwdWrW2<8>;
extern template struct ConvOclBwdWrW2<16>;

#if defined(__clang__) && defined(CONV_OCL_DIR2D_BWDWRW_2_CPP)
#pragma clang diagnostic pop
#endif

/// A separate solver from ConvOclBwdWrW2 to disable auto-tuning for certain configs.
/// Basically, this is *hack* for non-group 3x3 and 1x1 cases.
/// It is assumed that Solutions provided by the ConvOclBwdWrW2 solver
/// would never beat 3x3 and 1x1 assembly WrW kernels, even after tuning.
struct ConvOclBwdWrW2NonTunable final : ConvOclBwdWrW2<1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclBwdWrW2NonTunable>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution GetSolution(const ExecutionContext&,
                                                     const miopen::conv::ProblemDescription&) const;
    InvokerFactory GetInvokerFactory(const ExecutionContext& ctx,
                                     const miopen::conv::ProblemDescription& problem) const
    {
        return *GetSolution(ctx, problem).invoker_factory;
    }

private:
    // This function dervied from ConvOclBwdWrW2 is declared private
    // so that this solver is not marked searchable/tunable.
    using ConvOclBwdWrW2<1>::GetDefaultPerformanceConfig;
    using ConvOclBwdWrW2<1>::GetSolution;
    using ConvOclBwdWrW2<1>::GetInvokerFactory;
};

struct ConvOclBwdWrW53 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclBwdWrW53>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvOclBwdWrW1x1 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclBwdWrW1x1>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct fft final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<fft>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct PerformanceImplicitGemmWrwV4R4Xdlops : PerfConfigBase<PerformanceImplicitGemmWrwV4R4Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmK;
    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmWrwV4R4Xdlops(int, int, int, int, int, int, bool, bool, bool);
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops();
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops(bool spare);
    PerformanceImplicitGemmWrwV4R4Xdlops(int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceImplicitGemmWrwV4R4Xdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmK, "GemmBThreadCopyMoreGemmK");
    }

    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceImplicitGemmWrwV4R4Xdlops& other) const;

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsReallyValid(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsFastToBeUsedForTuning(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;

    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ExecutionContext&,
                                   const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool> CalculateBlockSize() const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool>
    CalculateGridSize(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmWrwV4R4Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmWrwV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmWrwV4R4Xdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmWrwV4R4Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmWrwV4R4Xdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    : PerfConfigBase<PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    int GemmMFactor;
    int GemmNFactor;
    int GemmKTotalFactor;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmK;

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool);
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm();
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm(bool)
        : PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm()
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmMFactor, "GemmMFactor");
        f(self.GemmNFactor, "GemmNFactor");
        f(self.GemmKTotalFactor, "GemmKTotalFactor");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmK, "GemmBThreadCopyMoreGemmK");
    }

    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm& other) const;

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    MIOPEN_INTERNALS_EXPORT bool IsValid(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool IsReallyValid(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsFastToBeUsedForTuning(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;

    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ExecutionContext&,
                                   const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ExecutionContext&,
                                            const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm final
    : ConvTunableSolver<PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct PerformanceConvCkIgemmFwdV6r1DlopsNchw
    : PerfConfigBase<PerformanceConvCkIgemmFwdV6r1DlopsNchw>
{
    int ck_tunable_list_id;

    PerformanceConvCkIgemmFwdV6r1DlopsNchw(int a) : ck_tunable_list_id(a) {}

    PerformanceConvCkIgemmFwdV6r1DlopsNchw() : PerformanceConvCkIgemmFwdV6r1DlopsNchw(-1) {}

    PerformanceConvCkIgemmFwdV6r1DlopsNchw(bool) : PerformanceConvCkIgemmFwdV6r1DlopsNchw(0) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.ck_tunable_list_id, "ck_tunable_list_id");
    }

    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config) const
    {
        return ck_tunable_list_id == config.ck_tunable_list_id;
    }
};

struct ConvCkIgemmFwdV6r1DlopsNchw final : ConvTunableSolver<PerformanceConvCkIgemmFwdV6r1DlopsNchw>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvCkIgemmFwdV6r1DlopsNchw>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsDynamic() const override { return false; }
    MIOPEN_INTERNALS_EXPORT PerformanceConvCkIgemmFwdV6r1DlopsNchw GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConvCkIgemmFwdV6r1DlopsNchw
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const override;
};

struct ConvDirectNaiveConvFwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvFwd>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.01f;
    }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvDirectNaiveConvBwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvBwd>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.01f;
    }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct ConvDirectNaiveConvWrw final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvWrw>();
    }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.01f;
    }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct GemmFwdBase : ConvSolver
{
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT float GetWti(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const override;

private:
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwd1x1_0_2;
    friend struct GemmFwd1x1_0_1_int8;
    friend struct GemmFwd1x1_0_1;
    friend struct GemmFwdRest;
};

struct GemmFwd1x1_0_2 final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_2>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwdRest;
};

struct GemmFwd1x1_0_1_int8 final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_1_int8>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwdRest;
};

struct GemmFwd1x1_0_1 final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_1>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwdRest;
};

struct GemmFwdRest final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwdRest>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct GemmBwdBase : ConvSolver
{
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT float GetWti(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const override;

private:
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmBwd1x1_stride2;
    friend struct GemmBwd1x1_stride1;
    friend struct GemmBwdRest;
};

struct GemmBwd1x1_stride2 final : GemmBwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwd1x1_stride2>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmBwdRest;
};

struct GemmBwd1x1_stride1 final : GemmBwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwd1x1_stride1>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&,
                 const miopen::conv::ProblemDescription& problem) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution GetSolution(
        const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const override;

    friend struct GemmBwdRest;
};

struct GemmBwdRest final : GemmBwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwdRest>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct GemmWrwBase : ConvSolver
{
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT float GetWti(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&) const override;

private:
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmWrw1x1_stride1;
    friend struct GemmWrwUniversal;
};

struct GemmWrw1x1_stride1 final : GemmWrwBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmWrw1x1_stride1>(); }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    friend struct GemmWrwUniversal;
};

struct GemmWrwUniversal final : GemmWrwBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmWrwUniversal>(); }

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
};

struct PerformanceConfigAsmImplicitGemmGTC : PerfConfigBase<PerformanceConfigAsmImplicitGemmGTC>
{
    std::string direction;
    std::string tensor_layout;
    std::string precision;
    int nxb;
    int nxe;

    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;

    int wave_tile_m;
    int wave_tile_n;
    int wave_tile_k;
    int wave_step_m;
    int wave_step_n;
    int wave_repeat_m;
    int wave_repeat_n;

    int multihead;
    int vector_store;
    int gemm_k_global_split;
    int merge_e;
    int tensor_a_pass_through;

    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;

    bool use_spare_set;
    int index;

    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTC(std::string dir,
                                                                std::string layout,
                                                                std::string prec,
                                                                int b,
                                                                int e,
                                                                int mpb,
                                                                int npb,
                                                                int kpb,
                                                                int wtm,
                                                                int wtn,
                                                                int wtk,
                                                                int wsm,
                                                                int wsn,
                                                                int wrm,
                                                                int wrn,
                                                                int mh,
                                                                int vs,
                                                                int gks,
                                                                int me,
                                                                int pta,
                                                                std::initializer_list<int> ta_t,
                                                                std::initializer_list<int> ta_c,
                                                                std::initializer_list<int> tb_t,
                                                                std::initializer_list<int> tb_c,
                                                                bool spare = false);
    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTC(std::string dir,
                                                                std::string layout,
                                                                miopenDataType_t prec,
                                                                int b,
                                                                int e,
                                                                int mpb,
                                                                int npb,
                                                                int kpb,
                                                                int wtm,
                                                                int wtn,
                                                                int wtk,
                                                                int wsm,
                                                                int wsn,
                                                                int wrm,
                                                                int wrn,
                                                                int mh,
                                                                int vs,
                                                                int gks,
                                                                int me,
                                                                int pta,
                                                                std::initializer_list<int> ta_t,
                                                                std::initializer_list<int> ta_c,
                                                                std::initializer_list<int> tb_t,
                                                                std::initializer_list<int> tb_c,
                                                                bool spare = false);
    PerformanceConfigAsmImplicitGemmGTC()
        : PerformanceConfigAsmImplicitGemmGTC("fwd",
                                              "nchw",
                                              "fp32",
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTC("fwd",
                                              "nchw",
                                              "fp32",
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.direction, "dir");
        f(self.tensor_layout, "lyt");
        f(self.precision, "pre");
        f(self.nxb, "nxb");
        f(self.nxe, "nxe");
        f(self.gemm_m_per_block, "mpb");
        f(self.gemm_n_per_block, "npb");
        f(self.gemm_k_per_block, "kpb");

        f(self.wave_tile_m, "wtm");
        f(self.wave_tile_n, "wtn");
        f(self.wave_tile_k, "wtk");
        f(self.wave_step_m, "wsm");
        f(self.wave_step_n, "wsn");
        f(self.wave_repeat_m, "wrm");
        f(self.wave_repeat_n, "wrn");

        f(self.multihead, "mh");
        f(self.vector_store, "vs");
        f(self.gemm_k_global_split, "gks");
        f(self.merge_e, "me");
        f(self.tensor_a_pass_through, "pta");

        f(self.tensor_a_thread_lengths[0], "ta0");
        f(self.tensor_a_thread_lengths[1], "ta1");
        f(self.tensor_a_thread_lengths[2], "ta2");
        f(self.tensor_a_thread_lengths[3], "ta3");

        f(self.tensor_a_cluster_lengths[0], "ca0");
        f(self.tensor_a_cluster_lengths[1], "ca1");
        f(self.tensor_a_cluster_lengths[2], "ca2");
        f(self.tensor_a_cluster_lengths[3], "ca3");

        f(self.tensor_b_thread_lengths[0], "tb0");
        f(self.tensor_b_thread_lengths[1], "tb1");
        f(self.tensor_b_thread_lengths[2], "tb2");
        f(self.tensor_b_thread_lengths[3], "tb3");

        f(self.tensor_b_cluster_lengths[0], "cb0");
        f(self.tensor_b_cluster_lengths[1], "cb1");
        f(self.tensor_b_cluster_lengths[2], "cb2");
        f(self.tensor_b_cluster_lengths[3], "cb3");
        f(self.index, "index");
    }

    // Chilrden must provide support for ComputedContainer.
    void HeuristicInit(const ExecutionContext&)                                          = delete;
    bool SetNextValue(const miopen::conv::ProblemDescription&)                           = delete;
    bool IsValidValue() const                                                            = delete;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const = delete;

    MIOPEN_INTERNALS_EXPORT bool IsDefaultConstructed() const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigAsmImplicitGemmGTC& other) const;
    MIOPEN_INTERNALS_EXPORT void CopyParameters(const PerformanceConfigAsmImplicitGemmGTC& other);
    MIOPEN_INTERNALS_EXPORT std::string ToString() const override;
    MIOPEN_INTERNALS_EXPORT std::string ToKernelName(const ExecutionContext&) const;
    MIOPEN_INTERNALS_EXPORT int BlockSize() const;
};

struct PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription& config);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC&) const override;
};

struct PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC&) const override;
};

struct PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT size_t ComputeKernelOccupancy() const;

private:
    void SetParamsForKSplit(const miopen::conv::ProblemDescription& problem,
                            const size_t& occupancy);
};

struct ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC&) const override;
};

struct PerformanceConfigAsmImplicitGemmGTCvector
    : PerfConfigBase<PerformanceConfigAsmImplicitGemmGTCvector>
{
    std::string direction;
    std::string tensor_layout;
    std::string precision;
    int nxb;
    int nxe;

    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;

    int lanegroup_tile_m;
    int lanegroup_tile_n;
    int lanegroup_wave_m;
    int lanegroup_wave_n;
    int lanegroup_repeat_m;
    int lanegroup_repeat_n;

    int vector_c;

    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;

    bool use_spare_set;
    int index;

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigAsmImplicitGemmGTCvector(std::string dir,
                                              std::string layout,
                                              std::string prec,
                                              int b,
                                              int e,
                                              int mpb,
                                              int npb,
                                              int kpb,
                                              int lgtm,
                                              int lgtn,
                                              int lgpwm,
                                              int lgpwn,
                                              int lgrm,
                                              int lgrn,
                                              int vec_c,
                                              std::initializer_list<int> ta_t,
                                              std::initializer_list<int> ta_c,
                                              std::initializer_list<int> tb_t,
                                              std::initializer_list<int> tb_c,
                                              bool spare = false);

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigAsmImplicitGemmGTCvector(std::string dir,
                                              std::string layout,
                                              miopenDataType_t prec,
                                              int b,
                                              int e,
                                              int mpb,
                                              int npb,
                                              int kpb,
                                              int lgtm,
                                              int lgtn,
                                              int lgpwm,
                                              int lgpwn,
                                              int lgrm,
                                              int lgrn,
                                              int vec_c,
                                              std::initializer_list<int> ta_t,
                                              std::initializer_list<int> ta_c,
                                              std::initializer_list<int> tb_t,
                                              std::initializer_list<int> tb_c,
                                              bool spare = false);

    PerformanceConfigAsmImplicitGemmGTCvector()
        : PerformanceConfigAsmImplicitGemmGTCvector("fwd",
                                                    "nchwc_kcyxc",
                                                    "Half",
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCvector(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCvector("fwd",
                                                    "nchwc_kcyxc",
                                                    "Half",
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.direction, "dir");
        f(self.tensor_layout, "lyt");
        f(self.precision, "pre");
        f(self.nxb, "nxb");
        f(self.nxe, "nxe");
        f(self.gemm_m_per_block, "mpb");
        f(self.gemm_n_per_block, "npb");
        f(self.gemm_k_per_block, "kpb");

        f(self.lanegroup_tile_m, "lgtm");
        f(self.lanegroup_tile_n, "lgtn");
        f(self.lanegroup_wave_m, "lgpwm");
        f(self.lanegroup_wave_n, "lgpwn");
        f(self.lanegroup_repeat_m, "lgrm");
        f(self.lanegroup_repeat_n, "lgrn");

        f(self.vector_c, "vec_c");

        f(self.tensor_a_thread_lengths[0], "ta0");
        f(self.tensor_a_thread_lengths[1], "ta1");
        f(self.tensor_a_thread_lengths[2], "ta2");
        f(self.tensor_a_thread_lengths[3], "ta3");

        f(self.tensor_a_cluster_lengths[0], "ca0");
        f(self.tensor_a_cluster_lengths[1], "ca1");
        f(self.tensor_a_cluster_lengths[2], "ca2");
        f(self.tensor_a_cluster_lengths[3], "ca3");

        f(self.tensor_b_thread_lengths[0], "tb0");
        f(self.tensor_b_thread_lengths[1], "tb1");
        f(self.tensor_b_thread_lengths[2], "tb2");
        f(self.tensor_b_thread_lengths[3], "tb3");

        f(self.tensor_b_cluster_lengths[0], "cb0");
        f(self.tensor_b_cluster_lengths[1], "cb1");
        f(self.tensor_b_cluster_lengths[2], "cb2");
        f(self.tensor_b_cluster_lengths[3], "cb3");
        f(self.index, "index");
    }

    // Chilrden must provide support for ComputedContainer.
    void HeuristicInit(const ExecutionContext&)                                          = delete;
    bool SetNextValue(const miopen::conv::ProblemDescription&)                           = delete;
    bool IsValidValue() const                                                            = delete;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const = delete;

    MIOPEN_INTERNALS_EXPORT bool IsDefaultConstructed() const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigAsmImplicitGemmGTCvector& other) const;
    MIOPEN_INTERNALS_EXPORT void
    CopyParameters(const PerformanceConfigAsmImplicitGemmGTCvector& other);
    MIOPEN_INTERNALS_EXPORT std::string ToString() const override;
    MIOPEN_INTERNALS_EXPORT std::string ToKernelName(const ExecutionContext&) const;
    MIOPEN_INTERNALS_EXPORT int BlockSize() const;
};
struct PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC : PerformanceConfigAsmImplicitGemmGTCvector
{

    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int lgtm,
                                                     int lgtn,
                                                     int lgpwm,
                                                     int lgpwn,
                                                     int lgrm,
                                                     int lgrn,
                                                     int vec_c,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTCvector(dir,
                                                    layout,
                                                    prec,
                                                    b,
                                                    e,
                                                    mpb,
                                                    npb,
                                                    kpb,
                                                    lgtm,
                                                    lgtn,
                                                    lgpwm,
                                                    lgpwn,
                                                    lgrm,
                                                    lgrn,
                                                    vec_c,
                                                    ta_t,
                                                    ta_c,
                                                    tb_t,
                                                    tb_c,
                                                    spare)
    {
    }

    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int lgtm,
                                                     int lgtn,
                                                     int lgpwm,
                                                     int lgpwn,
                                                     int lgrm,
                                                     int lgrn,
                                                     int vec_c,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTCvector(dir,
                                                    layout,
                                                    prec,
                                                    b,
                                                    e,
                                                    mpb,
                                                    npb,
                                                    kpb,
                                                    lgtm,
                                                    lgtn,
                                                    lgpwm,
                                                    lgpwn,
                                                    lgrm,
                                                    lgrn,
                                                    vec_c,
                                                    ta_t,
                                                    ta_c,
                                                    tb_t,
                                                    tb_c,
                                                    spare)
    {
    }

    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC()
        : PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC("fwd",
                                                           "nchwc_kcyxc",
                                                           "Half",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC("fwd",
                                                           "nchwc_kcyxc",
                                                           "Half",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC>();
    }
    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC&) const override;
};

struct PerformanceConfigHipImplicitGemmFwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmFwdXdlops>
{
    int index             = 0;
    std::string kernel_id = "";
    std::vector<std::string> valid_kernels;

    PerformanceConfigHipImplicitGemmFwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }

    PerformanceConfigHipImplicitGemmFwdXdlops() = default;

    explicit PerformanceConfigHipImplicitGemmFwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmFwdXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmFwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmFwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmFwdXdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmFwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmFwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmFwdXdlops&) const override;
    /// \anchor igemm_get_wti_magic_number
    // Magic Number Alert:
    // Naive convolutions have GetWti() that return very small value (0.01f).
    // This allows MIOpen to use Naive Solvers if no other applicable Solvers
    // have known WTIs. Right now this means that in case of find-db miss,
    // the library will try to use Winograd or GEMM (whatever is faster according
    // to their GetWti's), but if both are not applicable, the library will
    // use Naive Solver
    // Since we would like to us CK before naive, and use it instead (because
    // we do expect that CK is faster than Naive), therefore we use a
    // value bigger than 0.01f, e.g. 0.02f.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmBwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmBwdXdlops>
{
    int index             = 0;
    std::string kernel_id = "";
    std::vector<std::string> valid_kernels;

    PerformanceConfigHipImplicitGemmBwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }

    PerformanceConfigHipImplicitGemmBwdXdlops() = default;

    explicit PerformanceConfigHipImplicitGemmBwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmBwdXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmBwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmBwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmBwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmBwdXdlops GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmBwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmBwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmBwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmGroupFwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmGroupFwdXdlops>
{
    int index             = 0;
    std::string kernel_id = "";
    std::vector<std::string> valid_kernels;

    PerformanceConfigHipImplicitGemmGroupFwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }

    PerformanceConfigHipImplicitGemmGroupFwdXdlops() = default;

    explicit PerformanceConfigHipImplicitGemmGroupFwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmGroupFwdXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmGroupFwdXdlops& other) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    std::vector<int> heuristic_indexes;
    std::unordered_map<int, std::vector<std::string>> heuristic_kernels;
    template <typename DataType>
    bool RunParameterPredictionModel(const ExecutionContext& ctx,
                                     const miopen::conv::ProblemDescription& problem);
    void InitHeuristicKernelIDs(const std::string& type);
    bool ModelApplyToken(int idx, std::string value, const std::string& arch);
#endif
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmGroupFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmGroupFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmGroupFwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmGroupFwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupFwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmGroupFwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmGroupFwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemm3DGroupFwdXdlops>
{
    int index             = 0;
    std::string kernel_id = "";
    std::vector<std::string> valid_kernels;

    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }

    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops() = default;

    explicit PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(bool)
        : PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemm3DGroupFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemm3DGroupFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemm3DGroupFwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemm3DGroupWrwXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops()
        : PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(bool)
        : PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& other) const;

private:
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemm3DGroupWrwXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemm3DGroupWrwXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemm3DGroupWrwXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemm3DGroupBwdXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops()
        : PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(bool)
        : PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemm3DGroupBwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemm3DGroupBwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemm3DGroupBwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmGroupBwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmGroupBwdXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmGroupBwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmGroupBwdXdlops()
        : PerformanceConfigHipImplicitGemmGroupBwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmGroupBwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmGroupBwdXdlops(0, "")
    {
    }

    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmGroupBwdXdlops& other) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    std::vector<int> heuristic_indexes;
    std::unordered_map<int, std::vector<std::string>> heuristic_kernels;
    template <typename DataType>
    bool RunParameterPredictionModel(const ExecutionContext& ctx,
                                     const miopen::conv::ProblemDescription& problem);
    void InitHeuristicKernelIDs();
    bool ModelApplyToken(int idx, std::string value);
#endif
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmGroupBwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmGroupBwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmGroupBwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmGroupBwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupBwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmGroupBwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmGroupBwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmGroupWrwXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmGroupWrwXdlops>
{
    int index;
    int split_k;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmGroupWrwXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmGroupWrwXdlops()
        : PerformanceConfigHipImplicitGemmGroupWrwXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmGroupWrwXdlops(bool)
        : PerformanceConfigHipImplicitGemmGroupWrwXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmGroupWrwXdlops& other) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    std::vector<int> heuristic_indexes;
    std::unordered_map<int, std::vector<std::string>> heuristic_kernels;
    template <typename DataType>
    bool RunParameterPredictionModel(const ExecutionContext& ctx,
                                     const miopen::conv::ProblemDescription& problem);
    void InitHeuristicKernelIDs(const std::string& type);
    bool ModelApplyToken(int idx,
                         std::string value,
                         const std::string& arch,
                         const miopen::conv::ProblemDescription& problem);
#endif
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmGroupWrwXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmGroupWrwXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmGroupWrwXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmGroupWrwXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupWrwXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmGroupWrwXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmGroupWrwXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    template <typename DataType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops>
{
    int index             = 0;
    std::string kernel_id = "";
    std::vector<std::string> valid_kernels;

    PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }

    PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops() = default;

    explicit PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops& other) const;

private:
    template <typename DataType, typename ComputeType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType, typename ComputeType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmF16F8F16FwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmF16F8F16FwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType, typename ComputeType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops()
        : PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops& other) const;

private:
    template <typename DataType, typename OutComputeType, typename WeiComputeType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType, typename OutComputeType, typename WeiComputeType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmF16F8F16BwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmF16F8F16BwdXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType, typename OutComputeType, typename WeiComputeType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops()
        : PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops(bool)
        : PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops(0, "")
    {
    }
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool
    operator==(const PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops& other) const;

private:
    template <typename DataType, typename OutComputeType, typename InComputeType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType, typename OutComputeType, typename InComputeType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
};

struct ConvHipImplicitGemmF16F8F16WrwXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmF16F8F16WrwXdlops>();
    }

    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    MIOPEN_INTERNALS_EXPORT bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops&) const override;
    MIOPEN_INTERNALS_EXPORT PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    MIOPEN_INTERNALS_EXPORT bool
    IsApplicable(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemmF16F8F16WrwXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType, typename OutComputeType, typename InComputeType>
    bool CheckCKApplicability(const miopen::conv::ProblemDescription&) const;
};

} // namespace conv

// Use struct as a syntactic sugar to make the intent as clear as possible.
struct ThisSolverIsDeprecatedStatic
{
    MIOPEN_INTERNALS_EXPORT static bool IsDisabled(const ExecutionContext& ctx);
};

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_SOLVER_HPP_
