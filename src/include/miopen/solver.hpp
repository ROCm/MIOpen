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

#include <miopen/config.h>

#include <miopen/conv_solution.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/rocm_features.hpp>
#include <miopen/type_name.hpp>
#include <miopen/miopen.h>
#include <miopen/buffer_info.hpp>
#include <miopen/performance_config.hpp>

#include <boost/any.hpp>

#include <memory>
#include <string>
#include <vector>
#include <ostream>
#include <algorithm>
#include <initializer_list>

namespace miopen {

namespace debug {

/// If set to true, then always enable ConvDirectNaive* solver, regardless of environment value
/// MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_* that control enable/disable of these solvers.
/// Currently used during driver using naive kernel as gpu reference.
extern bool
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

    /// In some instances ( particularly fusions) the fused solver might like to
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
    /// * @see https://github.com/ROCmSoftwarePlatform/MIOpen/issues/410
    virtual float GetWti(const ExecutionContext& ctx, const boost::any& problem) const = 0;

    // Returns the workspace size required by the solver for a given ConvolutionContext
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
};

/// Typedef for convolution solvers
using ConvSolver = NonTunableSolverBase<ConvolutionContext, ProblemDescription>;

/// Base class for tunable solvers
struct ConvTunableSolverBase : SolverMixin<ConvolutionContext, ProblemDescription>
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
    virtual boost::any GetDefaultPerformanceConfig(const ConvolutionContext& ctx,
                                                   const ProblemDescription& problem,
                                                   int) const = 0;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                          const ProblemDescription& problem,
                                          const PerfConfig& config) const = 0;

    /// Search
    ///
    /// The int parameter is needed only to not change the name of the
    /// function in the derived class. Function declarations that differ
    /// only by its return type cannot be overloaded.
    virtual boost::any Search(const ConvolutionContext& ctx,
                              const ProblemDescription& problem,
                              const AnyInvokeParams& invoke_ctx,
                              int) const = 0;

    /// Tunable solvers provide a GetSolution that takes a Context and PerformanceConfig
    virtual ConvSolution GetSolution(const ConvolutionContext& ctx,
                                     const ProblemDescription& problem,
                                     const PerfConfig& config) const = 0;
};

template <class PerformanceConfig>
struct ConvTunableSolver : ConvTunableSolverBase
{
    static_assert(std::is_base_of<PerfConfig, PerformanceConfig>{},
                  "PerformanceConfig must be derived of PerfConfig");

    virtual PerformanceConfig GetDefaultPerformanceConfig(const ConvolutionContext&,
                                                          const ProblemDescription&) const = 0;
    virtual bool IsValidPerformanceConfig(const ConvolutionContext&,
                                          const ProblemDescription&,
                                          const PerformanceConfig&) const                  = 0;
    virtual PerformanceConfig
    Search(const ConvolutionContext&, const ProblemDescription&, const AnyInvokeParams&) const = 0;
    virtual ConvSolution GetSolution(const ConvolutionContext&,
                                     const ProblemDescription&,
                                     const PerformanceConfig&) const                           = 0;

    boost::any GetDefaultPerformanceConfig(const ConvolutionContext& ctx,
                                           const ProblemDescription& problem,
                                           int) const final
    {
        return GetDefaultPerformanceConfig(ctx, problem);
    }

    bool IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                  const ProblemDescription& problem,
                                  const PerfConfig& config) const final
    {
        return IsValidPerformanceConfig(
            ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }

    boost::any Search(const ConvolutionContext& ctx,
                      const ProblemDescription& problem,
                      const AnyInvokeParams& invoke_ctx,
                      int) const final
    {
        return Search(ctx, problem, invoke_ctx);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem,
                             const PerfConfig& config) const final
    {
        return GetSolution(ctx, problem, dynamic_cast<const PerformanceConfig&>(config));
    }
};

struct PerformanceConfigConvAsm3x3U : PerfConfigBase<PerformanceConfigConvAsm3x3U>
{
    int limit_wave_cnt;        // [0..9]
    int filters_per_wave;      // [1..8]
    int output_lines_per_wave; // [1..8]

    PerformanceConfigConvAsm3x3U(int lwc, int fpw, int olpw);
    PerformanceConfigConvAsm3x3U() : PerformanceConfigConvAsm3x3U(-1, -1, -1) {}
    PerformanceConfigConvAsm3x3U(bool) : PerformanceConfigConvAsm3x3U(0, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.filters_per_wave, "filters_per_wave");
        f(self.output_lines_per_wave, "output_lines_per_wave");
    }

    void HeuristicInit(const ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    bool operator==(const PerformanceConfigConvAsm3x3U& other) const;
};

struct ConvAsm3x3U final : ConvTunableSolver<PerformanceConfigConvAsm3x3U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm3x3U>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceConfigConvAsm3x3U
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigConvAsm3x3U&) const override;
    PerformanceConfigConvAsm3x3U Search(const ConvolutionContext&,
                                        const ProblemDescription&,
                                        const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
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

    PerformanceConfigConvAsm1x1U(int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1U(bool spare);

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

    void StaticHeuristic(const ProblemDescription& problem);
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    void
    RunParmeterPredictionModel(const ConvolutionContext&, const ProblemDescription&, bool& valid);
    bool ModelApplyToken(int index, int value, const ProblemDescription&);
#endif
    bool IsValidValue() const { return IsValidValueImpl(8); }
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription& problem) const { return IsValidImpl(problem, 8); }
    bool operator==(const PerformanceConfigConvAsm1x1U& other) const;

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    bool IsPartiallyValid(const ProblemDescription& problem, int sequence_length) const
    {
        return IsValidImpl(problem, sequence_length);
    }
    bool IsPartiallyValidValue(int sequence_length) const
    {
        return IsValidValueImpl(sequence_length);
    }
#endif
    bool IsValidImpl(const ProblemDescription& problem, int sequence_length) const;
    bool IsValidValueImpl(int sequence_length) const;
};

struct ConvAsm1x1U final : ConvTunableSolver<PerformanceConfigConvAsm1x1U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm1x1U>(); }

    PerformanceConfigConvAsm1x1U
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigConvAsm1x1U&) const override;
    PerformanceConfigConvAsm1x1U Search(const ConvolutionContext&,
                                        const ProblemDescription&,
                                        const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
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

    PerformanceConfigConvAsm1x1UV2(int, int, int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1UV2()
        : PerformanceConfigConvAsm1x1UV2(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1UV2(bool spare);

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

    void HeuristicInit(const ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    bool operator==(const PerformanceConfigConvAsm1x1UV2& other) const;
};

struct ConvAsm1x1UV2 final : ConvTunableSolver<PerformanceConfigConvAsm1x1UV2>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm1x1UV2>(); }

    PerformanceConfigConvAsm1x1UV2
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigConvAsm1x1UV2&) const override;
    PerformanceConfigConvAsm1x1UV2 Search(const ConvolutionContext&,
                                          const ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConfigConvAsm1x1UV2&) const override;
};

struct ConvAsm5x10u2v2f1 final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm5x10u2v2f1>(); }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsm5x10u2v2b1 final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm5x10u2v2b1>(); }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsm7x7c3h224w224k64u2v2p3q3f1 final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsm7x7c3h224w224k64u2v2p3q3f1>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }
    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvOclDirectFwd11x11 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvOclDirectFwd11x11>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
};

struct ConvOclDirectFwdGen final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclDirectFwdGen>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
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

    PerformanceImplicitGemm(bool spare);

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

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool operator==(const PerformanceImplicitGemm& other) const;
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

    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
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

    PerformanceImplicitGemmV4R4Fwd(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4Fwd(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4Fwd(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4Fwd() : PerformanceImplicitGemmV4R4Fwd(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4Fwd(bool spare);

    bool operator==(const PerformanceImplicitGemmV4R4Fwd& other) const;

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

    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool> CalculateGemmABlockCopyPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
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

    PerformanceImplicitGemmV4R4WrW(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4WrW(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4WrW(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4WrW() : PerformanceImplicitGemmV4R4WrW(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4WrW(bool spare);

    bool operator==(const PerformanceImplicitGemmV4R4WrW& other) const;

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

    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool> CalculateGemmABlockCopyPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
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

    PerformanceImplicitGemmBwdDataV1R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV1R1()
        : PerformanceImplicitGemmBwdDataV1R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV1R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV1R1(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmBwdDataV1R1(bool spare);

    bool operator==(const PerformanceImplicitGemmBwdDataV1R1& other) const;

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

    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext&,
                                            const ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ConvolutionContext&,
                                                 const ProblemDescription&) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ConvolutionContext&,
                                                 const ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ConvolutionContext&,
                                                           const ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
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

    PerformanceImplicitGemmBwdDataV4R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV4R1()
        : PerformanceImplicitGemmBwdDataV4R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV4R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV4R1(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmBwdDataV4R1(bool spare);

    bool operator==(const PerformanceImplicitGemmBwdDataV4R1& other) const;

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

    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
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
    PerformanceImplicitGemmBwdDataV4R1Xdlops(int, int, int, int, int, int, bool, bool, bool);

    PerformanceImplicitGemmBwdDataV4R1Xdlops();
    PerformanceImplicitGemmBwdDataV4R1Xdlops(bool spare);
    PerformanceImplicitGemmBwdDataV4R1Xdlops(
        int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceImplicitGemmBwdDataV4R1Xdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    bool operator==(const PerformanceImplicitGemmBwdDataV4R1Xdlops& other) const;

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

    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsReallyValid(const ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext&, const ProblemDescription&) const;
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
};

struct ConvHipImplicitGemmV4R1Fwd final : ConvTunableSolver<PerformanceImplicitGemmV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R1Fwd>();
    }

    PerformanceImplicitGemmV4R1
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmV4R1&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmV4R1&) const override;
    PerformanceImplicitGemmV4R1 Search(const ConvolutionContext&,
                                       const ProblemDescription&,
                                       const AnyInvokeParams& invoke_ctx) const override;
};

struct ConvHipImplicitGemmV4R4Fwd final : ConvTunableSolver<PerformanceImplicitGemmV4R4Fwd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R4Fwd>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceImplicitGemmV4R4Fwd
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmV4R4Fwd&) const override;
    PerformanceImplicitGemmV4R4Fwd Search(const ConvolutionContext&,
                                          const ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmV4R4Fwd&) const override;

private:
    static std::tuple<int, int, int> CalculateGemmSize(const ProblemDescription&);

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

    /// \ref https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1154
    static PerformanceConvMlirIgemm& MlirHeuristicInitRequest()
    {
        static PerformanceConvMlirIgemm heur;
        heur.SetMlirHeuristicInitRequest();
        return heur;
    }

    PerformanceConvMlirIgemm(int, int, int, int, int, int, bool);

    PerformanceConvMlirIgemm(int a, int b, int c, int d, int e, int f)
        : PerformanceConvMlirIgemm(a, b, c, d, e, f, false)
    {
    }

    PerformanceConvMlirIgemm() : PerformanceConvMlirIgemm(-1, -1, -1, -1, -1, -1, false) {}

    PerformanceConvMlirIgemm(bool spare);

    bool operator==(const PerformanceConvMlirIgemm& other) const;

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

    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool SetNextValue(const ProblemDescription&);

private:
    void SetMlirHeuristicInitRequest();
};

struct ConvMlirIgemmFwd final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmFwd>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceConvMlirIgemm GetDefaultPerformanceConfig(const ConvolutionContext&,
                                                         const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConvMlirIgemm&) const override;
    PerformanceConvMlirIgemm Search(const ConvolutionContext&,
                                    const ProblemDescription&,
                                    const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
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

    /// \ref https://github.com/ROCmSoftwarePlatform/MIOpen/issues/1154
    static PerformanceConvMlirIgemmXdlops& MlirHeuristicInitRequest()
    {
        static PerformanceConvMlirIgemmXdlops heur;
        heur.SetMlirHeuristicInitRequest();
        return heur;
    }

    PerformanceConvMlirIgemmXdlops(int, int, int, int, int, int, bool, bool, bool);

    PerformanceConvMlirIgemmXdlops();
    PerformanceConvMlirIgemmXdlops(bool spare);
    PerformanceConvMlirIgemmXdlops(int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceConvMlirIgemmXdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    bool operator==(const PerformanceConvMlirIgemmXdlops& other) const;

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

    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool SetNextValue(const ProblemDescription&);

private:
    void SetMlirHeuristicInitRequest();
};

struct ConvMlirIgemmFwdXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmFwdXdlops>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceConvMlirIgemmXdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConvMlirIgemmXdlops&) const override;
    PerformanceConvMlirIgemmXdlops Search(const ConvolutionContext&,
                                          const ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
};

struct ConvHipImplicitGemmV4R4WrW final : ConvTunableSolver<PerformanceImplicitGemmV4R4WrW>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R4WrW>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceImplicitGemmV4R4WrW
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmV4R4WrW&) const override;
    PerformanceImplicitGemmV4R4WrW Search(const ConvolutionContext&,
                                          const ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmV4R4WrW&) const override;

private:
    static std::tuple<int, int, int> CalculateGemmSize(const ProblemDescription&);

    friend struct PerformanceImplicitGemmV4R4WrW;
};

struct ConvMlirIgemmWrW final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmWrW>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceConvMlirIgemm GetDefaultPerformanceConfig(const ConvolutionContext&,
                                                         const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConvMlirIgemm&) const override;
    PerformanceConvMlirIgemm Search(const ConvolutionContext&,
                                    const ProblemDescription&,
                                    const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
};

struct ConvMlirIgemmWrWXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmWrWXdlops>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceConvMlirIgemmXdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConvMlirIgemmXdlops&) const override;
    PerformanceConvMlirIgemmXdlops Search(const ConvolutionContext&,
                                          const ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
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

    PerformanceImplicitGemmForwardV4R4Xdlops(int, int, int, int, int, int, bool, bool, int);
    PerformanceImplicitGemmForwardV4R4Xdlops();
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

    bool operator==(const PerformanceImplicitGemmForwardV4R4Xdlops& other) const;

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsReallyValid(const ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext&, const ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
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

    PerformanceImplicitGemmForwardV4R5Xdlops(int, int, int, int, int, int, bool, bool, int, bool);
    PerformanceImplicitGemmForwardV4R5Xdlops();
    PerformanceImplicitGemmForwardV4R5Xdlops(bool spare);

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

    bool operator==(const PerformanceImplicitGemmForwardV4R5Xdlops& other) const;

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsReallyValid(const ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext&, const ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
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

    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool, int);
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm();
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

    bool operator==(const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm& other) const;

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsReallyValid(const ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext&, const ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
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

    PerformanceImplicitGemmBwdV1R1Xdlops(int, int, int, int, int, int, bool, bool);
    PerformanceImplicitGemmBwdV1R1Xdlops();
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

    bool operator==(const PerformanceImplicitGemmBwdV1R1Xdlops& other) const;

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsReallyValid(const ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext&, const ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
};

struct ConvHipImplicitGemmForwardV4R4Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R4Xdlops>();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;
    PerformanceImplicitGemmForwardV4R4Xdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static std::tuple<int, int, int, int> CalculateGemmSize(const ProblemDescription&);

    friend struct PerformanceImplicitGemmForwardV4R4Xdlops;
};

struct ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm>();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ConvolutionContext&,
        const ProblemDescription&,
        const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution
    GetSolution(const ConvolutionContext&,
                const ProblemDescription&,
                const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm&) const override;
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static std::tuple<int, int, int, int, int, int, int>
    CalculateGemmSize(const ProblemDescription&, int GemmMFactor, int GemmNFactor, int GemmKFactor);

    friend struct PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm;
};

struct ConvHipImplicitGemmForwardV4R5Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R5Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R5Xdlops>();
    }

    PerformanceImplicitGemmForwardV4R5Xdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmForwardV4R5Xdlops&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R5Xdlops&) const override;
    PerformanceImplicitGemmForwardV4R5Xdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct ConvHipImplicitGemmV4R1WrW final : ConvTunableSolver<PerformanceImplicitGemmV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R1WrW>();
    }

    PerformanceImplicitGemmV4R1
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmV4R1&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmV4R1&) const override;
    PerformanceImplicitGemmV4R1 Search(const ConvolutionContext&,
                                       const ProblemDescription&,
                                       const AnyInvokeParams& invoke_ctx) const override;
};

struct ConvHipImplicitGemmBwdDataV1R1 final : ConvTunableSolver<PerformanceImplicitGemmBwdDataV1R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV1R1>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceImplicitGemmBwdDataV1R1
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmBwdDataV1R1&) const override;
    PerformanceImplicitGemmBwdDataV1R1 Search(const ConvolutionContext&,
                                              const ProblemDescription&,
                                              const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV1R1&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    static std::tuple<int, int, int> CalculateGemmSize(const ConvolutionContext&,
                                                       const ProblemDescription&);

    friend struct PerformanceImplicitGemmBwdDataV1R1;
};

struct ConvMlirIgemmBwd final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmBwd>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceConvMlirIgemm GetDefaultPerformanceConfig(const ConvolutionContext&,
                                                         const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConvMlirIgemm&) const override;
    PerformanceConvMlirIgemm Search(const ConvolutionContext&,
                                    const ProblemDescription&,
                                    const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
};

struct ConvMlirIgemmBwdXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmBwdXdlops>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceConvMlirIgemmXdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConvMlirIgemmXdlops&) const override;
    PerformanceConvMlirIgemmXdlops Search(const ConvolutionContext&,
                                          const ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
};

struct ConvHipImplicitGemmBwdDataV4R1 final : ConvTunableSolver<PerformanceImplicitGemmBwdDataV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV4R1>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    PerformanceImplicitGemmBwdDataV4R1
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmBwdDataV4R1&) const override;
    PerformanceImplicitGemmBwdDataV4R1 Search(const ConvolutionContext&,
                                              const ProblemDescription&,
                                              const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV4R1&) const override;

private:
    static int CalculateNumberOfGemm(const ProblemDescription&);
    static std::tuple<int, int, int> CalculateGemmSize(const ProblemDescription&, int gemm_id);

    friend struct PerformanceImplicitGemmBwdDataV4R1;
};

struct ConvHipImplicitGemmBwdDataV4R1Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmBwdDataV4R1Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV4R1Xdlops>();
    }

    PerformanceImplicitGemmBwdDataV4R1Xdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmBwdDataV4R1Xdlops&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV4R1Xdlops&) const override;
    PerformanceImplicitGemmBwdDataV4R1Xdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static int CalculateNumberOfGemm(const ProblemDescription&);
    static std::tuple<int, int, int, int> CalculateGemmSize(const ProblemDescription&, int gemm_id);

    friend struct PerformanceImplicitGemmBwdDataV4R1Xdlops;
};

struct ConvHipImplicitGemmBwdDataV1R1Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmBwdV1R1Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV1R1Xdlops>();
    }

    PerformanceImplicitGemmBwdV1R1Xdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmBwdV1R1Xdlops&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    PerformanceImplicitGemmBwdV1R1Xdlops Search(const ConvolutionContext&,
                                                const ProblemDescription&,
                                                const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmBwdV1R1Xdlops&) const override;

private:
    static std::tuple<int, int, int, int> CalculateGemmSize(const ProblemDescription&);

    friend struct PerformanceImplicitGemmBwdV1R1Xdlops;
};

struct ConvAsmImplicitGemmV4R1DynamicFwd final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicFwd>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmV4R1DynamicFwd_1x1 final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicFwd_1x1>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmV4R1DynamicWrw final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::GetWorkspaceSize;
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicWrw>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    size_t GetWorkspaceSize(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicWrwXdlops final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::GetWorkspaceSize;
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicWrwXdlops>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    size_t GetWorkspaceSize(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmV4R1DynamicBwd final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicBwd>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdXdlops final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdXdlops>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicBwdXdlops final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicBwdXdlops>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
struct ConvOclDirectFwdLegacyExhaustiveSearch : ConvTunableSolver<LegacyPerformanceConfig>
{
    LegacyPerformanceConfig GetDefaultPerformanceConfig(const ConvolutionContext&,
                                                        const ProblemDescription&) const override;
    LegacyPerformanceConfig Search(const ConvolutionContext&,
                                   const ProblemDescription&,
                                   const AnyInvokeParams& invoke_ctx) const override;

private:
    template <typename Tgpu>
    LegacyPerformanceConfig SearchImpl(const ConvolutionContext&,
                                       const ProblemDescription&,
                                       const AnyInvokeParams& invoke_ctx) const;
};

struct ConvOclDirectFwd : ConvOclDirectFwdLegacyExhaustiveSearch
{
    static ConvSolution BaseGetSolution(const ConvolutionContext& ctx,
                                        const ProblemDescription& problem,
                                        const LegacyPerformanceConfig& config);
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclDirectFwd>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const LegacyPerformanceConfig&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const LegacyPerformanceConfig&) const override;
};

struct ConvOclDirectFwd1x1 final : ConvOclDirectFwdLegacyExhaustiveSearch
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclDirectFwd1x1>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const LegacyPerformanceConfig&) const override;

    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const LegacyPerformanceConfig&) const override
    {
        return true;
    }
};

struct ConvBinWinograd3x3U final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBinWinograd3x3U>(); }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct ConvBinWinogradRxS final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBinWinogradRxS>(); }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
};

struct PerformanceConfigConvBinWinogradRxS : PerfConfigBase<PerformanceConfigConvBinWinogradRxS>
{
    int n_groups;
    PerformanceConfigConvBinWinogradRxS(int n_groups_);
    PerformanceConfigConvBinWinogradRxS() : PerformanceConfigConvBinWinogradRxS(-1) {}
    PerformanceConfigConvBinWinogradRxS(bool) : PerformanceConfigConvBinWinogradRxS(1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_groups, "n_groups");
    }
    int GetNGroups() const { return n_groups; }

    template <int Winodata, int Winofilter>
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext& ctx, const ProblemDescription&) const
    {
        return IsValid(ctx);
    }
    bool IsValid(const ConvolutionContext&) const;
    bool operator==(const PerformanceConfigConvBinWinogradRxS& other) const;
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

    PerformanceConfigConvBinWinogradRxS
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigConvBinWinogradRxS&) const override;
    PerformanceConfigConvBinWinogradRxS Search(const ConvolutionContext&,
                                               const ProblemDescription&,
                                               const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
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

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvMPBidirectWinograd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;

    // kernel_file_name for solver identification
    static std::string GetSolverFileNames(int id)
    {
        static const std::string names[3] = {"xform_bidirect_winograd_data.s",
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

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;

    bool IsDynamic() const override
    {
        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsDynamic() &&
               ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}
                   .IsDynamic() &&
               IsThisSolverDynamic();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops
    GetDefaultPerformanceConfig(const ConvolutionContext& ctx,
                                const ProblemDescription& problem) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvHipImplicitGemmForwardV4R4Xdlops{}.GetDefaultPerformanceConfig(xdlops_ctx,
                                                                                  xdlops_problem);
    }

    bool
    IsValidPerformanceConfig(const ConvolutionContext& ctx,
                             const ProblemDescription& problem,
                             const PerformanceImplicitGemmForwardV4R4Xdlops& config) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsValidPerformanceConfig(
            xdlops_ctx, xdlops_problem, config);
    }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>()
                   .GetWorkspaceSize(ctx, problem) +
               ConvHipImplicitGemmForwardV4R4Xdlops{}.GetWorkspaceSize(xdlops_ctx, xdlops_problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    PerformanceImplicitGemmForwardV4R4Xdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;

private:
    ConvolutionContext
    GetTransformedConvContext(const ConvolutionContext& ctx,
                              const ProblemDescription& transformed_problem) const;
    ProblemDescription GetTransformedProblem(const ProblemDescription& problem) const;

    // kernel_file_name for solver identification
    static std::string GetSolverFileNames(int id)
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
    // To suppress -Woverloaded-virtual
    using ConvSolver::GetWorkspaceSize;
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool IsDynamic() const override { return true; }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

    // kernel_file_name for solver identification
    static std::string GetSolverFileNames(int id)
    {
        static const std::string names[3] = {"xform_data.s", "xform_filter.s", "xform_out.s"};
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

    static int GetSolverWinoXformHWSize(const ProblemDescription& problem, int id)
    {
        if(id == 0)
            return WinoDataH +
                   (WinoFilterH - 1) * (WinoDataH == 7 ? 2 : problem.GetKernelStrideH());
        else
            return WinoDataW +
                   (WinoFilterW - 1) * (WinoDataW == 7 ? 2 : problem.GetKernelStrideW());
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    size_t GetWorkspaceSize(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;

    InvokerFactory PrepareInvokerFactory(const ExecutionContext&,
                                         const ProblemDescription&,
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

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool operator==(const PerformanceConfigAsmDirect3x3WrW& other) const;
};

struct ConvAsmBwdWrW3x3 final : ConvTunableSolver<PerformanceConfigAsmDirect3x3WrW>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsmBwdWrW3x3>(); }

    PerformanceConfigAsmDirect3x3WrW
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigAsmDirect3x3WrW&) const override;
    PerformanceConfigAsmDirect3x3WrW Search(const ConvolutionContext&,
                                            const ProblemDescription&,
                                            const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConfigAsmDirect3x3WrW& config) const override;
};

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoFuryRxS final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvWinoFuryRxS<Winodata, Winofilter>>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override;

    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;

    static constexpr bool is2x3() { return Winodata == 2 && Winofilter == 3; }
    static constexpr bool is3x2() { return Winodata == 3 && Winofilter == 2; }
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

    PerformanceConfigConvAsmBwdWrW1x1(int chunk_size_,
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

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool operator==(const PerformanceConfigConvAsmBwdWrW1x1& other) const;
};

struct ConvAsmBwdWrW1x1 final : ConvTunableSolver<PerformanceConfigConvAsmBwdWrW1x1>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsmBwdWrW1x1>(); }

    PerformanceConfigConvAsmBwdWrW1x1
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigConvAsmBwdWrW1x1&) const override;
    PerformanceConfigConvAsmBwdWrW1x1 Search(const ConvolutionContext&,
                                             const ProblemDescription&,
                                             const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
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

    void HeuristicInit(const ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool operator==(const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& other) const;
};

template <int N_BATCH_LOOPS>
struct ConvOclBwdWrW2 : ConvTunableSolver<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>
{
    const std::string& SolverDbId() const override
    {
        return this->template GetSolverDbId<ConvOclBwdWrW2<N_BATCH_LOOPS>>();
    }

    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool
    IsValidPerformanceConfig(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&) const override;
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&) const override;

protected:
    bool IsApplicableBase(const ConvolutionContext&, const ProblemDescription&) const;
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

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const;

private:
    // This function dervied from ConvOclBwdWrW2 is declared private
    // so that this solver is not marked searchable/tunable.
    using ConvOclBwdWrW2<1>::GetDefaultPerformanceConfig;
    using ConvOclBwdWrW2<1>::GetSolution;
};

struct ConvOclBwdWrW53 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclBwdWrW53>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
};

struct ConvOclBwdWrW1x1 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclBwdWrW1x1>(); }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
};

struct fft final : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::GetWorkspaceSize;
    using ConvSolver::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<fft>(); }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const ProblemDescription&) const;
    size_t GetWorkspaceSize(const ExecutionContext&, const ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const ProblemDescription&) const;
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

    PerformanceImplicitGemmWrwV4R4Xdlops(int, int, int, int, int, int, bool, bool, bool);
    PerformanceImplicitGemmWrwV4R4Xdlops();
    PerformanceImplicitGemmWrwV4R4Xdlops(bool spare);
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

    bool operator==(const PerformanceImplicitGemmWrwV4R4Xdlops& other) const;

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsReallyValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext&, const ProblemDescription&) const;

    std::tuple<int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ConvolutionContext&, const ProblemDescription&) const;
    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext&,
                                            const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
};

struct ConvHipImplicitGemmWrwV4R4Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmWrwV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmWrwV4R4Xdlops>();
    }

    PerformanceImplicitGemmWrwV4R4Xdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceImplicitGemmWrwV4R4Xdlops&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceImplicitGemmWrwV4R4Xdlops&) const override;
    PerformanceImplicitGemmWrwV4R4Xdlops Search(const ConvolutionContext&,
                                                const ProblemDescription&,
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

    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool);
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm();
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

    bool operator==(const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm& other) const;

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsReallyValid(const ConvolutionContext&, const ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ConvolutionContext&, const ProblemDescription&) const;

    std::tuple<int, int, int, int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ConvolutionContext&, const ProblemDescription&) const;
    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ConvolutionContext&,
                                            const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ProblemDescription&) const;
    std::tuple<std::size_t, bool> CalculateLdsNumberOfByte(const ProblemDescription&) const;
};

struct ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm final
    : ConvTunableSolver<PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm>();
    }

    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsValidPerformanceConfig(
        const ConvolutionContext&,
        const ProblemDescription&,
        const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm&) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    ConvSolution
    GetSolution(const ConvolutionContext&,
                const ProblemDescription&,
                const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm&) const override;
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    Search(const ConvolutionContext&,
           const ProblemDescription&,
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

    bool SetNextValue(const ProblemDescription&);
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
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

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsDynamic() const override { return false; }
    PerformanceConvCkIgemmFwdV6r1DlopsNchw
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const override;
    PerformanceConvCkIgemmFwdV6r1DlopsNchw Search(const ConvolutionContext&,
                                                  const ProblemDescription&,
                                                  const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const override;
};

struct ConvDirectNaiveConvFwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvFwd>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled due to MIOpenGemm or OCL compiler issues.
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override
    {
        return 0.01f;
    }
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
};

struct ConvDirectNaiveConvBwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvBwd>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled due to MIOpenGemm or OCL compiler issues.
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override
    {
        return 0.01f;
    }
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
};

struct ConvDirectNaiveConvWrw final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvWrw>();
    }

    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled due to MIOpenGemm or OCL compiler issues.
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override
    {
        return 0.01f;
    }
    ConvSolution GetSolution(const ConvolutionContext&, const ProblemDescription&) const override;
};

struct GemmFwdBase : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::GetWti;
    using ConvSolver::IsApplicable;

    bool IsDynamic() const override { return true; }
    float GetWti(const ConvolutionContext& ctx, const ProblemDescription& problem) const override
    {
        return GetWti(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    float GetWti(const ExecutionContext& context, const conv::ProblemDescription& problem) const;

    friend struct GemmFwd1x1_0_2;
    friend struct GemmFwd1x1_0_1_int8;
    friend struct GemmFwd1x1_0_1;
    friend struct GemmFwdRest;
};

struct GemmFwd1x1_0_2 final : GemmFwdBase
{
    // To suppress -Woverloaded-virtual
    using GemmFwdBase::GetWorkspaceSize;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_2>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;

    friend struct GemmFwdRest;
};

struct GemmFwd1x1_0_1_int8 final : GemmFwdBase
{
    // To suppress -Woverloaded-virtual
    using GemmFwdBase::GetWorkspaceSize;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_1_int8>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;

    friend struct GemmFwdRest;
};

struct GemmFwd1x1_0_1 final : GemmFwdBase
{
    // To suppress -Woverloaded-virtual
    using GemmFwdBase::GetWorkspaceSize;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_1>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;

    friend struct GemmFwdRest;
};

struct GemmFwdRest final : GemmFwdBase
{
    // To suppress -Woverloaded-virtual
    using GemmFwdBase::GetWorkspaceSize;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwdRest>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmBwdBase : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::GetWti;
    using ConvSolver::IsApplicable;

    bool IsDynamic() const override { return true; }
    float GetWti(const ConvolutionContext& ctx, const ProblemDescription& problem) const override
    {
        return GetWti(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    float GetWti(const ExecutionContext& context, const conv::ProblemDescription& problem) const;

    friend struct GemmBwd1x1_stride2;
    friend struct GemmBwd1x1_stride1;
    friend struct GemmBwdRest;
};

struct GemmBwd1x1_stride2 final : GemmBwdBase
{
    // To suppress -Woverloaded-virtual
    using GemmBwdBase::GetWorkspaceSize;
    using GemmBwdBase::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwd1x1_stride2>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;

    friend struct GemmBwdRest;
};

struct GemmBwd1x1_stride1 final : GemmBwdBase
{
    // To suppress -Woverloaded-virtual
    using GemmBwdBase::GetWorkspaceSize;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwd1x1_stride1>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicableBeforeWorkaround(const ExecutionContext&,
                                      const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;

    friend struct GemmBwdRest;
};

struct GemmBwdRest final : GemmBwdBase
{
    // To suppress -Woverloaded-virtual
    using GemmBwdBase::GetWorkspaceSize;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwdRest>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
};

struct GemmWrwBase : ConvSolver
{
    // To suppress -Woverloaded-virtual
    using ConvSolver::GetWti;
    using ConvSolver::IsApplicable;

    bool IsDynamic() const override { return true; }
    float GetWti(const ConvolutionContext& ctx, const ProblemDescription& problem) const override
    {
        return GetWti(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    float GetWti(const ExecutionContext& context, const conv::ProblemDescription& problem) const;

    friend struct GemmWrw1x1_stride1;
    friend struct GemmWrwUniversal;
};

struct GemmWrw1x1_stride1 final : GemmWrwBase
{
    // To suppress -Woverloaded-virtual
    using GemmWrwBase::IsApplicable;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmWrw1x1_stride1>(); }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;

    friend struct GemmWrwUniversal;
};

struct GemmWrwUniversal final : GemmWrwBase
{
    // To suppress -Woverloaded-virtual
    using GemmWrwBase::GetWorkspaceSize;

    const std::string& SolverDbId() const override { return GetSolverDbId<GemmWrwUniversal>(); }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx,
                            const ProblemDescription& problem) const override
    {
        return GetWorkspaceSize(static_cast<const ExecutionContext&>(ctx), problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ConvolutionContext& ctx,
                      const ProblemDescription& problem) const override
    {
        return IsApplicable(static_cast<const ExecutionContext&>(ctx), problem);
    }

    ConvSolution GetSolution(const ConvolutionContext& ctx,
                             const ProblemDescription& problem) const override
    {
        return GetSolution(static_cast<const ExecutionContext&>(ctx), problem);
    }

private:
    size_t GetWorkspaceSize(const ExecutionContext&, const conv::ProblemDescription&) const;
    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const;
    ConvSolution GetSolution(const ExecutionContext&, const conv::ProblemDescription&) const;
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

    PerformanceConfigAsmImplicitGemmGTC(std::string dir,
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
    PerformanceConfigAsmImplicitGemmGTC(std::string dir,
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
    void HeuristicInit(const ConvolutionContext&)                            = delete;
    bool SetNextValue(const ProblemDescription&)                             = delete;
    bool IsValidValue() const                                                = delete;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const = delete;

    bool IsDefaultConstructed() const;
    bool operator==(const PerformanceConfigAsmImplicitGemmGTC& other) const;
    void CopyParameters(const PerformanceConfigAsmImplicitGemmGTC& other);
    std::string ToString() const override;
    std::string ToKernelName(const ConvolutionContext& ctx) const;
    int BlockSize() const;
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

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription& config);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC>();
    }

    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ConvolutionContext&,
        const ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ConvolutionContext&,
                const ProblemDescription&,
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
    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC>();
    }

    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ConvolutionContext&,
        const ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ConvolutionContext&,
                const ProblemDescription&,
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

    void HeuristicInit(const ConvolutionContext&, const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    size_t ComputeKernelOccupancy() const;

private:
    void SetParamsForKSplit(const ProblemDescription& problem, const size_t& occupancy);
};

struct ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC>();
    }

    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ConvolutionContext&,
        const ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ConvolutionContext&, const ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ConvolutionContext&,
                const ProblemDescription&,
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
    void HeuristicInit(const ConvolutionContext&)                            = delete;
    bool SetNextValue(const ProblemDescription&)                             = delete;
    bool IsValidValue() const                                                = delete;
    bool IsValid(const ConvolutionContext&, const ProblemDescription&) const = delete;

    bool IsDefaultConstructed() const;
    bool operator==(const PerformanceConfigAsmImplicitGemmGTCvector& other) const;
    void CopyParameters(const PerformanceConfigAsmImplicitGemmGTCvector& other);
    std::string ToString() const override;
    std::string ToKernelName(const ConvolutionContext& ctx) const;
    int BlockSize() const;
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

    void HeuristicInit(const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC>();
    }
    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ConvolutionContext&,
        const ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ConvolutionContext&,
                const ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC&) const override;
};

struct PerformanceConfigHipImplicitGemmFwdXdlops
    : PerfConfigBase<PerformanceConfigHipImplicitGemmFwdXdlops>
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
    void HeuristicInit(const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    bool operator==(const PerformanceConfigHipImplicitGemmFwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const ProblemDescription&) const;
};

struct ConvHipImplicitGemmFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmFwdXdlops>();
    }

    PerformanceConfigHipImplicitGemmFwdXdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigHipImplicitGemmFwdXdlops&) const override;
    PerformanceConfigHipImplicitGemmFwdXdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
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
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType>
    bool CheckCKApplicability(const ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmBwdXdlops
    : PerfConfigBase<PerformanceConfigHipImplicitGemmBwdXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmBwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmBwdXdlops() : PerformanceConfigHipImplicitGemmBwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmBwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmBwdXdlops(0, "")
    {
    }
    void HeuristicInit(const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    bool operator==(const PerformanceConfigHipImplicitGemmBwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const ProblemDescription&) const;
};

struct ConvHipImplicitGemmBwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmBwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdXdlops>();
    }

    PerformanceConfigHipImplicitGemmBwdXdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const ProblemDescription&,
                                  const PerformanceConfigHipImplicitGemmBwdXdlops&) const override;
    PerformanceConfigHipImplicitGemmBwdXdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmBwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType>
    bool CheckCKApplicability(const ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemmGroupFwdXdlops
    : PerfConfigBase<PerformanceConfigHipImplicitGemmGroupFwdXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmGroupFwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmGroupFwdXdlops()
        : PerformanceConfigHipImplicitGemmGroupFwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmGroupFwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmGroupFwdXdlops(0, "")
    {
    }
    void HeuristicInit(const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    bool operator==(const PerformanceConfigHipImplicitGemmGroupFwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const ProblemDescription&) const;
};

struct ConvHipImplicitGemmGroupFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmGroupFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmGroupFwdXdlops>();
    }

    PerformanceConfigHipImplicitGemmGroupFwdXdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool
    IsValidPerformanceConfig(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupFwdXdlops&) const override;
    PerformanceConfigHipImplicitGemmGroupFwdXdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ConvolutionContext&,
                             const ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupFwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType>
    bool CheckCKApplicability(const ProblemDescription&) const;
};

struct PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    : PerfConfigBase<PerformanceConfigHipImplicitGemm3DGroupFwdXdlops>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops()
        : PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(bool)
        : PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(0, "")
    {
    }
    void HeuristicInit(const ProblemDescription&);
    bool SetNextValue(const ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ConvolutionContext&, const ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const ProblemDescription&) const;
    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    bool operator==(const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& other) const;

private:
    template <typename DataType>
    void Init(const ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const ProblemDescription&) const;
};

struct ConvHipImplicitGemm3DGroupFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemm3DGroupFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemm3DGroupFwdXdlops>();
    }

    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    GetDefaultPerformanceConfig(const ConvolutionContext&,
                                const ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ConvolutionContext&,
        const ProblemDescription&,
        const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops&) const override;
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    Search(const ConvolutionContext&,
           const ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ConvolutionContext&, const ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ConvolutionContext&,
                const ProblemDescription&,
                const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ConvolutionContext&, const ProblemDescription&) const override
    {
        return 0.02f;
    };

private:
    template <typename DataType>
    bool CheckCKApplicability(const ProblemDescription&) const;
};

// Use struct as a syntactic sugar to make the intent as clear as possible.
struct ThisSolverIsDeprecatedStatic
{
    static bool IsDisabled(const ConvolutionContext& ctx);
};

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_SOLVER_HPP_
