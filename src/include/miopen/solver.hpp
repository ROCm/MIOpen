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

#ifndef GUARD_MIOPEN_SOLVER_HPP_
#define GUARD_MIOPEN_SOLVER_HPP_

#include <miopen/config.h>

#include <memory>
#include <string>
#include <vector>
#include <ostream>

#include <miopen/logger.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/env.hpp>
#include <miopen/type_name.hpp>
#include <miopen/miopen.h>
#include <miopen/stringutils.hpp> // for IsPureOpenCLSolution()

namespace miopen {

/// Enables "Find first convolution only" mode. This mode is intended for
/// triaging problems which may reveal after PR #781. Globally disables
/// lookup for _all_ suitable convolutions (introduced in PR #781),
/// so lookup is stopped as soon as the first applicable Solution found.
/// Also enables performance filtering heuristics.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_FIND_FIRST_CONV)

/// Allows to explicitly disable performance filtering heuristics
/// in "Find first convolution only" mode.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)

/// \todo Remove MIOPEN_DEBUG_FIND_FIRST_CONV together with
/// MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING and all related code
/// as soon as "find all" mode is stable so backward compatibility mode
/// is not needed (for triaging etc) anymore.

/// It seems that legacy OpenCL solvers imply that only the 1st one
/// applicable solution shall be used. The rest of OpenCL
/// solutions, in spite of that their IsApplicabe() return TRUE,
/// may fail (1) during exhaustive search, (2) during compilation,
/// (3) on execution (like LDS overallocation) or (4) may reveal precision
/// problems. These problems impedes finding and using the really fastest OpenCL solution.

/// \todo Remove env.var (workaround is OFF by default):
MIOPEN_DECLARE_ENV_VAR(MIOPEN_OPENCL_WORKAROUND_FIND_ALL_CONV_DIRECT_FWD)
/// \todo Remove env.var (workaround is OFF by default):
MIOPEN_DECLARE_ENV_VAR(MIOPEN_OPENCL_WORKAROUND_FIND_ALL_CONV_DIRECT_BWD)
/// \todo Remove env.var (workaround is OFF by default):
MIOPEN_DECLARE_ENV_VAR(MIOPEN_OPENCL_WORKAROUND_FIND_ALL_CONV_DIRECT_WRW)

namespace solver {
/// \todo Move wave_size into abstraction wich represent GPU information
const int wave_size = 64;
/// Describes a kernel source and whatever information required in order
/// to build and run it (the former is unused for binary kernels).
struct KernelInfo
{
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::string kernel_file;
    std::string kernel_name;
    friend std::ostream& operator<<(std::ostream& os, const KernelInfo& k);
};

/// Information required to build and run a kernel (or a set of kernels),
/// which is expected to perform computatons as per the problem config.
///
/// TODO: Currently best suits a subset of existing solvers,
/// namely some OpenCL-written forward direct convolutions.
/// Shall be refactored (possibly, to a class hierarchy).
struct ConvSolution
{
    /// \todo Use better name than construction_params.
    std::vector<KernelInfo> construction_params; // impl may consist of multiple kernels.
    miopenStatus_t status;
    std::string solver_id;

    size_t workspce_sz;
    int grp_tile1;       // total number ALUs per group
    int grp_tile0;       // total number ALUs per group
    int in_tile1;        // size of in-tile in local memory
    int in_tile0;        // size of in-tile in local memory
    int out_pix_tile1;   // # of generated pixels per output per wk-item  (ALU)
    int out_pix_tile0;   // # of generated pixels per output per wk-item  (ALU)
    int n_out_pix_tiles; // # output pixel tiles per wk-item (ALU)
    int n_in_data_tiles; // # of blocks of different inputs in LDS
    int n_stacks;        // # of diff stacks (part of batch).

    ConvSolution(miopenStatus_t status_ = miopenStatusSuccess)
        : status(status_),
          solver_id("<unknown>"),
          workspce_sz(0),
          grp_tile1(-1),
          grp_tile0(-1),
          in_tile1(-1),
          in_tile0(-1),
          out_pix_tile1(-1),
          out_pix_tile0(-1),
          n_out_pix_tiles(-1),
          n_in_data_tiles(-1),
          n_stacks(-1)
    {
    }

    inline bool Succeeded() const { return status == miopenStatusSuccess; }
};

std::ostream& operator<<(std::ostream& os, const ConvSolution& s);

template <class Solver>
std::string ComputeSolverDbId(Solver)
{
    const auto& name = get_type_name<Solver>();
    auto idx         = name.find_last_of(':');
    return name.substr(idx + 1);
}

// This will retrieve the id of the solver to write to the database. By
// default it uses the class name. If the class is renamed, this function can
// overriden to keep the name to avoid DB corruption.
template <class Solver>
const std::string& SolverDbId(Solver solver)
{
    static const auto result = ComputeSolverDbId(solver);
    return result;
}

template <class Solver, class Context, class Db>
auto FindSolutionImpl(rank<1>, Solver s, const Context& context, Db& db)
    -> decltype(s.GetSolution(context, s.Search(context)))
{
    const FindEnforce enforce;
    MIOPEN_LOG_I(SolverDbId(s));
    if(enforce.IsDbClean(context))
    {
        if(db.Remove(context, SolverDbId(s)))
            MIOPEN_LOG_W("Perf Db: record removed: " << SolverDbId(s) << ", enforce: " << enforce);
    }
    else
    {
        if((context.do_search || enforce.IsSearch(context)) && enforce.IsDbUpdate(context))
        {
            MIOPEN_LOG_W("Perf Db: load skipped: " << SolverDbId(s) << ", enforce: " << enforce);
        }
        else
        {
            using PerformanceConfig = decltype(s.GetPerformanceConfig(context));
            PerformanceConfig config{};
            if(db.Load(context, SolverDbId(s), config))
            {
                MIOPEN_LOG_I2("Perf Db: record loaded: " << SolverDbId(s));
                if(s.IsValidPerformanceConfig(context, config))
                {
                    return s.GetSolution(context, config);
                }
                MIOPEN_LOG(
                    (MIOPEN_INSTALLABLE ? LoggingLevel::Warning : miopen::LoggingLevel::Error),
                    "Invalid config loaded from Perf Db: " << SolverDbId(s) << ": " << config
                                                           << ". Performance may degrade.");
            }
            else
            {
                MIOPEN_LOG_I("Perf Db: record not found for: " << SolverDbId(s));
            }
        }

        if(context.do_search || enforce.IsSearch(context)) // TODO: Make it a customization point
        {
            MIOPEN_LOG_I("Starting search: " << SolverDbId(s) << ", enforce: " << enforce);
            try
            {
                auto c = s.Search(context);
                db.Update(context, SolverDbId(s), c);
                return s.GetSolution(context, c);
            }
            catch(const miopen::Exception& ex)
            {
                MIOPEN_LOG_E("Search failed for: " << SolverDbId(s) << ": " << ex.what());
            }
        }
    }

    return s.GetSolution(context, s.GetPerformanceConfig(context));
}

template <class Solver, class Context, class Db>
auto FindSolutionImpl(rank<0>, Solver s, const Context& context, Db&)
    -> decltype(s.GetSolution(context))
{
    MIOPEN_LOG_I(SolverDbId(s) << " (not searchable)");
    return s.GetSolution(context);
}

/// Finds optimized Solution. Generic method.
///
/// Given the specific problem config, finds (hopefully) optimal
/// solution-specific parameters and returns the Solution object.
/// Could take long if an exhaustive search is requested/performed.
/// May read/write perfDb.
template <class Solver, class Context, class Db>
ConvSolution FindSolution(Solver s, const Context& context, Db& db)
{
    static_assert(std::is_empty<Solver>{} && std::is_trivially_constructible<Solver>{},
                  "Solver must be stateless");
    // TODO: This assumes all solutions are ConvSolution
    auto solution      = FindSolutionImpl(rank<1>{}, s, context, db);
    solution.solver_id = SolverDbId(s);
    return solution;
}

// Search for the 1st applicable solution among many solvers
template <class... Solvers, class Context, class Db>
auto SearchForSolution(const Context& search_params, Db db) ->
    typename std::common_type<decltype(FindSolution(Solvers{}, search_params, db))...>::type
{
    using Solution =
        typename std::common_type<decltype(FindSolution(Solvers{}, search_params, db))...>::type;
    Solution solution{miopenStatusUnknownError};

// Using const here causes gcc to ICE
#if(!defined(__GNUC__) || defined(__clang__))
    const
#endif
        auto no_perf_filtering = miopen::IsDisabled(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING{});

    miopen::each_args(
        [&](auto solver) {
            if(solver.IsApplicable(search_params) &&
               (no_perf_filtering || solver.IsFast(search_params)))
            {
                if(!solution.Succeeded())
                {
                    solution = FindSolution(solver, search_params, db);
                    if(solution.Succeeded())
                    {
                        MIOPEN_LOG_I2(SolverDbId(solver) << ": Success.");
                        if(solution.construction_params.empty())
                        {
                            MIOPEN_THROW(std::string("Internal error in solver: ") +
                                         SolverDbId(solver));
                        }
                    }
                }
                else
                    MIOPEN_LOG_I2(SolverDbId(solver) << ": Skipped");
            }
            else
                MIOPEN_LOG_I2(SolverDbId(solver) << ": Not applicable");
        },
        Solvers{}...);

    return solution;
}

template <class Solution>
static inline bool IsPureOpenCLSolution(const Solution& s)
{
    return std::all_of(s.construction_params.begin(),
                       s.construction_params.end(),
                       [](const auto& k) { return miopen::EndsWith(k.kernel_file, ".cl"); });
}

// Search for all applicable solutions among many solvers
template <class... Solvers, class Context, class Db, class Solution = miopen::solver::ConvSolution>
std::vector<Solution> SearchForAllSolutions(const Context& search_params, Db db)
{
    std::vector<Solution> ss;

// Using const here causes gcc to ICE
#if(!defined(__GNUC__) || defined(__clang__))
    const
#endif
        auto no_perf_filtering =
            miopen::IsDisabled(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING{}) ||
            !miopen::IsEnabled(MIOPEN_DEBUG_FIND_FIRST_CONV{});

    bool skip_the_rest = false;
    miopen::each_args( // clang-format off
        [&](auto solver) { // cppcheck-suppress knownConditionTrueFalse
            if(!skip_the_rest
               && solver.IsApplicable(search_params)
               && (no_perf_filtering || solver.IsFast(search_params)))
            { // clang-format on
                const Solution s = FindSolution(solver, search_params, db);
                if(s.Succeeded())
                {
                    ss.push_back(s);
                    MIOPEN_LOG_I2(SolverDbId(solver) << ": Success.");

                    if(miopen::IsEnabled(MIOPEN_DEBUG_FIND_FIRST_CONV{}))
                    {
                        skip_the_rest = true;
                    }
                    else if(IsPureOpenCLSolution(s))
                    {
                        /// \todo (algorithm == Direct) is not checked here.
                        /// This is ok so far, as SearchForAllSolutions() is used only for direct
                        /// convolutions (for now).
                        if((search_params.direction.IsForward() &&
                            miopen::IsEnabled(
                                MIOPEN_OPENCL_WORKAROUND_FIND_ALL_CONV_DIRECT_FWD{})) ||
                           (search_params.direction.IsBackwardData() &&
                            miopen::IsEnabled(
                                MIOPEN_OPENCL_WORKAROUND_FIND_ALL_CONV_DIRECT_BWD{})) ||
                           (search_params.direction.IsBackwardWrW() &&
                            miopen::IsEnabled(MIOPEN_OPENCL_WORKAROUND_FIND_ALL_CONV_DIRECT_WRW{})))
                        {
                            skip_the_rest = true;
                        }
                    }
                }
                else
                {
                    /// \todo If Solver is applicable it must provide an appropriate Solution.
                    /// This is not the case for some 20x5 convolutions (and possibly others).
                    /// Normally we should not get here and message level should be Error.
                    /// For now, let's use Info (not Warning) level to avoid
                    /// flooding the console.
                    MIOPEN_LOG_I(SolverDbId(solver)
                                 << ": [Warning] Applicable Solver not succeeded.");
                }
            }
            else
            {
                MIOPEN_LOG_I2(SolverDbId(solver) << ": "
                                                 << (skip_the_rest ? "Skipped" : "Not applicable"));
            }
        },
        Solvers{}...);
    return ss;
}

/// Base class for problem solvers.
///
/// Solvers are to be instantiated as const objects and shall not have any variable
/// internal state. Any non-const state information, if required, to be stored in the
/// solver-specific context objects.
///
/// There could be multiple solvers of the same algorithm for a problem config.
/// For example, ConvAsm3x3U and ConvOclDirectFwd3x3
/// are able to solve overlapping sets of 3x3 Direct convolution problems.
template <class Context>
struct SolverBase
{

    /// Initializes performance config to the default values.
    /// The function may involve some euristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    /// Only implemented by SearchableSolvers
    /// PerformanceConfig GetPerformanceConfig(const ConvolutionContext&) const;

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    /// Only implemented by SearchableSolvers
    /// bool IsValidPerformanceConfig(const Context&, const PerformanceConfig&) const
    /// {
    ///     return true; // Do not check by default.
    /// }

    /// Returns true if solution can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    ///
    /// Every SolverBase which IsApplicable() for some problem config must be able to
    /// GetPerformanceConfig() so that GetSolution() would return valid
    /// solution for a problem (i.e. convolution). In other words, if a Solution
    /// says "I'm suitable" for a problem, it agrees to solve that problem correctly.
    bool IsApplicable(const Context&) const { return true; }

    /// Legacy euristic method which shall return false when a solution
    /// is known to be slower than some another solution for the same problem config.
    /// Intended to be used for performance optimization.
    /// Warning: Non-trivial implementations introduce implicit dependencies between solutions.
    bool IsFast(const Context&) const { return true; }

    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    /// ConvSolution GetSolution(const ConvolutionContext& params) const;

    /// Searchable solvers provide a GetSolution that takes a Context and PerformanceConfig
    /// ConvSolution GetSolution(const ConvolutionContext& params,
    ///                          const PerformanceConfig& config) const;

    /// Temporary solver-specific method until we have generic means for running solutions.
    /// int RunAndMeasureSolution(miopen::Handle& profile_h,
    ///                          Data_t bot_ocl_buf,
    ///                          Data_t top_ocl_buf,
    ///                          Data_t wei_ocl_buf,
    ///                          Data_t bias_ocl_buf,
    ///                          const ConvolutionContext& params,
    ///                          const ConvSolution& solution,
    ///                          float& elapsed_time) const;
};

struct PerformanceConfigConvAsm3x3U : Serializable<PerformanceConfigConvAsm3x3U>
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

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue();
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvAsm3x3U& other) const;
    std::string ToString() const;
};

struct ConvAsm3x3U : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsFast(const ConvolutionContext& params) const;
    PerformanceConfigConvAsm3x3U GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvAsm3x3U&) const;
    PerformanceConfigConvAsm3x3U Search(const ConvolutionContext&) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsm3x3U& config,
                             bool disableConfigOverrideFromEnv = false) const;
    int RunAndMeasureSolution(miopen::Handle& profile_h,
                              Data_t bot_ocl_buf,
                              Data_t top_ocl_buf,
                              Data_t wei_ocl_buf,
                              Data_t bias_ocl_buf,
                              const ConvolutionContext& params,
                              const ConvSolution& solution,
                              float& elapsed_time) const;
};

struct PerformanceConfigConvAsm1x1U : Serializable<PerformanceConfigConvAsm1x1U>
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

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue();
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvAsm1x1U& other) const;
    std::string ToString() const;
};

struct ConvAsm1x1U : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvAsm1x1U GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvAsm1x1U&) const;
    PerformanceConfigConvAsm1x1U Search(const ConvolutionContext&) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsFast(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsm1x1U& config,
                             bool disableConfigOverrideFromEnv = false) const;
    int RunAndMeasureSolution(miopen::Handle& profile_h,
                              Data_t bot_ocl_buf,
                              Data_t top_ocl_buf,
                              Data_t wei_ocl_buf,
                              Data_t bias_ocl_buf,
                              const ConvolutionContext& params,
                              const ConvSolution& solution,
                              float& elapsed_time) const;
};

struct PerformanceConfigConvBiasActivAsm1x1U : PerformanceConfigConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U(bool spare) : PerformanceConfigConvAsm1x1U(spare) {}
    PerformanceConfigConvBiasActivAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvBiasActivAsm1x1U& other) const;
};

struct ConvBiasActivAsm1x1U : ConvAsm1x1U
{
    PerformanceConfigConvBiasActivAsm1x1U GetPerformanceConfig(const ConvolutionContext&) const;
    int RunAndMeasureSolution(miopen::Handle& profile_h,
                              Data_t bot_ocl_buf,
                              Data_t top_ocl_buf,
                              Data_t wei_ocl_buf,
                              Data_t bias_ocl_buf,
                              const ConvolutionContext& params,
                              const ConvSolution& solution,
                              float& elapsed_time) const;

    PerformanceConfigConvBiasActivAsm1x1U Search(const ConvolutionContext&) const;
};

struct ConvAsm5x10u2v2f1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvAsm5x10u2v2b1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvAsm7x7c3h224w224k64u2v2p3q3f1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwd11x11 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwdGen : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwd3x3 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
struct ConvOclDirectFwdLegacyExhaustiveSearch : SolverBase<ConvolutionContext>
{
    LegacyPerformanceConfig GetPerformanceConfig(const ConvolutionContext&) const;
    LegacyPerformanceConfig Search(const ConvolutionContext&) const;

    private:
    template <typename Tgpu>
    LegacyPerformanceConfig SearchImpl(const ConvolutionContext&) const;
};

struct ConvOclDirectFwd : ConvOclDirectFwdLegacyExhaustiveSearch
{
    bool IsApplicable(const ConvolutionContext& params) const;

    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& searched_params) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const LegacyPerformanceConfig&) const;

    protected:
    bool IsApplicableBase(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwdFused : ConvOclDirectFwd
{
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& searched_params) const;
};

struct ConvOclDirectFwd1x1 : ConvOclDirectFwdLegacyExhaustiveSearch
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& searched_params) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const LegacyPerformanceConfig&) const
    {
        return true;
    }
};

struct ConvBinWinograd3x3U : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvBinWinogradRxS : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct PerformanceConfigAsmDirect3x3WrW : Serializable<PerformanceConfigAsmDirect3x3WrW>
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

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue();
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigAsmDirect3x3WrW& other) const;
    std::string ToString() const;
};

struct ConvAsmBwdWrW3x3 : SolverBase<ConvolutionContext>
{
    PerformanceConfigAsmDirect3x3WrW GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigAsmDirect3x3WrW&) const;
    PerformanceConfigAsmDirect3x3WrW Search(const ConvolutionContext&) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsFast(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigAsmDirect3x3WrW& config,
                             bool disableConfigOverrideFromEnv = false) const;
    int RunAndMeasureSolution(miopen::Handle& profile_h,
                              Data_t bot_ocl_buf,
                              Data_t top_ocl_buf,
                              Data_t wei_ocl_buf,
                              Data_t bias_ocl_buf,
                              const ConvolutionContext& params,
                              const ConvSolution& solution,
                              float& elapsed_time) const;
};

struct PerformanceConfigConvAsmBwdWrW1x1 : Serializable<PerformanceConfigConvAsmBwdWrW1x1>
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

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidValue() const;
    bool SetNextValue();
    bool IsValid(const ConvolutionContext& config) const;
    bool operator==(const PerformanceConfigConvAsmBwdWrW1x1& other) const;
    std::string ToString() const;
};

struct ConvAsmBwdWrW1x1 : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvAsmBwdWrW1x1 GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvAsmBwdWrW1x1&) const;
    PerformanceConfigConvAsmBwdWrW1x1 Search(const ConvolutionContext&) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsFast(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvAsmBwdWrW1x1& config,
                             bool disableConfigOverrideFromEnv = false) const;
    int RunAndMeasureSolution(miopen::Handle& profile_h,
                              Data_t bot_ocl_buf,
                              Data_t top_ocl_buf,
                              Data_t wei_ocl_buf,
                              Data_t bias_ocl_buf,
                              const ConvolutionContext& params,
                              const ConvSolution& solution,
                              float& elapsed_time) const;
};

/// N_BATCH_LOOPS - {1,2,4,8,16} Num batches processed in single workitem.
///     Required workspace size depends on it. However there is a restriction in the internal
///     Solver API that this shouldn't be so. Therefore the family of Solvers created.
///     Each Solver in the family has constant value of this parameter.
template <int N_BATCH_LOOPS>
struct PerformanceConfigConvOclBwdWrw2
    : Serializable<PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>>
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

    void EuristicInit(const ConvolutionContext& params);
    bool IsValidValue() const;
    bool SetNextValue();
    bool IsValid(const ConvolutionContext& params) const;
    bool operator==(const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& other) const;
    std::string ToString() const;
};

template <int N_BATCH_LOOPS>
struct ConvOclBwdWrW2 : SolverBase<ConvolutionContext>
{
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
    GetPerformanceConfig(const ConvolutionContext&) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>&) const;
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS> Search(const ConvolutionContext&) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config,
                             bool disableConfigOverrideFromEnv = false) const;
    int RunAndMeasureSolution(miopen::Handle& profile_h,
                              Data_t bot_ocl_buf,
                              Data_t top_ocl_buf,
                              Data_t wei_ocl_buf,
                              Data_t bias_ocl_buf,
                              const ConvolutionContext& context,
                              const ConvSolution& solution,
                              float& elapsed_time) const;

    protected:
    bool IsApplicableBase(const ConvolutionContext& params) const;

    private:
    template <typename Tgpu>
    int RunAndMeasureSolutionImpl(miopen::Handle& profile_h,
                                  Data_t bot_ocl_buf,
                                  Data_t top_ocl_buf,
                                  Data_t wei_ocl_buf,
                                  Data_t bias_ocl_buf,
                                  const ConvolutionContext& context,
                                  const ConvSolution& solution,
                                  float& elapsed_time) const;
};

extern template struct ConvOclBwdWrW2<1>;
extern template struct ConvOclBwdWrW2<2>;
extern template struct ConvOclBwdWrW2<4>;
extern template struct ConvOclBwdWrW2<8>;
extern template struct ConvOclBwdWrW2<16>;

/// A separate solver from ConvOclBwdWrW2 to disable auto-tuning for certain configs.
/// Basically, this is *hack* for non-group 3x3 and 1x1 cases.
/// It is assumed that Solutions provided by the ConvOclBwdWrW2 solver
/// would never beat 3x3 and 1x1 assembly WrW kernels, even after tuning.
struct ConvOclBwdWrW2NonTunable : ConvOclBwdWrW2<1>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;

    private:
    // This function dervied from ConvOclBwdWrW2 is declared private
    // so that this solver is not marked searchable/tunable.
    template <int N_BATCH_LOOPS>
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config,
                             bool disableConfigOverrideFromEnv = false) const;
};

struct ConvOclBwdWrW53 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclBwdWrW1x1 : SolverBase<ConvolutionContext>
{
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct AnySolver
{
    AnySolver() : ptr_value(nullptr){};
    template <class U>
    AnySolver(U src) : ptr_value(new AnySolver_tmpl<U>(std::forward<U>(src))){};
    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsApplicable(ctx);
    };
    const std::type_info& Type() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->Type();
    };
    bool IsEmpty() const { return ptr_value == nullptr; };
    bool IsFast(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsFast(ctx);
    };
    ConvSolution FindSolution(const ConvolutionContext& ctx, MultiFileDb& db) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->FindSolution(ctx, db);
    };

    // virtual base class
    struct AnySolver_base
    {
        using ptr = std::shared_ptr<const AnySolver_base>;

        virtual ~AnySolver_base(){};
        virtual bool IsApplicable(const ConvolutionContext& ctx) const = 0;
        virtual bool IsFast(const ConvolutionContext& ctx) const       = 0;
        virtual const std::type_info& Type() const                     = 0;
        virtual ConvSolution FindSolution(const ConvolutionContext& ctx, MultiFileDb& db) const = 0;
    };

    // templated derived class
    template <class T>
    struct AnySolver_tmpl : AnySolver_base
    {
        AnySolver_tmpl(T obj) : value(std::move(obj)){};
        bool IsApplicable(const ConvolutionContext& ctx) const override
        {
            return value.IsApplicable(ctx);
        };
        bool IsFast(const ConvolutionContext& ctx) const override { return value.IsFast(ctx); };
        ConvSolution FindSolution(const ConvolutionContext& ctx, MultiFileDb& db) const override
        {
            return miopen::solver::FindSolution(value, ctx, db);
        };
        const std::type_info& Type() const override { return typeid(T); };

        private:
        T value;
    };

    AnySolver_base::ptr ptr_value;
};

} // namespace solver
} // namespace miopen

struct mlo_construct_direct2D_fusion : mlo_construct_direct2D
{
    mlo_construct_direct2D_fusion(int dir, bool do_bias = false)
        : mlo_construct_direct2D(dir, do_bias)
    {
    }
    mlo_construct_direct2D_fusion(const miopen::TensorDescriptor& in,
                                  const miopen::TensorDescriptor& weights,
                                  const miopen::TensorDescriptor& out,
                                  const miopen::ConvolutionDescriptor& conv,
                                  int dir,
                                  bool do_bias = false)
        : mlo_construct_direct2D(in, weights, out, conv, dir, do_bias)
    {
    }

    inline void mloCopyTo(miopen::ConvolutionContext& params) const /// TODO: get rid of this
    {
        params = _search_params;
    }
    miopen::solver::ConvSolution FindSolution(std::vector<miopen::solver::AnySolver> solvers);
};

#endif // GUARD_MIOPEN_SOLVER_HPP_
