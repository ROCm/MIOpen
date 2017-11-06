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

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <ostream>

#include <miopen/db_record.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/make_unique.hpp>
#include <miopen/env.hpp>
#include <miopen/miopen.h>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)

namespace solver {

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
    std::vector<KernelInfo> construction_params; // impl may consist of multiple kernels.
    miopenStatus_t status;
    int passes;

    size_t workspce_sz;
    int grp_tile1;
    int grp_tile0;
    int in_tile1;
    int in_tile0;
    int out_pix_tile1;
    int out_pix_tile0;
    int n_out_pix_tiles;
    int n_in_data_tiles;
    int n_stacks;

    ConvSolution(miopenStatus_t status_ = miopenStatusSuccess, int passes_ = 1)
        : status(status_),
          passes(passes_),
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

// Search for a solution among many solvers
template <class... Solvers, class Context>
miopen::solver::ConvSolution SearchForSolution(const Context& search_params,
                                               miopen::DbRecord dbRecord)
{
    miopen::solver::ConvSolution solution{miopenStatusUnknownError};

// Using const here causes gcc to ICE
#if(!defined(__GNUC__) || defined(__clang__))
    const
#endif
        auto no_perf_filtering = miopen::IsDisabled(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING{});

    // clang-format off
    MIOPEN_STATIC_FOR_EACH(solver, Solvers{}, {
        if(!solution.Succeeded() && solver.IsApplicable(search_params) &&
           (no_perf_filtering || solver.IsFast(search_params)))
        {
            solution = FindSolution(solver, search_params, dbRecord);
            if(solution.Succeeded() && solution.construction_params.empty())
            {
                MIOPEN_THROW(std::string("Internal error in solver: ") + typeid(solver).name());
            }
        }
    });
    // clang-format on

    return solution;
}

/// Finds optimized Solution. Generic method.
///
/// Given the specific problem config, finds (hopefully) optimal
/// solution-specific parameters and returns the Solution object.
/// Could take long if an exhaustive search is requested/performed.
/// May read/write perfDb.
template <class Solver, class Context>
ConvSolution FindSolution(Solver s, const Context& context, DbRecord& dbRecord)
{
    static_assert(std::is_empty<Solver>{}, "Solver must be stateless");
    // TODO: This assumes all solutions are ConvSolution
    return SearchSolution(rank<1>{}, s, context, dbRecord);
}

template <class Solver, class Context>
auto SearchSolution(rank<1>, Solver s, const Context& context, DbRecord& dbRecord)
    -> decltype(s.GetSolution(context, s.Search(context)))
{
    auto config = s.GetPerformanceConfig();
    MIOPEN_LOG_I("Finding solution: " << s.SolverId());
    if(dbRecord.Load(s.SolverId(), config))
    {
        MIOPEN_LOG_I("Perf Db: record loaded: " << s.SolverId());
        if(s.IsValidPerformanceConfig(context, config))
        {
            return s.GetSolution(context, config);
        }
        MIOPEN_LOG_E("Invalid config loaded from Perf Db: " << s.SolverId() << ": " << config);
    }
    else if(context.do_search)
    {
        MIOPEN_LOG_I("Starting search: " << s.SolverId());
        auto c = s.Search(context);
        dbRecord.Store(s.SolverId(), c);
        return s.GetSolution(context, c);
    }

    s.InitPerformanceConfigImpl(context, config);
    return s.GetSolution(context, config);
}

template <class Solver, class Context>
auto SearchSolution(rank<0>, Solver s, const Context& context, DbRecord&)
    -> decltype(s.GetSolution(context))
{
    MIOPEN_LOG_I("Not searchable: " << s.SolverId());
    return s.GetSolution(context);
}

/// The descendants of this class comprise an solution-specific
/// set of optimization parameters, i.e. those which expected to be used by
/// the solution to optimize its kernel(s) for the best performance.
///
/// This class provides its descendants with polymorphism and supplies syntax
/// glue at the source text level. Also serves as en "empty set of parameters"
/// for solutions which do not have parameters that affect performance
/// (e.g. for 3x3 Wingrad convolutions).
struct PerformanceConfig
{
    void Serialize(std::ostream&) const {}
    bool Deserialize(const std::string& s) { return s.empty(); }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    bool LegacyDeserialize(const std::string&) { return false; }
#endif
    friend std::ostream& operator<<(std::ostream& os, const PerformanceConfig& c)
    {
        c.Serialize(os); // Can be used here as provides text.
        return os;
    }
};

/// Base class for problem solvers.
///
/// Solvers are to be instantiated as const objects and shall not have any variable
/// internal state. Any non-const state information, if required, to be stored in the
/// solver-specific context objects.
///
/// There could be multiple solvers of the same algorithm for a problem config.
/// For example, ConvAsm3x3U and ConvOclDirectFwd3x3
/// are able to solve overlapping sets of 3x3 Direct convolution problems.
struct Solver
{
    /// Constructs performance config instance used by a Solver.
    // PerformanceConfig GetPerformanceConfig() const { return {}; }

    /// Initializes performance config to the default values.
    /// The function may involve some euristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    ///
    /// Every Solver which overrides GetPerformanceConfig() shall
    /// override this function as well.
    // void InitPerformanceConfigImpl(const ConvolutionContext&, PerformanceConfig& c) const
    // {
    //     c = PerformanceConfig();
    // }

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    // bool IsValidPerformanceConfig(const ConvolutionContext&, const PerformanceConfig&) const
    // {
    //     return true; // Do not check by default.
    // }

    /// Returns true if solution can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    ///
    /// Every Solver which IsApplicable() for some problem config, must be able to
    /// GetPerformanceConfig() in a way that GetSolution() would return valid
    /// solution for a problem (i.e. convolution). In other words, if a Solution
    /// says "i'am suitable" for a problem, it agrees to solve the problem correctly.
    bool IsApplicable(const ConvolutionContext&) const { return true; }

    /// Legacy euristic method which shall return false when a solution
    /// is known to be slower than some another solution for the same problem config.
    /// Intended to be used for performance optimization.
    /// Warning: Non-trivial implementations introduce implicit dependencies between solutions.
    bool IsFast(const ConvolutionContext&) const { return true; }

    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    // virtual ConvSolution GetSolution(const ConvolutionContext&, const PerformanceConfig&) const =
    // 0;
};

struct ConvAsm3x3U : Solver
{
    const char* SolverId() const { return "ConvAsm3x3U"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsFast(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvAsm5x10u2v2f1 : Solver
{
    const char* SolverId() const { return "ConvAsm5x10u2v2f1"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvAsm5x10u2v2b1 : Solver
{
    const char* SolverId() const { return "ConvAsm5x10u2v2b1"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvAsm7x7c3h224w224k64u2v2p3q3f1 : Solver
{
    const char* SolverId() const { return "ConvAsm7x7c3h224w224k64u2v2p3q3f1"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwd11x11 : Solver
{
    const char* SolverId() const { return "ConvOclDirectFwd11x11"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwdGen : Solver
{
    const char* SolverId() const { return "ConvOclDirectFwdGen"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclDirectFwd3x3 : Solver
{
    const char* SolverId() const { return "ConvOclDirectFwd3x3"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
struct ConvOclDirectFwdLegacyExhaustiveSearch : Solver
{
    LegacyPerformanceConfig GetPerformanceConfig() const;
    void InitPerformanceConfigImpl(const ConvolutionContext&,
                                   LegacyPerformanceConfig& result_) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&, const LegacyPerformanceConfig&) const
    {
        return true; // Do not check by default.
    }
    LegacyPerformanceConfig Search(const ConvolutionContext&) const;
};

struct ConvOclDirectFwd : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    const char* SolverId() const { return "ConvOclDirectFwd"; }
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& config) const;
};

struct ConvOclDirectFwd1x1 : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    const char* SolverId() const { return "ConvOclDirectFwd1x1"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& config) const;
};

struct ConvOclDirectFwdC : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    const char* SolverId() const { return "ConvOclDirectFwdC"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const LegacyPerformanceConfig& config) const;
};

struct ConvBinWinograd3x3U : Solver
{
    const char* SolverId() const { return "ConvBinWinograd3x3U"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvBinWinogradRxSFwd : Solver
{
    const char* SolverId() const { return "ConvBinWinogradRxSFwd"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct PerformanceConfigAsmDirect3x3WrW
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
    void Serialize(std::ostream&) const;
    bool Deserialize(const std::string& str);
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    bool LegacyDeserialize(const std::string&) { return false; }
#endif

    // clang-format off
    int GetLimitWaveCnt() const { return limit_wave_cnt; }
    int GetReverseInout() const { return reverse_inout; }
    int GetChunkSize() const { return chunk_size; }
    int GetKPerWave() const { return k_per_wave; }
    int GetPipeLinesDepth() const { return pipe_lines_depth; }
    int GetNPerGroup() const { return n_per_group; } 
    int GetCPerWave() const { assert(chunk_size); return 64 / chunk_size; } // clang-format on

    void EuristicInit(const ConvolutionContext& config);
    bool IsValidRange() const;
    bool IsValid(const ConvolutionContext& config) const;
    // TOOD: Use operator==
    bool IsEqual(const PerformanceConfigAsmDirect3x3WrW& other) const;
    std::string ToString() const;

    friend class VirtualIterator; // Modifies private data when advancing.

    friend std::ostream& operator<<(std::ostream& os, const PerformanceConfigAsmDirect3x3WrW& c)
    {
        c.Serialize(os); // Can be used here as provides text.
        return os;
    }
};

struct ConvAsmBwdWrW3x3 : Solver
{
    const char* SolverId() const { return "ConvAsmBwdWrW3x3"; }
    PerformanceConfigAsmDirect3x3WrW GetPerformanceConfig() const;
    void InitPerformanceConfigImpl(const ConvolutionContext&,
                                   PerformanceConfigAsmDirect3x3WrW& result) const;
    bool IsValidPerformanceConfig(const ConvolutionContext&,
                                  const PerformanceConfigAsmDirect3x3WrW&) const;
    PerformanceConfigAsmDirect3x3WrW Search(const ConvolutionContext&) const;
    bool IsApplicable(const ConvolutionContext& params) const;
    bool IsFast(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfigAsmDirect3x3WrW& config) const;

    private:
    int Measure(miopen::Handle& profile_h,
                Data_t bot_ocl_buf,
                Data_t top_ocl_buf,
                Data_t wei_ocl_buf,
                double& processing_time,
                const ConvolutionContext& params,
                const PerformanceConfig& result) const;
};

struct ConvOclBwdWrW2 : Solver
{
    const char* SolverId() const { return "ConvOclBwdWrW2"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclBwdWrW53 : Solver
{
    const char* SolverId() const { return "ConvOclBwdWrW53"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

struct ConvOclBwdWrW1x1 : Solver
{
    const char* SolverId() const { return "ConvOclBwdWrW1x1"; }
    bool IsApplicable(const ConvolutionContext& params) const;
    ConvSolution GetSolution(const ConvolutionContext& params) const;
};

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_SOLVER_HPP_
