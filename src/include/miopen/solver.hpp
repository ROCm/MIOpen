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
class ConvSolution
{
    public:
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
#if (!defined(__GNUC__) || defined (__clang__))
    const
#endif
    auto no_perf_filtering =
        miopen::IsDisabled(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING{});

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
    // TODO: This assumes all solutions are ConvSolution
    return SearchSolution(rank<1>{}, s, context, dbRecord, s.PerformanceConfigImpl());
}

template <class Solver, class Context, class Config>
auto SearchSolution(rank<1>, Solver s, const Context& context, DbRecord& dbRecord, Config config)
    -> decltype(s.Search(context, *config), s.GetSolution(context, *config))
{
    MIOPEN_LOG_I("Finding solution: " << s.SolverId());
    if(dbRecord.Load(s.SolverId(), *config))
    {
        MIOPEN_LOG_I("Perf Db: record loaded: " << s.SolverId());
        if(s.IsValidPerformanceConfigImpl(context, *config))
        {
            return s.GetSolution(context, *config);
        }
        MIOPEN_LOG_E("Invalid config loaded from Perf Db: " << s.SolverId() << ": " << *config);
    }
    else if(context.do_search)
    {
        MIOPEN_LOG_I("Starting search: " << s.SolverId());
        if(s.Search(context, *config))
        {
            dbRecord.Store(s.SolverId(), *config);
            return s.GetSolution(context, *config);
        }
        MIOPEN_LOG_E("Search failed: " << s.SolverId());
    }

    s.InitPerformanceConfigImpl(context, *config);
    return s.GetSolution(context, *config);
}

template <class Solver, class Context, class Config>
auto SearchSolution(rank<0>, Solver s, const Context& context, DbRecord&, Config config)
    -> decltype(s.GetSolution(context, *config))
{
    MIOPEN_LOG_I("Not searchable: " << s.SolverId());
    s.InitPerformanceConfigImpl(context, *config);
    return s.GetSolution(context, *config);
}

/// The descendants of this class comprise an solution-specific
/// set of optimization parameters, i.e. those which expected to be used by
/// the solution to optimize its kernel(s) for the best performance.
///
/// This class provides its descendants with polymorphism and supplies syntax
/// glue at the source text level. Also serves as en "empty set of parameters"
/// for solutions which do not have parameters that affect performance
/// (e.g. for 3x3 Wingrad convolutions).
class PerformanceConfig
{
    public:
    PerformanceConfig() {}
    PerformanceConfig(const PerformanceConfig&) = default;
    PerformanceConfig& operator=(const PerformanceConfig&) = default;
    virtual ~PerformanceConfig() {}
    virtual void Serialize(std::ostream&) const {}
    virtual bool Deserialize(const std::string& s) { return s.empty(); }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    virtual bool LegacyDeserialize(const std::string&) { return false; }
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
class Solver
{
    public:
    virtual ~Solver() {}

    // TODO: Make solvers regular
    Solver()              = default;
    Solver(const Solver&) = default;
    Solver& operator=(const Solver&) = default;

    /// Each non-abstract descendant shall have unique id.
    virtual const char* SolverId() const = 0;

    /// Constructs performance config instance used by a Solver.
    virtual std::unique_ptr<PerformanceConfig> PerformanceConfigImpl() const
    {
        return make_unique<PerformanceConfig>();
    }

    /// Initializes performance config to the default values.
    /// The function may involve some euristic to guess the best solution
    /// configuration. It is assumed that the function takes constant time
    /// to finish and does not run kernels to measure performance etc.
    /// The function shall always return valid config.
    ///
    /// Every Solver which overrides PerformanceConfigImpl() shall
    /// override this function as well.
    virtual void InitPerformanceConfigImpl(const ConvolutionContext&, PerformanceConfig& c) const
    {
        c = PerformanceConfig();
    }

    /// Should return false if performance config is wrong for a problem.
    /// Main use is validation of values read from the perf db.
    virtual bool IsValidPerformanceConfigImpl(const ConvolutionContext&,
                                              const PerformanceConfig&) const
    {
        return true; // Do not check by default.
    }

    /// Solver-specific implementation of exhaustive search procedure.
    /// Returns (hopefully) optimal performance config for a solution.
    /// May takes long time to finish.
    virtual bool Search(const ConvolutionContext&, PerformanceConfig&) const { return false; }

    /// Should return true if Search() is implemented in a Solver, false otherwise.
    virtual bool IsSearchable() const { return false; }

    /// Returns true if solution can work on given SW/HW platform (runtime/device)
    /// and provides correct result for the problem config.
    ///
    /// Every Solver which IsApplicable() for some problem config, must be able to
    /// InitPerformanceConfigImpl() in a way that GetSolution() would return valid
    /// solution for a problem (i.e. convolution). In other words, if a Solution
    /// says "i'am suitable" for a problem, it agrees to solve the problem correctly.
    virtual bool IsApplicable(const ConvolutionContext&) const { return true; }

    /// Legacy euristic method which shall return false when a solution
    /// is known to be slower than some another solution for the same problem config.
    /// Intended to be used for performance optimization.
    /// Warning: Non-trivial implementations introduce implicit dependencies between solutions.
    virtual bool IsFast(const ConvolutionContext&) const { return true; }

    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    virtual ConvSolution GetSolution(const ConvolutionContext&, const PerformanceConfig&) const = 0;
};

class ConvAsm3x3U : public Solver
{
    public:
    const char* SolverId() const override { return "ConvAsm3x3U"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    bool IsFast(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvAsm5x10u2v2f1 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvAsm5x10u2v2f1"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvAsm5x10u2v2b1 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvAsm5x10u2v2b1"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvAsm7x7c3h224w224k64u2v2p3q3f1 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvAsm7x7c3h224w224k64u2v2p3q3f1"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvOclDirectFwd11x11 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvOclDirectFwd11x11"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvOclDirectFwdGen : public Solver
{
    public:
    const char* SolverId() const override { return "ConvOclDirectFwdGen"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvOclDirectFwd3x3 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvOclDirectFwd3x3"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
class ConvOclDirectFwdLegacyExhaustiveSearch : public Solver
{
    public:
    std::unique_ptr<PerformanceConfig> PerformanceConfigImpl() const override;
    void InitPerformanceConfigImpl(const ConvolutionContext&,
                                   PerformanceConfig& result_) const override;
    bool Search(const ConvolutionContext&, PerformanceConfig& result_) const override;
    bool IsSearchable() const override { return true; }
};

class ConvOclDirectFwd : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    const char* SolverId() const override { return "ConvOclDirectFwd"; }
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvOclDirectFwd1x1 : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    const char* SolverId() const override { return "ConvOclDirectFwd1x1"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvOclDirectFwdC : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    const char* SolverId() const override { return "ConvOclDirectFwdC"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvBinWinograd3x3U : public Solver
{
    public:
    const char* SolverId() const override { return "ConvBinWinograd3x3U"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvBinWinogradRxSFwd : public Solver
{
    public:
    const char* SolverId() const override { return "ConvBinWinogradRxSFwd"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvAsmBwdWrW3x3 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvAsmBwdWrW3x3"; }
    std::unique_ptr<PerformanceConfig> PerformanceConfigImpl() const override;
    void InitPerformanceConfigImpl(const ConvolutionContext&,
                                   PerformanceConfig& result) const override;
    bool IsValidPerformanceConfigImpl(const ConvolutionContext&,
                                      const PerformanceConfig&) const override;
    bool IsSearchable() const override { return true; }
    bool Search(const ConvolutionContext&, PerformanceConfig& config) const override;
    bool IsApplicable(const ConvolutionContext& params) const override;
    bool IsFast(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;

    private:
    int Measure(miopen::Handle& profile_h,
                Data_t bot_ocl_buf,
                Data_t top_ocl_buf,
                Data_t wei_ocl_buf,
                double& processing_time,
                const ConvolutionContext& params,
                const PerformanceConfig& result) const;
};

class ConvOclBwdWrW2 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvOclBwdWrW2"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvOclBwdWrW53 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvOclBwdWrW53"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

class ConvOclBwdWrW1x1 : public Solver
{
    public:
    const char* SolverId() const override { return "ConvOclBwdWrW1x1"; }
    bool IsApplicable(const ConvolutionContext& params) const override;
    ConvSolution GetSolution(const ConvolutionContext& params,
                             const PerformanceConfig& config) const override;
};

} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_SOLVER_HPP_
