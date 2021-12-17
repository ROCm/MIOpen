#pragma once

#include <boost/any.hpp>

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

    /// Initializes performance config to the default values.
    /// The function may involve some heuristic to guess the best solution
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
    virtual bool IsApplicable(const boost::any& ctx) const = 0;

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
    virtual float GetWti(const boost::any& ctx) const = 0;

    // Returns the workspace size required by the solver for a given ConvolutionContext
    virtual size_t GetWorkspaceSize(const boost::any& ctx) const = 0;

    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    /// ConvSolution GetSolution(const ConvolutionContext& params) const;

    /// Searchable solvers provide a GetSolution that takes a Context and PerformanceConfig
    /// ConvSolution GetSolution(const ConvolutionContext& params,
    ///                          const PerformanceConfig& config) const;
};
