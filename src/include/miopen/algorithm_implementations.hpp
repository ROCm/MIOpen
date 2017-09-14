#ifndef KERNEL_TRAITS_HPP_
#define KERNEL_TRAITS_HPP_

#include <vector>
#include <memory>
#include "miopen/mlo_internal.hpp"
#include "miopen/miopen.h"

namespace miopen {

/// Describes a kernel source and whatever information required in order
/// to build and run it (the former is unused for binary kernels).
struct KernelInfo
{
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::string kernel_file;
    std::string kernel_name;
};

/// Information required to build and run a particular implementation of an algorithm.
///
/// TODO: This one is suitable for a subset of existing OpenCL-written forward direct
/// convolution implementations. Shall we elaborate this to a class hierarchy so that
/// descendants would be implementation-specific.
class ImplementationUsageDescription
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

    ImplementationUsageDescription(miopenStatus_t status_ = miopenStatusSuccess, int passes_ = 1)
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

/// Base class for algorithm implementations.
///
/// Note that there could be multiple implementations of the same algorithm for
/// the same problem config. For example, both ConvAsm3x3U and ConvOclDirectFwd3x3
/// implementations support 3x3 Direct convolution.
class Implementation
{
    public:
    /// The descendants of this class comprise an implementation-specific
    /// set of optimization parameters, i.e. those which expected to be used by
    /// an implementation to optimize its kernel(s) for the best performance.
    ///
    /// This class provides its descendants with polymorphism and supplies syntax
    /// glue at the source text level. Also serves as en "empty set of parameters"
    /// for implementations which do not have parameters that affect performance.
    class PerformanceConfig
    {
        public:
        PerformanceConfig() noexcept {}
        virtual ~PerformanceConfig() {}
    };

    virtual ~Implementation() {}

    /// Returns true if an implementation is suitable and expected to work correctly
    /// for given problem config on the given platform (runtime and device).
    virtual bool IsCorrect(const SearchParameters&) const { return true; }

    /// Legacy euristic method which shall return false when an implementation of an algorithm
    /// is known to be slower than some another implementation of the same algorithm.
    /// Intended to be used for performance optimization.
    /// Warning: Introduces implicit dependencies between implementations.
    virtual bool IsFast(const SearchParameters&) const { return true; }

    /// Finds optimization parameters for a given problem config.
    /// Could take long if an exhaustive search is performed/requested.
    ///
    /// Legacy behavior:
    ///     Lookup for a suitable PerformanceConfig in the perfDb;
    ///     if (found) {
    ///         return (value found in the PerfDb);
    ///     } else if (exhaustive search is requested) {
    ///         Do exhaustive search (can be slow);
    ///         Update PerfDb with the PerformanceConfig found;
    ///         return (value found);
    ///     }
    ///     return (implementation-specific defaults); /* may involve some heuristic math */
    ///
    virtual std::unique_ptr<PerformanceConfig>
    Find(const SearchParameters&) const
    {
        return std::unique_ptr<PerformanceConfig>(new PerformanceConfig());
    }

    /// Takes problem config, optimization parameters and other info
    /// and computes information required to build and run the kernel(s).
    virtual void
    PrepareForUsage(ImplementationUsageDescription& out,
                    const SearchParameters& params,
                    const PerformanceConfig& exhaustive_search_result) const = 0;
};

class ConvAsm3x3U : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    bool IsFast(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvAsm5x10u2v2f1 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvAsm5x10u2v2b1 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvAsm7x7c3h224w224k64u2v2p3q3f1 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclDirectFwd11x11 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclDirectFwdGen : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclDirectFwd3x3 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclDirectFwdLegacyExhaustiveSearch : public Implementation
{
    public:
    class PerformanceConfigImpl : public PerformanceConfig
    {
        public:
        int grp_tile1;
        int grp_tile0;
        int in_tile1;
        int in_tile0;
        int out_pix_tile1;
        int out_pix_tile0;
        int n_out_pix_tiles;
        int n_in_data_tiles;
        int n_stacks;
    
        PerformanceConfigImpl() noexcept : grp_tile1(),
                                                grp_tile0(),
                                                in_tile1(),
                                                in_tile0(),
                                                out_pix_tile1(),
                                                out_pix_tile0(),
                                                n_out_pix_tiles(),
                                                n_in_data_tiles(),
                                                n_stacks()
        {
        }
    
        inline void CopyTo(ImplementationUsageDescription& iud) const
        {
            iud.grp_tile0       = grp_tile0;
            iud.grp_tile1       = grp_tile1;
            iud.in_tile0        = in_tile0;
            iud.in_tile1        = in_tile1;
            iud.out_pix_tile0   = out_pix_tile0;
            iud.out_pix_tile1   = out_pix_tile1;
            iud.n_out_pix_tiles = n_out_pix_tiles;
            iud.n_in_data_tiles = n_in_data_tiles;
            iud.n_stacks        = n_stacks;
        }
    };

    std::unique_ptr<PerformanceConfig>
    Find(const SearchParameters& params) const override;

    private:
    void SearchDirect2D(const SearchParameters& params,
                        PerformanceConfigImpl& result) const;

    int MeasureLoop(miopen::Handle* profile_h,
                     Data_t bot_ocl_buf,
                     Data_t top_ocl_buf,
                     Data_t wei_ocl_buf,
                     Data_t bias_ocl_buf,
                     double& processing_time,
                     const SearchParameters& params) const;

    static const std::vector<std::unique_ptr<const Implementation>>&
    GetImplementationsToMeasure();
};

class ConvOclDirectFwd : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclDirectFwd1x1 : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclDirectFwdC : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvBinWinograd3x3U : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvBinWinogradRxSFwd : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvAsmBwdWrW3x3 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    bool IsFast(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclBwdWrW2 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclBwdWrW53 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};

class ConvOclBwdWrW1x1 : public Implementation
{
    public:
    bool IsCorrect(const SearchParameters& params) const override;
    void PrepareForUsage(ImplementationUsageDescription& out,
                         const SearchParameters& params,
                         const PerformanceConfig& exhaustive_search_result) const override;
};
} // namespace miopen

#endif // !KERNEL_TRAITS_HPP
