#ifndef KERNEL_TRAITS_HPP_
#define KERNEL_TRAITS_HPP_

#include <vector>
#include <memory>
#include "miopen/mlo_internal.hpp"
#include "miopen/miopen.h"

namespace miopen {

/// Describes a kernel source and whatever information required in order
/// to build (except kernels supplied in binary format) and run it.
struct KernelUsageDescription
{
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::string kernel_file;
    std::string kernel_name;
};

/// Information used to build and run an implementation of particular
/// algorithm/action (primitive, i.e. Winograd/convolution).
/// An implementation may consist of multiple kernels.
class ImplementationUsageDescription
{
    public:
    std::vector<KernelUsageDescription> construction_params;
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


/// The "Exchaustive search result" classes represent implementation-specific
/// sets of parameters. This class is a common ancestor for those.
/// Provides polymorphism and syntax glue.
class ExhaustiveSearchResult
{
    public:
    ExhaustiveSearchResult() noexcept {}
    virtual ~ExhaustiveSearchResult() {}
};

class NoneExhaustiveSearchResult : public ExhaustiveSearchResult
{
};

class Direct2DfwdExhaustiveSearchResult : public ExhaustiveSearchResult
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

    Direct2DfwdExhaustiveSearchResult() noexcept : grp_tile1(),
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

class ImplementationDescription
{
    public:
    virtual std::unique_ptr<ExhaustiveSearchResult>
    PrepareExhaustiveSearchResult(const ImplementationSearchParameters&) const
    {
        return std::unique_ptr<ExhaustiveSearchResult>(new NoneExhaustiveSearchResult());
    }
    virtual ~ImplementationDescription() {}
    virtual bool IsCorrect(const ImplementationSearchParameters&) const { return true; }
    virtual bool IsFast(const ImplementationSearchParameters&) const { return true; }
    virtual ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const = 0;
};

class ConvAsm3x3U : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    bool IsFast(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvAsm5x10u2v2f1 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvAsm5x10u2v2b1 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvAsm7x7c3h224w224k64u2v2p3q3f1 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclDirectFwd11x11 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclDirectFwdGen : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclDirectFwd3x3 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclDirectFwdLegacyExhaustiveSearch : public ImplementationDescription
{
    public:
    std::unique_ptr<ExhaustiveSearchResult>
    PrepareExhaustiveSearchResult(const ImplementationSearchParameters& params) const override;

    private:
    void SearchDirect2D(const ImplementationSearchParameters& params,
                        Direct2DfwdExhaustiveSearchResult& result) const;

    int MeasuredLoop(miopen::Handle* profile_h,
                     Data_t bot_ocl_buf,
                     Data_t top_ocl_buf,
                     Data_t wei_ocl_buf,
                     Data_t bias_ocl_buf,
                     double& processing_time,
                     const ImplementationSearchParameters& params) const;

    static const std::vector<std::unique_ptr<const ImplementationDescription>>&
    GetImplementationsToMeasure();
};

class ConvOclDirectFwd : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclDirectFwd1x1 : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclDirectFwdC : public ConvOclDirectFwdLegacyExhaustiveSearch
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvBinWinograd3x3U : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvBinWinogradRxSFwd : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvAsmBwdWrW3x3 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    bool IsFast(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclBwdWrW2 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclBwdWrW53 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};

class ConvOclBwdWrW1x1 : public ImplementationDescription
{
    public:
    bool IsCorrect(const ImplementationSearchParameters& params) const override;
    ImplementationUsageDescription
    PrepareForUsage(const ImplementationSearchParameters& params,
                    const ExhaustiveSearchResult& exhaustive_search_result) const override;
};
} // namespace miopen

#endif // !KERNEL_TRAITS_HPP
