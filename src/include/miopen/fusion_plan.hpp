#pragma once
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/fusion.hpp>
#include <miopen/md_graph.hpp>

namespace miopen {

struct FusionPlanDescriptor : miopenFusionPlanDescriptor
{
    FusionPlanDescriptor(const miopenFusionDirection_t dir, const TensorDescriptor& inDesc);
    ~FusionPlanDescriptor();
    bool isValid() { return is_valid; };
    miopenStatus_t AddOp(std::shared_ptr<FusionOpDescriptor> desc);
    miopenStatus_t RemoveOp(FusionOpDescriptor& desc);
    TensorDescriptor DeriveOutputDescriptor();
    miopenStatus_t
    GetWorkspaceSizeImmed(Handle& handle, size_t& workSpaceSize, miopenConvFwdAlgorithm_t algo);
    miopenStatus_t Execute(Handle& handle,
                           TensorDescriptor& inputDesc,
                           ConstData_t input,
                           TensorDescriptor& outputDesc,
                           Data_t output,
                           const OperatorArgs& op_args);
    miopenStatus_t Compile(Handle& handle);
    friend std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& x);

    void InitMDGraphs();
    void InitConvMDGraph(FusionMDGraph& g);

    protected:
    std::string GetKernelName(Handle& handle);
    std::string GetProgramName(Handle& handle);
    auto GetLocalWGSz();
    auto GetGlobalWGSz();

    private:
    miopenFusionDirection_t fusion_dir;
    const TensorDescriptor& input_desc;
    TensorDescriptor output_desc;
    int op_count = 0;
    std::vector<std::shared_ptr<FusionOpDescriptor>> op_map;
    FusionMDGraph lu;
    // FusionOpLU lu;
    bool is_valid;
    FusionKernelSourceType kernel_source_type;
    bool fp_contains_bn;
    std::string program_name;
    std::string kernel_name;
    std::string algorithm_name;
    std::string network_config;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenFusionPlanDescriptor, miopen::FusionPlanDescriptor);
