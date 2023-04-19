
#ifndef MIOPEN_GUARD_MLOPEN_FUSION_PLAN_HPP
#define MIOPEN_GUARD_MLOPEN_FUSION_PLAN_HPP

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/fusion.hpp>

#include <boost/optional.hpp>

namespace miopen {

namespace solver {
struct ConvSolution;
} // namespace solver

//"Fusion mode (cbna = 0, cna = 1, na = 2, cn = 3, cba = 4, ca = 5, cb = 6) (Default=cbna)",
enum fusionMode_t
{
    miopen_fusion_cbna = 0,
    miopen_fusion_cna  = 1,
    miopen_fusion_na   = 2,
    miopen_fusion_cn   = 3,
    miopen_fusion_cba  = 4,
    miopen_fusion_ca   = 5,
    miopen_fusion_cb   = 6,
} ;

enum Exec_Arg_Type_t
{
    Scalar,
    Input_Ptr,
    Output_Ptr,
    Pointer,
    Padding,
    Default
};

struct Exec_arg_t
{
    std::string key;
    Exec_Arg_Type_t type;
    int size;
    OpKernelArg val;
    Exec_arg_t(std::string k, Exec_Arg_Type_t t, int s)
        : key(std::move(k)), type(t), size(s), val(OpKernelArg(0))
    {
    }
    Exec_arg_t(std::string k, Exec_Arg_Type_t t, int s, OpKernelArg v)
        : key(std::move(k)), type(t), size(s), val(v)
    {
    }
};

struct FusionContext;
struct FusionPlanDescriptor : miopenFusionPlanDescriptor
{
    FusionPlanDescriptor() {}
    FusionPlanDescriptor(miopenFusionDirection_t dir, const TensorDescriptor& inDesc, int fmode);
    bool isValid() const { return is_valid; };
    miopenStatus_t AddOp(std::shared_ptr<FusionOpDescriptor> desc);
    TensorDescriptor DeriveOutputDescriptor();
    miopenStatus_t
    GetWorkspaceSizeImmed(Handle& handle, size_t& workSpaceSize, miopenConvFwdAlgorithm_t algo);
    miopenStatus_t Execute(const Handle& handle,
                           const TensorDescriptor& inputDesc,
                           ConstData_t input,
                           const TensorDescriptor& outputDesc,
                           Data_t output,
                           const OperatorArgs& op_args);
    miopenStatus_t Compile(Handle& handle);
    friend std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& fpd);

    miopenStatus_t
    GetConvAlgos(int reqAlgoCount, int& retAlgoCount, miopenConvFwdAlgorithm_t* ptrAlgos);
    miopenStatus_t SetConvAlgo(miopenConvFwdAlgorithm_t algo);

    miopenStatus_t GetOp(int op_idx, std::shared_ptr<FusionOpDescriptor>& desc);

    std::string GetAlgorithmName(const Handle& handle);
    std::vector<std::shared_ptr<FusionOpDescriptor>> op_map;

    miopenFusionDirection_t fusion_dir;
    TensorDescriptor input_desc;
    TensorDescriptor output_desc;
    int op_count = 0;
    bool is_valid;
    FusionKernelSourceType kernel_source_type;
    bool fp_contains_bn;
    miopenDataType_t data_type;
    std::vector<Exec_arg_t> arg_list;
    std::vector<solver::ConvSolution> solutions;
    NetworkConfig network_config;
    std::optional<miopenConvFwdAlgorithm_t> conv_fwd_algo;

    fusionMode_t GetFusionMode() const { return fusion_mode; }

private:
    fusionMode_t fusion_mode;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenFusionPlanDescriptor, miopen::FusionPlanDescriptor);

#endif
