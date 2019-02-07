#pragma once
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/fusion.hpp>
#include <miopen/md_graph.hpp>

namespace miopen {

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

struct FusionPlanDescriptor : miopenFusionPlanDescriptor
{
    FusionPlanDescriptor(miopenFusionDirection_t dir, const TensorDescriptor& inDesc);
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
    friend std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& fpd);

    miopenStatus_t
    GetConvAlgos(int reqAlgoCount, int& retAlgoCount, miopenConvFwdAlgorithm_t* ptrAlgos);
    miopenStatus_t SetConvAlgo(miopenConvFwdAlgorithm_t algo);

    miopenStatus_t GetOp(int op_idx, std::shared_ptr<FusionOpDescriptor>& desc);

    std::string GetKernelName(Handle& handle);
    std::string GetProgramName(Handle& handle);
    std::string GetAlgorithmName(Handle& handle);

    protected:
    auto GetLocalWGSz();
    auto GetGlobalWGSz();
    std::vector<Exec_arg_t> CalcArgOrder(Handle& handle);
    bool GetEnumVal(const std::string& sym, int& val) const;
    OpKernelArg GetDevAttribute(const std::string& k, Handle& handle) const;
    OpKernelArg GetTensorAttr(const std::string& sym) const;
    bool GetTensorAttr(const std::string& sym, int& val) const;

    private:
    miopenFusionDirection_t fusion_dir;
    TensorDescriptor input_desc;
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
    miopenDataType_t data_type;
    std::vector<Exec_arg_t> arg_list;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenFusionPlanDescriptor, miopen::FusionPlanDescriptor);
