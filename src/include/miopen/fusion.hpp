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
#ifndef MIOPEN_FUSION_HPP_
#define MIOPEN_FUSION_HPP_

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>
#include <miopen/convolution.hpp>
#include <miopen/solver.hpp>
#include <miopen/op_kernel_args.hpp>

#include <set>
#include <vector>
#include <unordered_map>

namespace miopen {

// Some utils
namespace solver {
KernelInfo CBAFusionGetSolution(const ConvolutionContext& params);
}

typedef enum {
    miopenFusionOpConv      = 0,
    miopenFusionOpActiv     = 1,
    miopenFusionOpBatchNorm = 2,
    miopenFusionOpPool      = 3,
    miopenFusionOpBias      = 4,
} miopenFusionOp_t;

using any_t = OpKernelArg;
struct OperatorArgs : miopenOperatorArgs
{
    OperatorArgs();
    void ins_arg(std::string name, any_t v);
    friend std::ostream& operator<<(std::ostream& stream, const OperatorArgs& x);
    std::vector<any_t> args_vec;
    std::unordered_map<std::string, any_t> args_map;
};

struct FusionOpDescriptor : miopenFusionOpDescriptor
{
    virtual ~FusionOpDescriptor(){};
    void SetIdx(int _id) { plan_idx = _id; };
    int GetIdx() const { return plan_idx; };
    virtual miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) = 0;
    virtual miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle);
    virtual miopenStatus_t
    GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm = false);
    friend std::ostream& operator<<(std::ostream& stream, const FusionOpDescriptor& x);
    virtual miopenFusionOp_t name()                  = 0;
    virtual std::vector<std::string> GetArgs() const = 0;
    void SetInputDesc(TensorDescriptor i_desc) { input_desc = i_desc; };

    TensorDescriptor input_desc;

    private:
    int plan_idx                       = 0;
    std::shared_ptr<OperatorArgs> args = nullptr;
};

struct BiasFusionOpDescriptor : FusionOpDescriptor
{
    BiasFusionOpDescriptor(TensorDescriptor& desc) : base_desc(desc){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc);
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle);
    miopenStatus_t
    GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm = false);
    miopenStatus_t
    SetArgs(OperatorArgs& args, const void* alpha, const void* beta, ConstData_t dbias);
    std::vector<std::string> GetArgs() const;
    miopenFusionOp_t name() { return miopenFusionOpBias; };
    TensorDescriptor& base_desc;
};

struct ActivFusionOpDescriptor : FusionOpDescriptor
{
    ActivFusionOpDescriptor(miopenActivationMode_t mode) : activMode(mode){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc);
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle);
    miopenStatus_t
    GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm = false);
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           double activAlpha,
                           double activBeta,
                           double activGamm);
    std::vector<std::string> GetArgs() const;
    miopenFusionOp_t name() { return miopenFusionOpActiv; };
    miopenActivationMode_t activMode;
};

struct BatchNormFusionOpDescriptor : FusionOpDescriptor
{
    BatchNormFusionOpDescriptor(TensorDescriptor& desc) : base_desc(desc){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc);
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle);
    miopenStatus_t
    GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm = false);
    // miopenStatus_t SetArgs(OperatorArgs& args, const void* alpha, const void* beta);
    std::vector<std::string> GetArgs() const;
    miopenFusionOp_t name() { return miopenFusionOpActiv; };
    TensorDescriptor& base_desc;
};

struct ConvForwardOpDescriptor : FusionOpDescriptor
{
    ConvForwardOpDescriptor(ConvolutionDescriptor& conv_descriptor,
                            TensorDescriptor& filter_descriptor,
                            miopenConvFwdAlgorithm_t fwd_algo)
        : base_desc(conv_descriptor),
          filter_desc(filter_descriptor),
          algo(fwd_algo),
          kernel_info_valid(false){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc);
    miopenStatus_t SetArgs(OperatorArgs& args, const void* alpha, const void* beta, ConstData_t w);
    std::vector<std::string> GetArgs() const;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle);
    miopenStatus_t
    GetCompileParms(std::string& compile_config, Handle& handle, bool is_asm = false);
    bool isASMApplicable();
    solver::KernelInfo& GetKernelInfo(Handle& handle);
    miopenFusionOp_t name() { return miopenFusionOpConv; };
    ConvolutionDescriptor& base_desc;
    TensorDescriptor& filter_desc;
    miopenConvFwdAlgorithm_t algo;
    solver::KernelInfo kernel_info;
    bool kernel_info_valid;

    private:
    mlo_construct_direct2D_fusion ConstructParams(Handle& handle);
};

struct FusionOpLU
{
    FusionOpLU()
    {
        lut = {
            miopenFusionOpConv, miopenFusionOpBias, miopenFusionOpBatchNorm, miopenFusionOpActiv};
        cur_idx = 0;
    }
    void Reset() { cur_idx = 0; };
    miopenStatus_t Advance(miopenFusionOp_t op);
    auto GetPaths();

    protected:
    std::vector<miopenFusionOp_t> lut;
    std::vector<int> lut_hit;
    size_t cur_idx;
};

struct FusionPlanDescriptor : miopenFusionPlanDescriptor
{
    FusionPlanDescriptor(const miopenFusionDirection_t dir, const TensorDescriptor& inDesc)
        : fusion_dir(dir), input_desc(inDesc), is_valid(false){};
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
    friend std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& x);

    protected:
    std::string GetKernelName(Handle& handle);
    std::string GetProgramName();
    auto GetLocalWGSz();
    auto GetGlobalWGSz();

    private:
    miopenFusionDirection_t fusion_dir;
    const TensorDescriptor& input_desc;
    TensorDescriptor output_desc;
    int op_count = 0;
    std::unordered_map<int, std::shared_ptr<FusionOpDescriptor>> op_map;
    std::vector<int> ins_order;
    FusionOpLU lu;
    bool is_valid;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenFusionOpDescriptor, miopen::FusionOpDescriptor);
MIOPEN_DEFINE_OBJECT(miopenFusionPlanDescriptor, miopen::FusionPlanDescriptor);
MIOPEN_DEFINE_OBJECT(miopenOperatorArgs, miopen::OperatorArgs);

#endif // _MIOPEN_FUSION_HPP_
