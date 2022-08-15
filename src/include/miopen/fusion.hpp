/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/solver.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/fusion_ops.hpp>

#include <set>
#include <vector>
#include <unordered_map>

namespace miopen {

struct Handle;

enum FusionKernelSourceType
{
    OpenclText,
    AsmText,
    Binary, /// \todo Unused, consider removing.
};

struct OperatorArgs : miopenOperatorArgs
{
    OperatorArgs();
    void ins_arg(std::string name, OpKernelArg v);
    friend std::ostream& operator<<(std::ostream& stream, const OperatorArgs& x);
    std::vector<OpKernelArg> args_vec;
    std::unordered_map<std::string, OpKernelArg> args_map;
};

struct FusionOpDescriptor : miopenFusionOpDescriptor
{
    virtual ~FusionOpDescriptor()                 = default;
    FusionOpDescriptor(const FusionOpDescriptor&) = delete;
    FusionOpDescriptor()                          = default;
    FusionOpDescriptor& operator=(const FusionOpDescriptor&) = delete;
    void SetIdx(int _id) { plan_idx = _id; };
    int GetIdx() const { return plan_idx; };
    virtual miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) = 0;
    virtual miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle);
    virtual miopenStatus_t GetCompileParms(std::string& compile_config,
                                           Handle& handle,
                                           FusionKernelSourceType source,
                                           const std::vector<solver::AnySolver>& solvers);
    friend std::ostream& operator<<(std::ostream& stream, const FusionOpDescriptor& x);
    virtual miopenFusionOp_t kind() const                                    = 0;
    virtual std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const = 0;
    virtual std::string GetArgKey(const std::string& k) const                = 0;
    virtual OpKernelArg GetOpAttr(const std::string& k) const                = 0;
    virtual bool GetOpAttr(const std::string& /*sym*/, int& /*val*/) const { return false; };
    virtual std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name);
    virtual std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name);
    void SetInputDesc(TensorDescriptor i_desc) { input_desc = i_desc; };
    TensorDescriptor input_desc;

private:
    int plan_idx                       = 0;
    std::shared_ptr<OperatorArgs> args = nullptr;
};

struct BiasFusionOpDescriptor : FusionOpDescriptor
{
    BiasFusionOpDescriptor(const TensorDescriptor& desc) : base_desc(desc){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) override;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle) override;
    miopenStatus_t GetCompileParms(std::string& compile_config,
                                   Handle& handle,
                                   FusionKernelSourceType source,
                                   const std::vector<solver::AnySolver>& solvers) override;
    miopenStatus_t
    SetArgs(OperatorArgs& args, const void* alpha, const void* beta, ConstData_t bdata);
    std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const override;
    std::string GetArgKey(const std::string& k) const override;
    OpKernelArg GetOpAttr(const std::string& k) const override;
    miopenFusionOp_t kind() const override { return miopenFusionOpBiasForward; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name) override;
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name) override;
    TensorDescriptor base_desc;
};

struct ActivFwdFusionOpDescriptor : FusionOpDescriptor
{
    ActivFwdFusionOpDescriptor(miopenActivationMode_t mode) : activMode(mode){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) override;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle) override;
    miopenStatus_t GetCompileParms(std::string& compile_config,
                                   Handle& handle,
                                   FusionKernelSourceType source,
                                   const std::vector<solver::AnySolver>& solvers) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           double activAlpha,
                           double activBeta,
                           double activGamma);
    std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const override;
    std::string GetArgKey(const std::string& k) const override;
    bool GetOpAttr(const std::string& sym, int& val) const override;
    OpKernelArg GetOpAttr(const std::string& k) const override;
    miopenFusionOp_t kind() const override { return miopenFusionOpActivForward; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name) override;
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name) override;
    miopenActivationMode_t activMode;
};

struct ActivBwdFusionOpDescriptor : FusionOpDescriptor
{
    ActivBwdFusionOpDescriptor(miopenActivationMode_t mode) : activMode(mode){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) override;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle) override;
    miopenStatus_t GetCompileParms(std::string& compile_config,
                                   Handle& handle,
                                   FusionKernelSourceType source,
                                   const std::vector<solver::AnySolver>& solvers) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           const void* y,
                           const void* x,
                           double activAlpha,
                           double activBeta,
                           double activGamma);
    std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const override;
    std::string GetArgKey(const std::string& k) const override;
    OpKernelArg GetOpAttr(const std::string& k) const override;
    miopenFusionOp_t kind() const override { return miopenFusionOpActivBackward; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name) override;
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name) override;
    miopenActivationMode_t activMode;
};

struct BatchNormInferenceFusionOpDescriptor : FusionOpDescriptor
{
    BatchNormInferenceFusionOpDescriptor(miopenBatchNormMode_t bn_mode,
                                         const TensorDescriptor& desc)
        : mode(bn_mode), base_desc(desc){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) override;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle) override;
    miopenStatus_t GetCompileParms(std::string& compile_config,
                                   Handle& handle,
                                   FusionKernelSourceType source,
                                   const std::vector<solver::AnySolver>& solvers) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           ConstData_t estimatedMean,
                           ConstData_t estimatedVariance,
                           double epsilon);
    std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const override;
    std::string GetArgKey(const std::string& k) const override;
    OpKernelArg GetOpAttr(const std::string& k) const override;
    bool GetOpAttr(const std::string& sym, int& val) const override;
    miopenFusionOp_t kind() const override { return miopenFusionOpBatchNormInference; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name) override;
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name) override;

    miopenBatchNormMode_t mode;
    TensorDescriptor base_desc;
};

struct BatchNormFwdTrainFusionOpDescriptor : FusionOpDescriptor
{
    BatchNormFwdTrainFusionOpDescriptor(miopenBatchNormMode_t bn_mode, bool runningMeanVariance)
        : mode(bn_mode), runningMeanVar(runningMeanVariance){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) override;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle) override;
    miopenStatus_t GetCompileParms(std::string& compile_config,
                                   Handle& handle,
                                   FusionKernelSourceType source,
                                   const std::vector<solver::AnySolver>& solvers) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           Data_t runningMean,
                           Data_t runningVariance,
                           Data_t savedMean,
                           Data_t savedInvVariance,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           double expAvgFactor,
                           double epsilon);
    std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const override;
    std::string GetArgKey(const std::string& k) const override;
    bool GetOpAttr(const std::string& sym, int& val) const override;
    OpKernelArg GetOpAttr(const std::string& k) const override;
    miopenFusionOp_t kind() const override { return miopenFusionOpBatchNormFwdTrain; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name) override;
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name) override;
    void calcBNParams(Handle& handle,
                      std::vector<size_t> in_lens,
                      int& variant,
                      size_t& in_cstride,
                      size_t& in_nstride,
                      size_t& in_nchw,
                      unsigned int& ldsgcn,
                      unsigned int& ldsnogcn);
    miopenBatchNormMode_t mode;
    TensorDescriptor base_desc;
    bool runningMeanVar;
};

struct BatchNormBwdTrainFusionOpDescriptor : FusionOpDescriptor
{
    BatchNormBwdTrainFusionOpDescriptor(miopenBatchNormMode_t bn_mode)
        : mode(bn_mode), useBatchStats(true){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) override;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle) override;
    miopenStatus_t GetCompileParms(std::string& compile_config,
                                   Handle& handle,
                                   FusionKernelSourceType source,
                                   const std::vector<solver::AnySolver>& solvers) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           ConstData_t x,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           Data_t resBnScaleDiff,
                           Data_t resBnBiasDiff,
                           ConstData_t savedMean,
                           ConstData_t savedInvVariance);
    std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const override;
    std::string GetArgKey(const std::string& k) const override;
    bool GetOpAttr(const std::string& sym, int& val) const override;
    OpKernelArg GetOpAttr(const std::string& k) const override;
    miopenFusionOp_t kind() const override { return miopenFusionOpBatchNormBwdTrain; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name) override;
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name) override;
    void calcBNParams(Handle& handle,
                      std::vector<size_t> in_lens,
                      int& variant,
                      size_t& in_cstride,
                      size_t& in_nstride,
                      size_t& in_nchw,
                      unsigned int& ldsgcn,
                      unsigned int& ldsnogcn);

    miopenBatchNormMode_t mode;
    bool useBatchStats;
};

struct ConvForwardOpDescriptor : FusionOpDescriptor
{
    ConvForwardOpDescriptor(ConvolutionDescriptor& conv_descriptor,
                            TensorDescriptor& filter_descriptor)
        : base_desc(conv_descriptor),
          filter_desc(filter_descriptor),
          kernel_info_valid(false),
          conv_compiler_options(""){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) override;
    miopenStatus_t SetArgs(OperatorArgs& args, const void* alpha, const void* beta, ConstData_t w);
    std::vector<std::pair<std::string, OpKernelArg>> GetArgs() const override;
    std::string GetArgKey(const std::string& k) const override;
    OpKernelArg GetOpAttr(const std::string& k) const override;
    bool GetOpAttr(const std::string& sym, int& val) const override;
    miopenStatus_t GetNetworkConfig(std::string& network_config, Handle& handle) override;
    miopenStatus_t GetCompileParms(std::string& compile_config,
                                   Handle& handle,
                                   FusionKernelSourceType source,
                                   const std::vector<solver::AnySolver>& solvers) override;
    bool isASMApplicable(Handle& handle);
    miopenFusionOp_t kind() const override { return miopenFusionOpConvForward; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name) override;
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name) override;

    ConvolutionDescriptor base_desc;
    TensorDescriptor filter_desc;
    solver::KernelInfo kernel_info;
    bool kernel_info_valid;
    std::string conv_compiler_options;

private:
    mlo_construct_direct2D_fusion ConstructParams(Handle& handle);
};

namespace fusion {

bool IsWinograd(const std::vector<solver::AnySolver>& ss);

} // namespace fusion

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenFusionOpDescriptor, miopen::FusionOpDescriptor);
MIOPEN_DEFINE_OBJECT(miopenOperatorArgs, miopen::OperatorArgs);

#endif // _MIOPEN_FUSION_HPP_
