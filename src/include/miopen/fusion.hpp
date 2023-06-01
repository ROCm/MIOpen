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
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/activ.hpp>

#include <set>
#include <vector>
#include <unordered_map>

namespace miopen {

struct Handle;

// Perhaps redundant
enum FusionKernelSourceType
{
    OpenclText,
    AsmText,
    Binary, /// \todo Unused, consider removing.
};

struct FusionOpDescriptor : miopenFusionOpDescriptor
{
    virtual ~FusionOpDescriptor()                 = default;
    FusionOpDescriptor(const FusionOpDescriptor&) = delete;
    FusionOpDescriptor()                          = default;
    FusionOpDescriptor& operator=(const FusionOpDescriptor&) = delete;
    void SetIdx(int _id) { plan_idx = _id; };
    int GetIdx() const { return plan_idx; };
    virtual miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const = 0;
    virtual miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle);
    friend std::ostream& operator<<(std::ostream& stream, const FusionOpDescriptor& x);
    virtual miopenFusionOp_t kind() const = 0;
    void SetInputDesc(TensorDescriptor i_desc) { input_desc = i_desc; };
    TensorDescriptor input_desc;

    int plan_idx = 0;
};

struct BiasFusionOpDescriptor : FusionOpDescriptor
{
    BiasFusionOpDescriptor(const TensorDescriptor& desc) : base_desc(desc) {}
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const override;
    miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle) override;
    miopenStatus_t
    SetArgs(OperatorArgs& args, const void* alpha, const void* beta, ConstData_t bdata);
    miopenFusionOp_t kind() const override { return miopenFusionOpBiasForward; };
    TensorDescriptor base_desc;
};

struct ActivFwdFusionOpDescriptor : FusionOpDescriptor
{
    ActivFwdFusionOpDescriptor(miopenActivationMode_t mode) : activMode(mode) {}
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const override;
    miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           double activAlpha,
                           double activBeta,
                           double activGamma);
    miopenFusionOp_t kind() const override { return miopenFusionOpActivForward; };
    miopenActivationMode_t activMode;
};

struct ActivBwdFusionOpDescriptor : FusionOpDescriptor
{
    ActivBwdFusionOpDescriptor(miopenActivationMode_t mode) : activMode(mode) {}
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const override;
    miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           ConstData_t y,
                           ConstData_t x,
                           double activAlpha,
                           double activBeta,
                           double activGamma);
    miopenFusionOp_t kind() const override { return miopenFusionOpActivBackward; };
    miopenActivationMode_t activMode;
};

struct BatchNormInferenceFusionOpDescriptor : FusionOpDescriptor
{
    BatchNormInferenceFusionOpDescriptor(miopenBatchNormMode_t bn_mode,
                                         const TensorDescriptor& desc)
        : mode(bn_mode), base_desc(desc)
    {
    }
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const override;
    miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle) override;
    miopenStatus_t SetArgs(OperatorArgs& args,
                           const void* alpha,
                           const void* beta,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           ConstData_t estimatedMean,
                           ConstData_t estimatedVariance,
                           double epsilon);
    miopenFusionOp_t kind() const override { return miopenFusionOpBatchNormInference; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name);
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name);

    miopenBatchNormMode_t mode;
    TensorDescriptor base_desc;
};

struct BatchNormFwdTrainFusionOpDescriptor : FusionOpDescriptor
{
    BatchNormFwdTrainFusionOpDescriptor(miopenBatchNormMode_t bn_mode, bool runningMeanVariance)
        : mode(bn_mode), runningMeanVar(runningMeanVariance)
    {
    }
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const override;
    miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle) override;
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
    miopenFusionOp_t kind() const override { return miopenFusionOpBatchNormFwdTrain; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name);
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name);
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
        : mode(bn_mode), useBatchStats(true)
    {
    }
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const override;
    miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle) override;
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
    miopenFusionOp_t kind() const override { return miopenFusionOpBatchNormBwdTrain; };
    std::vector<size_t> GetLocalWGSz(Handle& handle, std::string algorithm_name);
    std::vector<size_t> GetGlobalWGSz(Handle& handle, std::string algorithm_name);
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
    ConvForwardOpDescriptor(const ConvolutionDescriptor& conv_descriptor,
                            const TensorDescriptor& filter_descriptor)
        : base_desc(conv_descriptor),
          filter_desc(filter_descriptor),
          kernel_info_valid(false),
          conv_compiler_options(""){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) const override;
    miopenStatus_t SetArgs(OperatorArgs& args, const void* alpha, const void* beta, ConstData_t w);
    miopenStatus_t GetNetworkConfig(std::stringstream& network_config, Handle& handle) override;
    bool isASMApplicable(Handle& handle);
    miopenFusionOp_t kind() const override { return miopenFusionOpConvForward; };

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
miopenStatus_t ConvBiasActivFusion(Handle& handle,
                                   const void* alpha1,
                                   const TensorDescriptor& xDesc,
                                   ConstData_t x,
                                   const TensorDescriptor& wDesc,
                                   ConstData_t w,
                                   const ConvolutionDescriptor& conv_desc,
                                   miopenConvFwdAlgorithm_t algo,
                                   void* workspace,
                                   size_t workspaceSizeInBytes,
                                   const void* alpha2,
                                   const TensorDescriptor& zDesc,
                                   ConstData_t z,
                                   const TensorDescriptor& biasDesc,
                                   ConstData_t bias,
                                   const ActivationDescriptor& activationDesc,
                                   const TensorDescriptor& yDesc,
                                   Data_t y);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenFusionOpDescriptor, miopen::FusionOpDescriptor);
MIOPEN_DEFINE_OBJECT(miopenOperatorArgs, miopen::OperatorArgs);

#endif // _MIOPEN_FUSION_HPP_
