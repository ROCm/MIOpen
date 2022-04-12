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
#include <cassert>
#include <miopen/fusion.hpp>
#include <miopen/md_graph.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/stringutils.hpp>
#include <ostream>
#include <ios>
#include <algorithm>
#include <string>
#include <half.hpp>

namespace miopen {

FusionPlanDescriptor::FusionPlanDescriptor(const miopenFusionDirection_t dir,
                                           const TensorDescriptor& inDesc)
    : fusion_dir(dir),
      input_desc(inDesc),
      is_valid(false),
      kernel_source_type(OpenclText),
      fp_contains_bn(false),
      program_name(""),
      kernel_name(""),
      algorithm_name(""),
      network_config(inDesc.ToString()),
      data_type(inDesc.GetType())
{
}

FusionPlanDescriptor::~FusionPlanDescriptor() { op_map.clear(); }

miopenStatus_t FusionPlanDescriptor::AddOp(std::shared_ptr<FusionOpDescriptor> desc)
{
    // load the md graph for the first op
    if(op_count == 0)
    {
        FusionMDGraph::Init(lu, desc->kind());
    }
    desc->SetIdx(op_count);
    if(op_map.empty())
        desc->SetInputDesc(input_desc);
    else
        desc->SetInputDesc(output_desc);
    desc->GetOutputDesc(output_desc);
    op_map.emplace_back(desc);
    op_count++;
    is_valid = false;
    miopen::try_([&] {
        is_valid = lu.Advance(desc, [&](const std::string& sym, int& val) -> bool {
            // check tensor attr
            if(GetTensorAttr(sym, val))
                return true;
            // check op attr
            if(desc->GetOpAttr(sym, val))
                return true;
            // check the values of enum types
            if(GetEnumVal(sym, val))
                return true;
            // check dev attr
            // if(GetDevAttribute(sym, val, handle))
            //     return true;
            return false;
        });
    });
    if(is_valid)
        return miopenStatusSuccess;
    else
        return miopenStatusUnsupportedOp;
}

miopenStatus_t FusionPlanDescriptor::GetOp(int op_idx, std::shared_ptr<FusionOpDescriptor>& desc)
{
    auto err = miopenStatusSuccess;

    if(op_idx >= op_map.size())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Operator index out of bounds");
    }

    desc = op_map.at(op_idx);
    return err;
}

TensorDescriptor FusionPlanDescriptor::FusionPlanDescriptor::DeriveOutputDescriptor()
{
    TensorDescriptor i_desc = input_desc;
    TensorDescriptor o_desc;
    if(fusion_dir == miopenVerticalFusion)
    {
        // All the ops should have the same output descriptor otherwise
        // fusion would not be feasible, thus we need to call GetOutputDesc on all
        // the ops and make sure it returns the same value
        for(auto&& op : op_map)
        {
            op->SetInputDesc(i_desc);
            op->GetOutputDesc(o_desc);
            i_desc = o_desc;
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported fusion direction");
    }
    return o_desc;
}

miopenStatus_t FusionPlanDescriptor::GetWorkspaceSizeImmed(Handle& handle,
                                                           size_t& workSpaceSize,
                                                           miopenConvFwdAlgorithm_t /*algo*/)
{
    workSpaceSize = 0;
    for(auto&& op : op_map)
    {
        if(op->kind() == miopenFusionOpConvForward)
        {
            auto ptr = std::dynamic_pointer_cast<ConvForwardOpDescriptor>(op);
            TensorDescriptor opd;
            ptr->GetOutputDesc(opd);
            size_t tmp_sz = ptr->base_desc.ForwardGetWorkSpaceSize(
                handle, ptr->filter_desc, ptr->input_desc, opd);
            if(tmp_sz > workSpaceSize)
                workSpaceSize = tmp_sz;
        }
    }
    return miopenStatusSuccess;
}

miopenStatus_t FusionPlanDescriptor::GetConvAlgos(int reqAlgoCount,
                                                  int& retAlgoCount,
                                                  miopenConvFwdAlgorithm_t* ptrAlgos)
{
    auto algos   = lu.GetConvAlgos();
    retAlgoCount = std::min(reqAlgoCount, static_cast<int>(algos.size()));

    for(auto idx = 0; idx < retAlgoCount; idx++)
    {
        ptrAlgos[idx] = algos[idx];
    }

    return miopenStatusSuccess;
}

miopenStatus_t FusionPlanDescriptor::SetConvAlgo(miopenConvFwdAlgorithm_t algo)
{
    bool res = lu.SetConvAlgo(algo);

    if(res)
        return miopenStatusSuccess;
    else
        return miopenStatusUnknownError;
}

std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& fpd)
{
    stream << "kernel_name: " << fpd.kernel_name;
    return stream;
}

// Fusion operator descriptors
// Conv Forward
miopenStatus_t ConvForwardOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    return miopen::try_(
        [&]() { output_desc = base_desc.GetForwardOutputTensor(input_desc, filter_desc); });
}

miopenStatus_t ConvForwardOpDescriptor::SetArgs(OperatorArgs& args,
                                                const void* /*alpha*/,
                                                const void* /*beta*/,
                                                ConstData_t w)
{
    auto w_any = OpKernelArg(w);
    args.ins_arg("weights" + std::to_string(GetIdx()), w_any);

    return miopenStatusSuccess;
}

std::vector<std::pair<std::string, OpKernelArg>> ConvForwardOpDescriptor::GetArgs() const
{
    ConstData_t w = nullptr;
    std::vector<std::pair<std::string, OpKernelArg>> keys;
    keys.emplace_back("weights" + std::to_string(GetIdx()), OpKernelArg(w));
    return keys;
}

std::string ConvForwardOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

bool ConvForwardOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    int o, c, x, y;
    std::tie(o, c, x, y) = tien<4>(filter_desc.GetLengths());

    auto f_strides     = filter_desc.GetStrides();
    const int f_t_size = miopen::GetTypeSize(input_desc.GetType());
    std::transform(f_strides.begin(),
                   f_strides.end(),
                   f_strides.begin(),
                   [&f_t_size](const auto& s) { return s * f_t_size; });

    if(sym == "x")
    {
        val = x;
    }
    else if(sym == "y")
    {
        val = y;
    }
    else if(sym == "c")
    {
        val = c;
    }
    else if(sym == "pad_h")
    {
        val = base_desc.GetConvPads()[0];
    }
    else if(sym == "pad_w")
    {
        val = base_desc.GetConvPads()[1];
    }
    else if(sym == "dilation_h")
    {
        val = base_desc.GetConvDilations()[0];
    }
    else if(sym == "dilation_w")
    {
        val = base_desc.GetConvDilations()[1];
    }
    else if(sym == "stride_h")
    {
        val = base_desc.GetConvStrides()[0];
    }
    else if(sym == "stride_w")
    {
        val = base_desc.GetConvStrides()[1];
    }
    else if(sym == "k")
    {
        val = o;
    }
    else if(sym == "group_count")
    {
        val = base_desc.GetGroupCount();
    }
    else if(sym == "f_byte_stride_nk")
    {
        val = f_strides[0];
    }
    else if(sym == "f_byte_stride_c")
    {
        val = f_strides[1];
    }
    else if(sym == "f_byte_stride_h")
    {
        val = f_strides[2];
    }
    else if(sym == "f_byte_stride_w")
    {
        val = f_strides[3];
    }
    else
        return false;

    return true;
}

OpKernelArg ConvForwardOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown Convolution Op Attribute");
    }
}
// Activ Forward ------------------------------------

miopenStatus_t ActivFwdFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                   const void* /*alpha*/,
                                                   const void* /*beta*/,
                                                   double activAlpha,
                                                   double activBeta,
                                                   double activGamma)
{
    auto id = std::to_string(GetIdx());
    if(input_desc.GetType() == miopenFloat)
    {
        args.ins_arg("activAlpha" + id, OpKernelArg(static_cast<float>(activAlpha)));
        args.ins_arg("activBeta" + id, OpKernelArg(static_cast<float>(activBeta)));
        args.ins_arg("activGamma" + id, OpKernelArg(static_cast<float>(activGamma)));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        args.ins_arg("activAlpha" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activAlpha))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activBeta" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activBeta))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activGamma" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activGamma))); // NOLINT (cppcoreguidelines-narrowing-conversions)
    }
    return miopenStatusSuccess;
}

std::string ActivFwdFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

std::vector<std::pair<std::string, OpKernelArg>> ActivFwdFusionOpDescriptor::GetArgs() const
{
    std::vector<std::pair<std::string, OpKernelArg>> keys;
    auto id = std::to_string(GetIdx());
    if(input_desc.GetType() == miopenFloat)
    {
        float a = 0.0;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        half_float::half a;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }

    return keys;
}

bool ActivFwdFusionOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    if(sym == "activ_mode")
    {
        val = activMode;
        return true;
    }
    return false;
}

OpKernelArg ActivFwdFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    MIOPEN_THROW(miopenStatusInternalError, "Unknown Activation Op Attribute");
}

miopenStatus_t ActivFwdFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}
// Activ Backwards-----------------------------------------
miopenStatus_t ActivBwdFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                   const void* /*alpha*/,
                                                   const void* /*beta*/,
                                                   const void* y,
                                                   const void* x,
                                                   double activAlpha,
                                                   double activBeta,
                                                   double activGamma)
{
    auto id             = std::to_string(GetIdx());
    auto activDiffScale = activBeta * activGamma;
    if(input_desc.GetType() == miopenFloat)
    {
        args.ins_arg("activAlpha" + id, OpKernelArg(static_cast<float>(activAlpha)));
        args.ins_arg("activBeta" + id, OpKernelArg(static_cast<float>(activBeta)));
        args.ins_arg("activGamma" + id, OpKernelArg(static_cast<float>(activGamma)));
        args.ins_arg("activDiffScale" + id, OpKernelArg(static_cast<float>(activDiffScale)));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        args.ins_arg("activAlpha" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activAlpha))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activBeta" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activBeta))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activGamma" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activGamma))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activDiffScale" + id,
                     OpKernelArg(static_cast<half_float::half>(activDiffScale)));
    }

    auto y_any = OpKernelArg(y);
    auto x_any = OpKernelArg(x);
    args.ins_arg("y" + id, y_any);
    args.ins_arg("x" + id, x_any);
    return miopenStatusSuccess;
}

std::string ActivBwdFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

OpKernelArg ActivBwdFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    MIOPEN_THROW("ActivBwdFusionOpDescriptor op does not support attribute: " + k);
}

std::vector<std::pair<std::string, OpKernelArg>> ActivBwdFusionOpDescriptor::GetArgs() const
{
    std::vector<std::pair<std::string, OpKernelArg>> keys;
    auto id = std::to_string(GetIdx());
    if(input_desc.GetType() == miopenFloat)
    {
        float a = 0.0;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        half_float::half a;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }
    keys.emplace_back("activDiffScale" + id, OpKernelArg(nullptr));
    keys.emplace_back("y" + id, OpKernelArg(nullptr));
    keys.emplace_back("x" + id, OpKernelArg(nullptr));
    return keys;
}

miopenStatus_t ActivBwdFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}
//==============================

miopenStatus_t BatchNormInferenceFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                             const void*,
                                                             const void*,
                                                             ConstData_t bnScale,
                                                             ConstData_t bnBias,
                                                             ConstData_t estimatedMean,
                                                             ConstData_t estimatedVariance,
                                                             double epsilon)
{
    auto id                    = std::to_string(GetIdx());
    auto bnScale_any           = OpKernelArg(bnScale);
    auto bnBias_any            = OpKernelArg(bnBias);
    auto estimatedMean_any     = OpKernelArg(estimatedMean);
    auto estimatedVariance_any = OpKernelArg(estimatedVariance);
    auto epsilon_any           = OpKernelArg(static_cast<double>(epsilon));
    args.ins_arg("epsilon" + id, epsilon_any);
    args.ins_arg("bnScale" + id, bnScale_any);
    args.ins_arg("bnBias" + id, bnBias_any);
    args.ins_arg("estimatedMean" + id, estimatedMean_any);
    args.ins_arg("estimatedVariance" + id, estimatedVariance_any);
    return miopenStatusSuccess;
}

std::string BatchNormInferenceFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

std::vector<std::pair<std::string, OpKernelArg>>
BatchNormInferenceFusionOpDescriptor::GetArgs() const
{
    std::vector<std::pair<std::string, OpKernelArg>> keys;
    auto id        = std::to_string(GetIdx());
    double epsilon = 0.0;
    keys.emplace_back("epsilon" + id, OpKernelArg(epsilon));
    ConstData_t bnScale = nullptr;
    keys.emplace_back("bnScale" + id, OpKernelArg(bnScale));
    ConstData_t bnBias = nullptr;
    keys.emplace_back("bnBias" + id, OpKernelArg(bnBias));
    ConstData_t estimatedMean = nullptr;
    keys.emplace_back("estimatedMean" + id, OpKernelArg(estimatedMean));
    ConstData_t estimatedVariance = nullptr;
    keys.emplace_back("estimatedVariance" + id, OpKernelArg(estimatedVariance));
    return keys;
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

OpKernelArg BatchNormInferenceFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown Activation Op Attribute");
    }
}
bool BatchNormInferenceFusionOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    if(sym == "bn_mode")
    {
        val = mode;
        return true;
    }
    else
    {
        return false;
    }
}

// Batch Normalization Forward Training --------------
miopenStatus_t BatchNormFwdTrainFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                            const void* /*alpha*/,
                                                            const void* /*beta*/,
                                                            Data_t runningMean,
                                                            Data_t runningVariance,
                                                            Data_t savedMean,
                                                            Data_t savedInvVariance,
                                                            ConstData_t bnScale,
                                                            ConstData_t bnBias,
                                                            double expAvgFactor,
                                                            double epsilon)
{

    // @todo add in saved versus running boolean toggles
    auto id                   = std::to_string(GetIdx());
    auto bnScale_any          = OpKernelArg(bnScale);
    auto bnBias_any           = OpKernelArg(bnBias);
    auto runningMean_any      = OpKernelArg(runningMean);
    auto runningVariance_any  = OpKernelArg(runningVariance);
    auto savedMean_any        = OpKernelArg(savedMean);
    auto savedInvVariance_any = OpKernelArg(savedInvVariance);
    auto expAvgFactor_any     = OpKernelArg(static_cast<double>(expAvgFactor));
    auto epsilon_any          = OpKernelArg(static_cast<double>(epsilon));
    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    auto nhw             = static_cast<float>(n * h * w);

    auto inhw_any = static_cast<float>(1.0f / nhw);

    if(runningMeanVar && (runningMean == nullptr || runningVariance == nullptr))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Save batch statistics was turned on at op creation time "
                     "but runningMean or runningVariance is set to nullptr");
    }

    args.ins_arg("inhw" + id, inhw_any);
    args.ins_arg("expAvgFactor" + id, expAvgFactor_any);
    args.ins_arg("epsilon" + id, epsilon_any);
    args.ins_arg("bnScale" + id, bnScale_any);
    args.ins_arg("bnBias" + id, bnBias_any);
    args.ins_arg("savedMean" + id, savedMean_any);
    args.ins_arg("savedInvVariance" + id, savedInvVariance_any);
    args.ins_arg("runningMean" + id, runningMean_any);
    args.ins_arg("runningVariance" + id, runningVariance_any);
    return miopenStatusSuccess;
}

std::vector<std::pair<std::string, OpKernelArg>>
BatchNormFwdTrainFusionOpDescriptor::GetArgs() const
{

    // @todo add in saved versus running boolean toggles
    std::vector<std::pair<std::string, OpKernelArg>> keys;
    auto id        = std::to_string(GetIdx());
    Data_t d       = nullptr;
    ConstData_t cd = nullptr;
    auto f_any     = OpKernelArg(static_cast<float>(0.0f));
    auto d_any     = OpKernelArg(d);
    auto cd_any    = OpKernelArg(cd);

    if(mode == miopenBNSpatial)
    {
        keys.emplace_back("inhw" + id, f_any);
    }

    keys.emplace_back("epsilon" + id, OpKernelArg(static_cast<double>(0)));
    keys.emplace_back("bnScale" + id, cd_any);
    keys.emplace_back("bnBias" + id, cd_any);
    keys.emplace_back("savedMean" + id, d_any);
    keys.emplace_back("savedInvVariance" + id, d_any);
    keys.emplace_back("expAvgFactor" + id, OpKernelArg(static_cast<double>(0)));
    keys.emplace_back("runningMean" + id, d_any);
    keys.emplace_back("runningVariance" + id, d_any);
    return keys;
}

miopenStatus_t BatchNormFwdTrainFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

// end BN forward training -----------------------------

// Batch Normalization Backward Training --------------
std::string BatchNormBwdTrainFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}
bool BatchNormBwdTrainFusionOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    if(sym == "bn_mode")
    {
        val = mode;
        return true;
    }
    else
    {
        return false;
    }
}
OpKernelArg BatchNormBwdTrainFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    else if(k == "diff_scale")
    {
        return {static_cast<float>(0.0)};
    }
    else if(k == "iNHW")
    {
        int n, h, w;
        std::tie(n, std::ignore, h, w) = tien<4>(input_desc.GetLengths());
        auto nhw                       = static_cast<float>(n * h * w);
        return {static_cast<float>(1.0f / nhw)};
    }
    else
        MIOPEN_THROW("BatchNormBwdTrainFusionOpDescriptor does not support attribute: " + k);
}
miopenStatus_t BatchNormBwdTrainFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                            const void* /*alpha*/,
                                                            const void* /*beta*/,
                                                            ConstData_t x,
                                                            ConstData_t bnScale,
                                                            ConstData_t bnBias,
                                                            Data_t resBnScaleDiff,
                                                            Data_t resBnBiasDiff,
                                                            ConstData_t savedMean,
                                                            ConstData_t savedInvVariance)
{

    // @todo add in saved boolean toggle
    auto id                   = std::to_string(GetIdx());
    auto x_any                = OpKernelArg(x);
    auto bnScale_any          = OpKernelArg(bnScale);
    auto bnBias_any           = OpKernelArg(bnBias);
    auto resBnScaleDiff_any   = OpKernelArg(resBnScaleDiff);
    auto resBnBiasDiff_any    = OpKernelArg(resBnBiasDiff);
    auto savedMean_any        = OpKernelArg(savedMean);
    auto savedInvVariance_any = OpKernelArg(savedInvVariance);

    args.ins_arg("x" + id, x_any);
    args.ins_arg("bnScale" + id, bnScale_any);
    args.ins_arg("bnBias" + id, bnBias_any);
    args.ins_arg("resBnScaleDiff" + id, resBnScaleDiff_any);
    args.ins_arg("resBnBiasDiff" + id, resBnBiasDiff_any);
    args.ins_arg("savedMean" + id, savedMean_any);
    args.ins_arg("savedInvVariance" + id, savedInvVariance_any);
    return miopenStatusSuccess;
}

std::vector<std::pair<std::string, OpKernelArg>>
BatchNormBwdTrainFusionOpDescriptor::GetArgs() const
{
    std::vector<std::pair<std::string, OpKernelArg>> keys;
    auto id        = std::to_string(GetIdx());
    Data_t d       = nullptr;
    ConstData_t cd = nullptr;
    auto d_any     = OpKernelArg(d);
    auto cd_any    = OpKernelArg(cd);

    keys.emplace_back("x" + id, cd_any);
    keys.emplace_back("bnScale" + id, cd_any);
    keys.emplace_back("bnBias" + id, cd_any);
    keys.emplace_back("resBnScaleDiff" + id, d_any);
    keys.emplace_back("resBnBiasDiff" + id, d_any);
    keys.emplace_back("savedMean" + id, cd_any);
    keys.emplace_back("savedInvVariance" + id, cd_any);
    return keys;
}

miopenStatus_t BatchNormBwdTrainFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

// end BN backwards training ---------------------------

// Bias forward
miopenStatus_t BiasFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

miopenStatus_t BiasFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                               const void* /*alpha*/,
                                               const void* /*beta*/,
                                               ConstData_t bdata)
{
    auto bdata_any = OpKernelArg(bdata);
    args.ins_arg("bias" + std::to_string(GetIdx()), bdata_any);
    return miopenStatusSuccess;
}

std::string BiasFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

OpKernelArg BiasFusionOpDescriptor::GetOpAttr(const std::string& /* k */) const
{
    MIOPEN_THROW(miopenStatusInternalError, "Unknown Bias Op Attribute");
}

std::vector<std::pair<std::string, OpKernelArg>> BiasFusionOpDescriptor::GetArgs() const
{
    ConstData_t bdata = nullptr;
    std::vector<std::pair<std::string, OpKernelArg>> keys;
    keys.emplace_back("bias" + std::to_string(GetIdx()), OpKernelArg(bdata));
    return keys;
}

static inline void
find_replace_first(std::string& s_where, const std::string& s_find, const std::string& s_replace)
{
    const auto pos = s_where.find(s_find);
    if(pos != std::string::npos)
        s_where.replace(pos, s_find.length(), s_replace);
}

std::string FusionPlanDescriptor::GetProgramName(const Handle& handle)
{
    if(!op_map.empty())
    {
        program_name = lu.GetProgramName(handle);
        // Replace "GFX*" wildcard by device name (in lowercase)
        auto d = handle.GetDeviceName();
        std::transform(d.begin(), d.end(), d.begin(), ::tolower);
        find_replace_first(program_name, "GFX*", d);
        return program_name;
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported starting op in Fusion Plan");
    }
}

std::string FusionPlanDescriptor::GetKernelName(const Handle& handle)
{
    if(!op_map.empty())
    {
        kernel_name = lu.GetKernelName(handle);
        return kernel_name;
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported starting op in Fusion Plan");
    }
}

std::string FusionPlanDescriptor::GetAlgorithmName(const Handle& handle)
{
    if(!op_map.empty())
    {
        algorithm_name = lu.GetAlgoName(handle);
        return algorithm_name;
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported starting op in Fusion Plan");
    }
}

bool FusionPlanDescriptor::GetEnumVal(const std::string& sym, int& val) const
{
    if(sym == "miopenFloat")
    {
        val = miopenFloat;
        return true;
    }
    else if(sym == "miopenConvolutionFwdAlgoDirect")
    {
        val = miopenConvolutionFwdAlgoDirect;
        return true;
    }
    else if(sym == "miopenConvolutionFwdAlgoWinograd")
    {
        val = miopenConvolutionFwdAlgoWinograd;
        return true;
    }
    else if(sym == "miopenBNPerActivation")
    {
        val = miopenBNPerActivation;
        return true;
    }
    else if(sym == "miopenBNSpatial")
    {
        val = miopenBNSpatial;
        return true;
    }
    else if(sym == "miopenActivationRELU")
    {
        val = miopenActivationRELU;
        return true;
    }
    else if(sym == "miopenActivationLEAKYRELU")
    {
        val = miopenActivationLEAKYRELU;
        return true;
    }
    return false;
}

bool FusionPlanDescriptor::GetTensorAttr(const std::string& sym, int& val) const
{
    int N, C, H, W, oN, K, oH, oW;
    std::tie(N, C, H, W)    = miopen::tien<4>(input_desc.GetLengths(), 1);
    std::tie(oN, K, oH, oW) = miopen::tien<4>(output_desc.GetLengths(), 1);

    const int d_t_size = miopen::GetTypeSize(input_desc.GetType());
    const int o_t_size = miopen::GetTypeSize(output_desc.GetType());
    auto d_strides     = input_desc.GetStrides();
    auto o_strides     = output_desc.GetStrides();
    std::transform(d_strides.begin(),
                   d_strides.end(),
                   d_strides.begin(),
                   [&d_t_size](const auto& s) { return s * d_t_size; });
    std::transform(o_strides.begin(),
                   o_strides.end(),
                   o_strides.begin(),
                   [&o_t_size](const auto& s) { return s * o_t_size; });

    if(sym == "iN")
    {
        val = N;
    }
    else if(sym == "iC")
    {
        val = C;
    }
    else if(sym == "iH")
    {
        val = H;
    }
    else if(sym == "iW")
    {
        val = W;
    }
    else if(sym == "oN")
    {
        val = oN;
    }
    else if(sym == "oK")
    {
        val = K;
    }
    else if(sym == "oH")
    {
        val = oH;
    }
    else if(sym == "oW")
    {
        val = oW;
    }
    else if(sym == "d_byte_stride_nk")
    {
        val = d_strides[0];
    }
    else if(sym == "d_byte_stride_c")
    {
        val = d_strides[1];
    }
    else if(sym == "d_byte_stride_h")
    {
        val = d_strides[2];
    }
    else if(sym == "d_byte_stride_w")
    {
        val = d_strides[3];
    }
    else if(sym == "o_byte_stride_nk")
    {
        val = o_strides[0];
    }
    else if(sym == "o_byte_stride_c")
    {
        val = o_strides[1];
    }
    else if(sym == "o_byte_stride_h")
    {
        val = o_strides[2];
    }
    else if(sym == "o_byte_stride_w")
    {
        val = o_strides[3];
    }
    else if(sym == "precision")
    {
        assert(input_desc.GetType() == output_desc.GetType());
        val = input_desc.GetType();
    }
    else
        return false;

    return true;
}

OpKernelArg FusionPlanDescriptor::GetTensorAttr(const std::string& sym) const
{
    int val;
    if(FusionPlanDescriptor::GetTensorAttr(sym, val))
        return {val};
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown Tensor Attribute: " + sym);
    }
}
OpKernelArg FusionPlanDescriptor::GetDevAttribute(const std::string& k, const Handle& handle) const
{
    if(k == "devCUs")
    {
        int num_cus = handle.GetMaxComputeUnits();
        return {num_cus};
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown device attribute " + k);
    }
}

miopenStatus_t FusionPlanDescriptor::Compile(Handle& handle)
{
    miopenStatus_t status = miopenStatusUnknownError;
    if(!isValid() || (lu.GetCurVertex(handle) == nullptr))
    {
        MIOPEN_LOG_I2(
            "A previous attempt to add an operator unsuccessful or the GPU architecture is not "
            "supported for the fusion plan");
        MIOPEN_THROW(miopenStatusBadParm);
    }
    network_config =
        input_desc.ToString() + ((input_desc.GetType() == miopenHalf) ? "FP16" : "FP32");
    network_config +=
        output_desc.ToString() + ((input_desc.GetType() == miopenHalf) ? "FP16" : "FP32");

    for(auto&& op : op_map)
    {
        op->GetNetworkConfig(network_config, handle);
    }
    // Check if the kernel is assembly or OpenCL
    auto ops_head  = op_map[0];
    algorithm_name = lu.GetAlgoName(handle);
    program_name   = GetProgramName(handle);
    kernel_name    = GetKernelName(handle);
    MIOPEN_LOG_I2(program_name << ',' << kernel_name);

    if(program_name.empty())
    {

        MIOPEN_LOG_I2("Trying to compile an invalid FusionPlan");
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(miopen::EndsWith(program_name, ".s"))
        kernel_source_type = AsmText;
    else if(miopen::EndsWith(program_name, ".so"))
        kernel_source_type = Binary;
    else
        kernel_source_type = OpenclText;

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);
    if(!kernels.empty())
    {
        status = miopenStatusSuccess;
    }
    else
    {
        MIOPEN_LOG_I2("Precompiled kernel does not exist, compiling fused-kernel");
        std::string compile_config;
        auto success = true;
        // lu.cur_vertex is sorted according to the weights from MDGraph::Advance method
        std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> new_list;
        for(auto& kinder : lu.cur_vertex)
        {
            if(kinder.first == nullptr)
            {
                MIOPEN_LOG_I2("Invalid FusionPlan");
                MIOPEN_THROW(miopenStatusBadParm);
            }

            success = true;
            solver::AnySolver sol;
            if(kinder.second.find("solver") != kinder.second.end())
            {
                sol = boost::any_cast<solver::AnySolver>(kinder.second.at("solver"));
            }
            program_name = kinder.first->vertex_data.at("program");
            auto d       = handle.GetDeviceName();

            auto it = std::find(
                kinder.first->supported_arch.begin(), kinder.first->supported_arch.end(), d);
            // Empty inidicates any arch is supported (say OpenCL kernels)
            if(!kinder.first->supported_arch.empty() && (it == kinder.first->supported_arch.end()))
                continue;

            const auto target = handle.GetTargetProperties();
            if(kinder.first->supported_xnack && target.Xnack() &&
               *kinder.first->supported_xnack != *target.Xnack())
                continue;

            std::transform(d.begin(), d.end(), d.begin(), ::tolower);
            find_replace_first(program_name, "GFX*", d);

            kernel_name    = kinder.first->vertex_data.at("kernel");
            algorithm_name = kinder.first->vertex_data.at("algorithm");
            if(miopen::EndsWith(program_name, ".s"))
                kernel_source_type = AsmText;
            else if(miopen::EndsWith(program_name, ".so"))
                kernel_source_type = Binary;
            else
                kernel_source_type = OpenclText;
            // MIOPEN_LOG_I2("Trying solver: " << *sol);
            std::vector<solver::AnySolver> sol_vec = {sol};
            for(auto&& op : op_map)
            {
                MIOPEN_LOG_I2("GetCompileParms, " << *op);
                if(op->GetCompileParms(compile_config, handle, kernel_source_type, sol_vec) ==
                   miopenStatusSuccess)
                    continue;
                else
                {
                    success = false;
                    break;
                }
            }
            if(success)
            {
                new_list.emplace_back(kinder.first, kinder.second);
                break;
            }
        }
        if(success)
        {
            lu.cur_vertex   = new_list;
            auto&& kernels2 = handle.GetKernels(algorithm_name, network_config);
            if(!kernels2.empty())
            {
                status = miopenStatusSuccess;
            }
            else
            {
                auto dType = input_desc.GetType();
                if(kernel_source_type == OpenclText)
                {
                    if(dType == miopenFloat)
                    {
                        compile_config += " -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1";
                    }
                    else
                    {
                        compile_config += " -DMIOPEN_USE_FP16=1 -DMIOPEN_USE_FP32=0";
                    }
                }
                // TODO: This true for inference but might not be true in general
                // This is sill an open question
                // Must be preceded by GetCompileParms
                const auto& vld = ops_head->GetLocalWGSz(handle, algorithm_name);
                const auto& vgd = ops_head->GetGlobalWGSz(handle, algorithm_name);
                MIOPEN_LOG_I2("Program: " << program_name << ", kernel: " << kernel_name);
                MIOPEN_LOG_I2("Build options: " << compile_config);
                handle.AddKernel(algorithm_name,
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 compile_config);

                status = miopenStatusSuccess;
            }
        }
        else
        {
            MIOPEN_LOG_I("No viable kernel found to execute the fusion plan");
            status = miopenStatusInternalError;
            return status;
        }
    }
    arg_list = CalcArgOrder(handle);
    return status;
}

std::vector<Exec_arg_t> FusionPlanDescriptor::CalcArgOrder(const Handle& handle)
{
    std::vector<Exec_arg_t> arg_keys;
    // Construct the kernel args
    std::set<size_t> arg_sizes; // a set of argument sizes
    // A map between argument sizes and argument names
    std::map<std::pair<size_t, size_t>, std::vector<std::string>> size_map;
    // A map between argument pointers (buffers) and argument names
    std::map<size_t, std::vector<std::string>> ptr_map;

    for(size_t idx = 0; idx < op_map.size(); idx++)
    {
        auto op   = op_map.at(idx);
        auto keys = op->GetArgs();
        for(auto&& key_arg : keys)
        {
            if(!key_arg.second.is_ptr)
            {
                arg_sizes.insert(key_arg.second.size());
                size_map[std::make_pair(idx, key_arg.second.size())].push_back(key_arg.first);
            }
            else
            {
                ptr_map[idx].push_back(key_arg.first);
            }
        }
    }

    arg_keys.clear();

    // if(kernel_source_type != Binary)
    if(lu.GetCurVertex(handle)->default_args.empty())
    {
        MIOPEN_LOG_I2("Predefined kernel args order not found");
        for(auto sz : arg_sizes) // Populate args for scalars
        {
            for(std::size_t idx = 0; idx < op_map.size(); idx++)
            {
                auto key_pair = std::make_pair(idx, sz);
                if(size_map.count(key_pair) > 0)
                {
                    auto keys = size_map.at(key_pair);
                    std::sort(keys.begin(), keys.end());
                    for(auto& key : keys)
                    {
                        MIOPEN_LOG_I("Scalar " << key << " = " << key);
                        arg_keys.emplace_back(key, Scalar, sz);
                    }
                }
            }
        }
        // insert input / output pointer
        arg_keys.emplace_back("reserved_input_tensor_ptr", Input_Ptr, sizeof(ConstData_t));
        arg_keys.emplace_back("reserved_output_tensor_ptr", Output_Ptr, sizeof(ConstData_t));
        // add other pointers in op-order
        for(std::size_t idx = 0; idx < op_map.size(); idx++)
        {
            auto op = op_map.at(idx);
            if(ptr_map.count(idx) > 0)
            {
                auto keys = ptr_map.at(idx);
                std::sort(keys.begin(), keys.end());
                std::transform(keys.begin(),
                               keys.end(),
                               std::back_inserter(arg_keys),
                               [&](auto&& key) -> Exec_arg_t {
                                   return {key, Pointer, sizeof(ConstData_t)};
                               });
            }
        }
        if(kernel_source_type == AsmText)
        { // Padded arguments
            std::vector<Exec_arg_t> padded_args;
            size_t running_sz = arg_keys[0].size;
            padded_args.push_back(arg_keys[0]);
            for(std::size_t idx = 1; idx < arg_keys.size(); idx++)
            {
                if(arg_keys[idx - 1].size != arg_keys[idx].size)
                {
                    auto padding = arg_keys[idx].size - running_sz % arg_keys[idx].size;
                    if(padding != 0)
                    {
                        MIOPEN_LOG_I("*** Padding: " << padding);
                        padded_args.emplace_back("reserved_padding", Padding, padding);
                        running_sz += padding;
                    }
                }
                padded_args.push_back(arg_keys[idx]);
                running_sz += arg_keys[idx].size;
            }
            arg_keys = std::move(padded_args);
        }

        if(arg_keys.empty())
        {
            MIOPEN_THROW("Kernel arguments not setup properly");
        }
    }
    else
    {
        auto default_args = lu.GetKernelArgs(handle);
        if(default_args.empty())
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "Default kernel args no supplied in metadata graph");
        }
        for(auto& arg : default_args)
        {
            MIOPEN_LOG_I2("Setting arg: " + arg.key);
            switch(arg.type)
            {
            case OpArg:
                if(arg.op_idx < op_map.size())
                {
                    auto& op = op_map.at(arg.op_idx);
                    auto k   = op->GetArgKey(arg.key);
                    arg_keys.emplace_back(
                        k, arg.default_val.is_ptr ? Pointer : Scalar, arg.default_val.size());
                    break;
                }
                else
                {
                    arg_keys.emplace_back(
                        arg.key, Default, arg.default_val.size(), arg.default_val);
                }
                break;
            case OpAttr:
                if(arg.op_idx < op_map.size())
                {
                    auto& op     = op_map.at(arg.op_idx);
                    auto op_attr = op->GetOpAttr(arg.key);
                    arg_keys.emplace_back(arg.key, Default, op_attr.size(), op_attr);
                }
                else
                {
                    arg_keys.emplace_back(
                        arg.key, Default, arg.default_val.size(), arg.default_val);
                }
                break;
            case Other:
                // The operator does not exist in the fusion plan, load the default value
                arg_keys.emplace_back(arg.key, Default, arg.default_val.size(), arg.default_val);
                break;
            case InputTensor:
                arg_keys.emplace_back("reserved_input_tensor_ptr", Input_Ptr, sizeof(ConstData_t));
                break;
            case OutputTensor:
                arg_keys.emplace_back(
                    "reserved_output_tensor_ptr", Output_Ptr, sizeof(ConstData_t));
                break;
            case DevAttribute: {
                auto dev_attr = GetDevAttribute(arg.key, handle);
                arg_keys.emplace_back(arg.key, Default, dev_attr.size(), dev_attr);
            }
            break;
            case InputTensorDesc:
            case OutputTensorDesc:
                auto tensor_arg = GetTensorAttr(arg.key);
                arg_keys.emplace_back(arg.key, Default, tensor_arg.size(), tensor_arg);
                break;
            }
        }
    }
    return arg_keys;
}

miopenStatus_t FusionPlanDescriptor::Execute(const Handle& handle,
                                             const TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             const TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const OperatorArgs& op_args)
{
    if(!isValid() || (lu.GetCurVertex(handle) == nullptr))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Attempting to execute an invalid fusion plan.");
    }

    if(output_desc != outputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The output descriptors dont match.");
    }
    if(input_desc != inputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The input descriptors dont match.");
    }

    auto ops_head = op_map[0];

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);
    MIOPEN_LOG_I(algorithm_name << ',' << network_config);
    if(kernels.empty())
    {
        MIOPEN_THROW(miopenStatusBadParm, "The FusionPlan was not compiled for execution");
    }
    KernelInvoke kernel = kernels.front();

    std::vector<OpKernelArg> args;
    if(arg_list.empty())
    {
        MIOPEN_THROW("Kernel arguments not setup properly");
    }
    for(auto& arg : arg_list)
    {
        MIOPEN_LOG_I2("Key: " + arg.key);
        switch(arg.type)
        {
        case Input_Ptr: args.emplace_back(OpKernelArg(input)); break;
        case Output_Ptr: args.emplace_back(OpKernelArg(output)); break;
        case Padding: args.emplace_back(OpKernelArg(0, arg.size)); break;
        case Scalar:
        case Pointer: {
            auto it = op_args.args_map.find(arg.key);
            if(it != op_args.args_map.end())
            {
                args.push_back(it->second);
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError, "Argument Not Set: " + arg.key);
            }
            break;
        }
        case Default: args.push_back(arg.val); break;
        }
    }
    if(args.empty())
    {
        MIOPEN_THROW("Operator args not populated properly");
    }
    kernel(args);
    return miopenStatusSuccess;
}

} // namespace miopen
