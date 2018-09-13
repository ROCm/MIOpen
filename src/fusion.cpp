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
#include <miopen/fusion_plan.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/visit_float.hpp>
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
    is_valid = lu.Advance(desc);
    return miopenStatusSuccess;
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
    std::size_t n, c, h, w;
    std::tie(n, c, h, w) = base_desc.GetForwardOutputDim(input_desc, filter_desc);
    TensorDescriptor desc(input_desc.GetType(), {n, c, h, w});
    output_desc = desc;
    return miopenStatusSuccess;
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

std::vector<std::string> ConvForwardOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    keys.push_back("weights" + std::to_string(GetIdx()));
    return keys;
}

FusionMDGraph_Edge_Map ConvForwardOpDescriptor::MDGraphKey(miopenConvolutionMode_t conv_mode,
                                                           miopenPaddingMode_t pad_mode,
                                                           size_t pad_h,
                                                           size_t pad_w,
                                                           size_t u,
                                                           size_t v,
                                                           size_t dilation_h,
                                                           size_t dilation_w,
                                                           int k,
                                                           int c,
                                                           int x,
                                                           int y)
{
    return {
        {"conv_mode", {EdgeOp(conv_mode, true, OpEqual)}},
        {"pad_mode", {EdgeOp(pad_mode, true, OpEqual)}},
        {"pad_h", {EdgeOp(pad_h, true, OpEqual)}},
        {"pad_w", {EdgeOp(pad_w, true, OpEqual)}},
        {"u", {EdgeOp(u, true, OpEqual)}},
        {"v", {EdgeOp(v, true, OpEqual)}},
        {"dilation_h", {EdgeOp(dilation_h, true, OpEqual)}},
        {"dilation_w", {EdgeOp(dilation_w, true, OpEqual)}},
        {"k", {EdgeOp(k, true, OpAny)}},
        {"c", {EdgeOp(c, true, OpAny)}},
        {"x", {EdgeOp(x, true, OpEqual)}},
        {"y", {EdgeOp(y, true, OpEqual)}},
    };
}

FusionMDGraph_Edge_Map ConvForwardOpDescriptor::MDGraphKey() const
{
    auto lens = filter_desc.GetLengths();
    int k, c, x, y;
    std::tie(k, c, x, y) = tien<4>(lens);
    auto m = ConvForwardOpDescriptor::MDGraphKey(base_desc.mode,
                                                 base_desc.paddingMode,
                                                 base_desc.pad_h,
                                                 base_desc.pad_w,
                                                 base_desc.u,
                                                 base_desc.v,
                                                 base_desc.dilation_h,
                                                 base_desc.dilation_w,
                                                 k,
                                                 c,
                                                 x,
                                                 y);
    map_emplace(m, "precision", EdgeOp(input_desc.GetType(), true, OpEqual));
    return m;
}

// Activ Forward
miopenStatus_t ActivFusionOpDescriptor::SetArgs(OperatorArgs& args,
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
        args.ins_arg("activAlpha" + id, OpKernelArg(static_cast<half_float::half>(activAlpha)));
        args.ins_arg("activBeta" + id, OpKernelArg(static_cast<half_float::half>(activBeta)));
        args.ins_arg("activGamma" + id, OpKernelArg(static_cast<half_float::half>(activGamma)));
    }
    return miopenStatusSuccess;
}

std::vector<std::string> ActivFusionOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    auto id = std::to_string(GetIdx());
    keys.push_back("activAlpha" + id);
    keys.push_back("activBeta" + id);
    keys.push_back("activGamma" + id);
    return keys;
}

miopenStatus_t ActivFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}
FusionMDGraph_Edge_Map ActivFusionOpDescriptor::MDGraphKey(miopenActivationMode_t mode)
{
    return {{"activ_mode", {EdgeOp(mode, true, OpEqual)}}};
}

FusionMDGraph_Edge_Map ActivFusionOpDescriptor::MDGraphKey() const
{
    return ActivFusionOpDescriptor::MDGraphKey(activMode);
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

miopenStatus_t BatchNormInferenceFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                             const void* alpha,
                                                             const void* beta,
                                                             ConstData_t bnScale,
                                                             ConstData_t bnBias,
                                                             ConstData_t estimatedMean,
                                                             ConstData_t estimatedVariance,
                                                             double epsilon)
{
    auto id                    = std::to_string(GetIdx());
    auto alpha_any             = OpKernelArg(*(static_cast<const float*>(alpha)));
    auto beta_any              = OpKernelArg(*(static_cast<const float*>(beta)));
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

std::vector<std::string> BatchNormInferenceFusionOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    auto id = std::to_string(GetIdx());
    keys.push_back("epsilon" + id);
    keys.push_back("bnScale" + id);
    keys.push_back("bnBias" + id);
    keys.push_back("estimatedMean" + id);
    keys.push_back("estimatedVariance" + id);
    return keys;
}

FusionMDGraph_Edge_Map
BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBatchNormMode_t bn_mode)
{
    return {{"bn_mode", {EdgeOp(bn_mode, true, OpEqual)}}};
}

FusionMDGraph_Edge_Map BatchNormInferenceFusionOpDescriptor::MDGraphKey() const
{
    return BatchNormInferenceFusionOpDescriptor::MDGraphKey(mode);
}

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

std::vector<std::string> BiasFusionOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    keys.push_back("bias" + std::to_string(GetIdx()));
    return keys;
}

FusionMDGraph_Edge_Map BiasFusionOpDescriptor::MDGraphKey() const
{
    return FusionMDGraph::EmptyEdgeMap();
}

static inline void
find_replace_first(std::string& s_where, const std::string& s_find, const std::string& s_replace)
{
    const auto pos = s_where.find(s_find);
    if(pos != std::string::npos)
        s_where.replace(pos, s_find.length(), s_replace);
}

std::string FusionPlanDescriptor::GetProgramName(Handle& handle)
{
    if(!op_map.empty())
    {
        program_name = lu.GetProgramName();
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

std::string FusionPlanDescriptor::GetKernelName()
{
    if(!op_map.empty())
    {
        kernel_name = lu.GetKernelName();
        return kernel_name;
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported starting op in Fusion Plan");
    }
}

std::string FusionPlanDescriptor::GetAlgorithmName()
{
    if(!op_map.empty())
    {
        algorithm_name = lu.GetAlgoName();
        return algorithm_name;
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported starting op in Fusion Plan");
    }
}

miopenStatus_t FusionPlanDescriptor::Compile(Handle& handle)
{
    miopenStatus_t status = miopenStatusUnknownError;
    if(!isValid())
    {
        MIOPEN_LOG_I2("Trying to compile an invalid FusionPlan");
        MIOPEN_THROW(miopenStatusBadParm);
    }
    network_config = "";
    network_config += output_desc.ToString();
    for(auto&& op : op_map)
    {
        op->GetNetworkConfig(network_config, handle);
    }
    // Check if the kernel is assembly or OpenCL
    auto ops_head  = op_map[0];
    algorithm_name = lu.GetAlgoName();
    program_name   = GetProgramName(handle);
    kernel_name    = GetKernelName();
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
        std::string compile_config;
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
        for(auto&& op : op_map)
        {
            MIOPEN_LOG_I2("GetCompileParms, " << *op);
            if(op->GetCompileParms(compile_config, handle, kernel_source_type, lu.GetSolvers()) !=
               miopenStatusSuccess)
            {
                MIOPEN_LOG_I2("Unsupported fusion plan");
                MIOPEN_THROW(miopenStatusInternalError);
            }
        }
        // TODO: This true for inference but might not be true in general
        // This is sill an open question
        // Must be preceded by GetCompileParms
        const auto& vld = ops_head->GetLocalWGSz(handle, algorithm_name);
        const auto& vgd = ops_head->GetGlobalWGSz(handle, algorithm_name);
        MIOPEN_LOG_I2("Program: " << program_name << ", kernel: " << kernel_name);
        MIOPEN_LOG_I2("Build options: " << compile_config);
        handle.AddKernel(
            algorithm_name, network_config, program_name, kernel_name, vld, vgd, compile_config);
        status = miopenStatusSuccess;
    }
    return status;
}

std::ostream& operator<<(std::ostream& s, const OpKernelArg& arg)
{
    union
    {
        unsigned long long ul = 0;
        double d;
        float f;
        half_float::half h;
        char c[sizeof(ul)];
    } val = {0};
    if(arg.buffer.size() > sizeof(val.ul))
        return s << "<too long value>";
    for(int i    = 0; i < arg.buffer.size(); ++i)
        val.c[i] = arg.buffer[i];
    s << std::hex << "0x" << val.ul << std::dec << " = " << static_cast<long long>(val.ul);
    switch(arg.buffer.size())
    {
    case 2: s << " = " << static_cast<float>(val.h) << "H"; break;
    case 4: s << " = " << val.f << "F"; break;
    case 8: s << " = " << val.d << "D"; break;
    default: break;
    }
    return s;
}

static OpKernelArg GetArg(std::vector<std::shared_ptr<FusionOpDescriptor>>& op_map,
                          const OperatorArgs& op_args,
                          const miopenFusionOp_t op,
                          const std::string arg_name)
{
    for(auto idx = 0; idx < op_map.size(); ++idx)
    {
        if(op_map[idx]->kind() == op)
        {
            std::string key = arg_name + std::to_string(idx);
            MIOPEN_LOG_I2(*op_map[idx] << ", finding: " << key);
            auto it = op_args.args_map.find(key);
            if(it != op_args.args_map.end())
            {
                MIOPEN_LOG_I2("found " << (it->second.is_ptr ? "pointer: " : "scalar: ") << key
                                       << " = "
                                       << it->second);
                return it->second;
            }
        }
    }
    MIOPEN_THROW(miopenStatusInternalError, "Not found: arg_name = " + arg_name);
}

#ifdef ADD_ARGUMENT
#error "ADD_ARGUMENT defined"
#endif
#define ADD_ARGUMENT(argument_name)                                                     \
    do                                                                                  \
    {                                                                                   \
        const OpKernelArg argument(argument_name);                                      \
        args.emplace_back(argument);                                                    \
        MIOPEN_LOG_I((argument.is_ptr ? "Pointer " : "Scalar ") << #argument_name " = " \
                                                                << argument);           \
    } while(false)

miopenStatus_t FusionPlanDescriptor::Execute(Handle& handle,
                                             TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const OperatorArgs& op_args)
{
    if(!isValid())
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

    // Construct the kernel args
    std::set<size_t> arg_sizes; // a set of argument sizes
    // A map between argument sizes and argument names
    std::map<std::pair<size_t, size_t>, std::vector<std::string>> size_map;
    // A map between argument pointers (buffers) and argument names
    std::map<size_t, std::vector<std::string>> ptr_map;

    for(auto idx = 0; idx < op_map.size(); idx++)
    {
        auto op   = op_map.at(idx);
        auto keys = op->GetArgs();
        for(auto&& key : keys)
        {
            auto it = op_args.args_map.find(key);
            if(it != op_args.args_map.end())
            {
                if(!it->second.is_ptr)
                {
                    arg_sizes.insert(it->second.size());
                    size_map[std::pair<size_t, size_t>(idx, it->second.size())].push_back(key);
                }
                else
                {
                    ptr_map[idx].push_back(key);
                }
            }
            else
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Arg " + key + " was not set for Operator: " +
                                 std::to_string(op->kind()));
            }
        }
    }
    std::vector<OpKernelArg> args;
    if(kernel_source_type == Binary)
    {
        if((input_desc.GetType() != miopenFloat) || (output_desc.GetType() != miopenFloat))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Only FP32 floats are currently supported for the Winograd kernel");
        }
        int N, C, H, W, oN, K, oH, oW;
        std::tie(N, C, H, W)    = miopen::tien<4>(input_desc.GetLengths(), 1);
        std::tie(oN, K, oH, oW) = miopen::tien<4>(output_desc.GetLengths(), 1);
        if(N != oN)
            MIOPEN_THROW("input and output batch sizes do not match");
        const int n_groups = handle.GetMaxComputeUnits();
        // Get topology (C>B>A, C>B, C>A), find out activation mode.
        assert(op_map[0]->kind() == miopenFusionOpConvForward && 2 <= op_map.size() &&
               op_map.size() <= 3);
        bool is_bias       = false;
        bool is_activation = false;
        bool is_leakyRELU  = false;
        for(const auto& op : op_map)
        {
            if(op->kind() == miopenFusionOpBiasForward)
                is_bias = true;
            else if(op->kind() == miopenFusionOpActivForward)
            {
                is_activation = true;
                is_leakyRELU =
                    (boost::any_cast<miopenActivationMode_t>(
                         op->MDGraphKey().at("activ_mode").at(0).val) == miopenActivationLEAKYRELU);
            }
        }
        const int flags        = (is_bias ? (1 << 7) : 0) + (is_activation ? (1 << 8) : 0);
        const int reserved     = 0;
        const int R            = 3;
        const int S            = 3;
        const int pad_h        = 0;
        const int pad_w        = 0;
        int* const return_addr = nullptr;
        const auto weights     = GetArg(op_map, op_args, miopenFusionOpConvForward, "weights");
        const auto bias = is_bias ? GetArg(op_map, op_args, miopenFusionOpBiasForward, "bias")
                                  : OpKernelArg(nullptr); // Kernel does not use it.
        const auto alpha = (is_activation && is_leakyRELU)
                               ? GetArg(op_map, op_args, miopenFusionOpActivForward, "activAlpha")
                               : OpKernelArg(0.0f); // Fixed to 0.0 for RELU.
        ADD_ARGUMENT(N);
        ADD_ARGUMENT(C);
        ADD_ARGUMENT(H);
        ADD_ARGUMENT(W);
        ADD_ARGUMENT(K);
        ADD_ARGUMENT(n_groups);
        ADD_ARGUMENT(flags);
        ADD_ARGUMENT(reserved);
        ADD_ARGUMENT(input);
        ADD_ARGUMENT(weights);
        ADD_ARGUMENT(output);
        ADD_ARGUMENT(return_addr);
        ADD_ARGUMENT(R);
        ADD_ARGUMENT(S);
        ADD_ARGUMENT(pad_h);
        ADD_ARGUMENT(pad_w);
        ADD_ARGUMENT(oH);
        ADD_ARGUMENT(oW);
        ADD_ARGUMENT(bias);
        ADD_ARGUMENT(alpha);
    }
    else
    {
        for(auto sz : arg_sizes) // Populate args for scalars
        {
            for(auto idx = 0; idx < op_map.size(); idx++)
            {
                auto key_pair = std::pair<size_t, size_t>(idx, sz);
                if(size_map.count(key_pair) > 0)
                {
                    auto keys = size_map.at(key_pair);
                    std::sort(keys.begin(), keys.end());
                    for(auto& key : keys)
                    {
                        auto it = op_args.args_map.find(key);
                        if(it != op_args.args_map.end())
                        {
                            MIOPEN_LOG_I("Scalar " << key << " = " << it->second);
                            args.push_back(it->second);
                        }
                    }
                }
            }
        }
        // insert input / output pointer
        args.emplace_back(OpKernelArg(input));
        MIOPEN_LOG_I("Input ptr = " << input);
        args.emplace_back(OpKernelArg(output));
        MIOPEN_LOG_I("Output ptr = " << output);
        // add other pointers in op-order
        for(auto idx = 0; idx < op_map.size(); idx++)
        {
            auto op = op_map.at(idx);
            if(ptr_map.count(idx) > 0)
            {
                auto keys = ptr_map.at(idx);
                std::sort(keys.begin(), keys.end());
                for(auto& key : keys)
                {
                    auto it = op_args.args_map.find(key);
                    if(it != op_args.args_map.end())
                    {
                        MIOPEN_LOG_I("Pointer " << key << " = " << it->second);
                        args.push_back(it->second);
                    }
                }
            }
        }
    }
    if(kernel_source_type == AsmText)
    { // Padded arguments
        std::vector<OpKernelArg> padded_args;
        size_t running_sz = args[0].size();
        padded_args.push_back(std::move(args[0]));
        for(auto idx = 1; idx < args.size(); idx++)
        {
            if(args[idx - 1].size() != args[idx].size())
            {
                auto padding = running_sz % args[idx].size();
                if(padding != 0)
                {
                    MIOPEN_LOG_I("*** Padding: " << padding);
                    OpKernelArg tmp(0, padding);
                    padded_args.push_back(tmp);
                    running_sz += padding;
                }
            }
            padded_args.push_back(std::move(args[idx]));
            running_sz += args[idx].size();
        }
        kernel(padded_args);
    }
    else
    {
        kernel(args);
    }
    return miopenStatusSuccess;
}

} // namespace miopen
