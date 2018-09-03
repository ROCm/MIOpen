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
    desc = op_map.at(op_idx);
    return miopenStatusSuccess;
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
        MIOPEN_THROW("Unsupported fusion direction");
    }
    return o_desc;
}

std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& fpd)
{
    stream << "kernel_name: " << fpd.kernel_name;
    return stream;
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
        args.ins_arg("activAlpha" + id, any_t(static_cast<float>(activAlpha)));
        args.ins_arg("activBeta" + id, any_t(static_cast<float>(activBeta)));
        args.ins_arg("activGamma" + id, any_t(static_cast<float>(activGamma)));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        args.ins_arg("activAlpha" + id, any_t(static_cast<half_float::half>(activAlpha)));
        args.ins_arg("activBeta" + id, any_t(static_cast<half_float::half>(activBeta)));
        args.ins_arg("activGamma" + id, any_t(static_cast<half_float::half>(activGamma)));
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
    return {{"activ_mode", EdgeOp(mode, true, OpEqual)}};
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
    auto alpha_any             = any_t(*(static_cast<const float*>(alpha)));
    auto beta_any              = any_t(*(static_cast<const float*>(beta)));
    auto bnScale_any           = any_t(bnScale);
    auto bnBias_any            = any_t(bnBias);
    auto estimatedMean_any     = any_t(estimatedMean);
    auto estimatedVariance_any = any_t(estimatedVariance);
    auto epsilon_any           = any_t(static_cast<double>(epsilon));
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
    return {{"bn_mode", EdgeOp(bn_mode, true, OpEqual)}};
}

FusionMDGraph_Edge_Map BatchNormInferenceFusionOpDescriptor::MDGraphKey() const
{
    return BatchNormInferenceFusionOpDescriptor::MDGraphKey(mode);
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
        MIOPEN_THROW("Unsupported starting op in Fusion Plan");
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
        MIOPEN_THROW("Unsupported starting op in Fusion Plan");
    }
}

miopenStatus_t FusionPlanDescriptor::Compile(Handle& handle)
{
    miopenStatus_t status = miopenStatusUnknownError;
    if(!isValid())
    {
        MIOPEN_THROW("Trying to compile and invalid FusionPlan");
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
        MIOPEN_THROW("Invalid Fusion Plan");
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
        // TODO: This true for inference but might not be true in general
        // This is sill an open question
        const auto& vld = ops_head->GetLocalWGSz(handle, algorithm_name);
        const auto& vgd = ops_head->GetGlobalWGSz(handle, algorithm_name);
        for(auto&& op : op_map)
        {
            MIOPEN_LOG_I2("GetCompileParms, " << *op);
            op->GetCompileParms(compile_config, handle, kernel_source_type);
        }
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

miopenStatus_t FusionPlanDescriptor::Execute(Handle& handle,
                                             TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const OperatorArgs& op_args)
{
    if(!isValid())
    {
        MIOPEN_THROW("Attempting to execute an invalid fusion plan.");
    }

    if(output_desc != outputDesc)
    {
        MIOPEN_THROW("The output descriptors dont match.");
    }
    if(input_desc != inputDesc)
    {
        MIOPEN_THROW("The input descriptors dont match.");
    }

    auto ops_head = op_map[0];

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);
    MIOPEN_LOG_I(algorithm_name << ',' << network_config);
    if(kernels.empty())
    {
        MIOPEN_THROW("The FusionPlan was not compiled for execution");
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
        auto op   = op_map[idx];
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
                MIOPEN_THROW("Arg " + key + " was not set for Operator: " +
                             std::to_string(op->kind()));
        }
    }
    std::vector<any_t> args;

    for(auto sz : arg_sizes) // Populate args for scalars
    {
        for(auto idx = 0; idx < op_map.size(); idx++)
        {
            auto op   = op_map[idx];
            auto keys = size_map[std::pair<size_t, size_t>(idx, sz)];
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
    // insert input / output pointer
    args.emplace_back(any_t(input));
    MIOPEN_LOG_I("Input ptr = " << input);
    args.emplace_back(any_t(output));
    MIOPEN_LOG_I("Output ptr = " << output);
    // add other pointers in op-order
    for(auto idx = 0; idx < op_map.size(); idx++)
    { // Populate args for pointers based operator order
        auto op   = op_map[idx];
        auto keys = ptr_map[idx];
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

    if(kernel_source_type == AsmText)
    { // Padded arguments
        std::vector<any_t> padded_args;
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
                    any_t tmp(0, padding);
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
