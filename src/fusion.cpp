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
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/visit_float.hpp>

namespace miopen {

FusionPlanDescriptor::~FusionPlanDescriptor()
{
    for(auto el : op_map)
        delete el.get();
}

miopenStatus_t FusionPlanDescriptor::AddOp(std::shared_ptr<FusionOpDescriptor> desc)
{
    desc->SetIdx(op_count);
    if(op_map.empty())
        desc->SetInputDesc(input_desc);
    else
        desc->SetInputDesc(output_desc);
    desc->GetOutputDesc(output_desc);
    op_map.push_back(desc);
    op_count++;
    is_valid = lu.Advance(op_map);
    return miopenStatusSuccess;
}

TensorDescriptor FusionPlanDescriptor::DeriveOutputDescriptor()
{
    TensorDescriptor i_desc = input_desc;
    TensorDescriptor o_desc;
    if(fusion_dir == miopenVerticalFusion)
    {
        for(auto op : op_map)
        {
            op->SetInputDesc(i_desc);
            op->GetOutputDesc(o_desc);
            i_desc = o_desc;
        }
    }
    else
    {
        // TODO: All the ops should have the same output descriptor otherwise
        // fusion would not be feasible, thus we need to call GetOutputDesc on all
        // the ops and make sure it returns the same value
        MIOPEN_THROW("Unsupported fusion direction");
    }
    return o_desc;
}

miopenStatus_t FusionPlanDescriptor::GetWorkspaceSizeImmed(Handle& handle,
                                                           size_t& workSpaceSize,
                                                           miopenConvFwdAlgorithm_t algo)
{
    workSpaceSize = 0;
    // iterate over all the conv ops in the plan and return the max amount of
    // ws required
    for(auto op : op_map)
    {
        if(op->kind() == miopenFusionOpConvForward)
        {
            auto ptr = std::dynamic_pointer_cast<ConvForwardOpDescriptor>(op);
            TensorDescriptor opd;
            ptr->GetOutputDesc(opd);
            bool supported = false;
            size_t tmp_sz  = ptr->base_desc.ForwardGetWorkSpaceSizeImmed(
                handle, ptr->filter_desc, ptr->input_desc, opd, algo, supported);
            if(supported && (tmp_sz > workSpaceSize))
                workSpaceSize = tmp_sz;
        }
    }
    return miopenStatusSuccess;
}

std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& fpd)
{
    (void)(fpd);
    /*    MIOPEN_LOG_ENUM(stream,
                        x.mode,
                        miopenActivationPASTHRU,
                        miopenActivationLOGISTIC,
                        miopenActivationTANH,
                        miopenActivationRELU,
                        miopenActivationSOFTRELU,
                        miopenActivationABS,
                        miopenActivationPOWER,
                        miopenActivationCLIPPEDRELU,
                        miopenActivationLEAKYRELU,
                        miopenActivationELU)*/
    // LogRange(stream, x.parms, ", ") << ", ";
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
                                                const void* alpha,
                                                const void* beta,
                                                ConstData_t w)
{
    (void)(alpha);
    (void)(beta);
    // const float* f_alpha = static_cast<const float*>(alpha);
    auto id = std::to_string(GetIdx());
    // args.ins_arg("alpha" + id, any_t(*f_alpha));
    // args.ins_arg("beta" + id, any_t(*(static_cast<const float*>(beta))));
    auto w_any = any_t(w);
    args.ins_arg("weights" + id, w_any);
    return miopenStatusSuccess;
}

std::vector<std::string> ConvForwardOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    keys.push_back("weights" + std::to_string(GetIdx()));
    return keys;
}

// Activ Forward
miopenStatus_t ActivFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                                const void* alpha,
                                                const void* beta,
                                                double activAlpha,
                                                double activBeta,
                                                double activGamma)
{
    auto id             = std::to_string(GetIdx());
    auto alpha_any      = any_t(*(static_cast<const float*>(alpha)));
    auto beta_any       = any_t(*(static_cast<const float*>(beta)));
    auto activAlpha_any = any_t(static_cast<double>(activAlpha));
    auto activBeta_any  = any_t(static_cast<double>(activBeta));
    auto activGamma_any = any_t(static_cast<double>(activGamma));
    // args.ins_arg("alpha" + id, alpha_any);
    // args.ins_arg("beta" + id, beta_any);
    args.ins_arg("activAlpha" + id, activAlpha_any);
    args.ins_arg("activBeta" + id, activBeta_any);
    args.ins_arg("activGamma" + id, activGamma_any);
    return miopenStatusSuccess;
}

std::vector<std::string> ActivFusionOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    auto id = std::to_string(GetIdx());
    // keys.push_back("alpha" + id);
    // keys.push_back("beta" + id);
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
    args.ins_arg("bnScale" + id, bnScale_any);
    args.ins_arg("bnBias" + id, bnBias_any);
    args.ins_arg("estimatedMean" + id, estimatedMean_any);
    args.ins_arg("estimatedVariance" + id, estimatedVariance_any);
    args.ins_arg("epsilon" + id, epsilon_any);

    return miopenStatusSuccess;
}

std::vector<std::string> BatchNormInferenceFusionOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    auto id = std::to_string(GetIdx());
    // keys.push_back("alpha" + id);
    // keys.push_back("beta" + id);
    keys.push_back("bnScale" + id);
    keys.push_back("bnBias" + id);
    keys.push_back("estimatedMean" + id);
    keys.push_back("estimatedVariance" + id);
    keys.push_back("epsilon" + id);
    return keys;
}

// Bias forward
miopenStatus_t BiasFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

miopenStatus_t BiasFusionOpDescriptor::SetArgs(OperatorArgs& args,
                                               const void* alpha,
                                               const void* beta,
                                               ConstData_t bdata)
{
    auto id = std::to_string(GetIdx());
    (void)(alpha);
    (void)(beta);
    //    args.ins_arg("alpha" + id, any_t(*static_cast<const float*>(alpha)));
    //    args.ins_arg("beta" + id, any_t(*static_cast<const float*>(beta)));
    auto bdata_any = any_t(bdata);
    args.ins_arg("bias" + id, bdata_any);

    return miopenStatusSuccess;
}

std::vector<std::string> BiasFusionOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    keys.push_back("bias" + std::to_string(GetIdx()));
    return keys;
}
// Op LUT
bool FusionOpLU::Advance(std::vector<std::shared_ptr<FusionOpDescriptor>> op_map)
{

    for(auto supportedOps : lut)
    {
        auto valid = std::equal(supportedOps.begin(),
                                supportedOps.end(),
                                op_map.begin(),
                                [](miopenFusionOp_t x, std::shared_ptr<FusionOpDescriptor> y) {
                                    return x == y->kind();
                                });
        if(valid)
            return valid;
    }
    /*    if(valid)
        {
            cur_idx = idx + 1;
            lut_hit.push_back(1);
            return valid;
        }
        else
            lut_hit.push_back(0);

        lut_hit.resize(cur_idx_tmp);*/
    return false;
}

/*auto FusionOpLU::GetPaths()
{
    std::vector<miopenFusionOp_t> vec;
    for(size_t idx = 0; lut_hit.size(); idx++)
    {
        if(lut_hit[idx] == 1)
            vec.push_back(lut[idx]);
    }
    return vec;
}
*/
std::string FusionPlanDescriptor::GetKernelName(Handle& handle)
{
    auto starting_op = op_map.at(0);
    if(starting_op->kind() == miopenFusionOpConvForward)
    {
        auto ki =
            std::dynamic_pointer_cast<ConvForwardOpDescriptor>(starting_op)->GetKernelInfo(handle);
        return ki.kernel_name;
    }
    else if(starting_op->kind() == miopenFusionOpBatchNormInference)
    {
        /*        auto ki =
                    std::dynamic_pointer_cast<BatchNormInferenceFusionOpDescriptor>(starting_op)->GetKernelInfo(handle);
                return ki.kernel_name;*/
        return "";
    }
    else
    {
        MIOPEN_THROW("Unsupported starting op in Fusion Plan");
    }
}

miopenStatus_t FusionPlanDescriptor::Execute(Handle& handle,
                                             TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const OperatorArgs& op_args)
{
    std::string network_config;
    // TODO: move the hard coded algo name to the LUT
    std::string algorithm_name = "miopenDirConvBatchNormActivAlgo";
    if(output_desc != outputDesc)
    {
        MIOPEN_THROW("The output descriptors dont match");
    }
    if(input_desc != inputDesc)
    {
        MIOPEN_THROW("The input descriptors dont match");
    }
    if(!isValid())
        MIOPEN_THROW("The execution plan is not valid.");
    // TODO: The fusion plan is keeping track of the insertion order,
    // should we move this to the Graph ?
    for(auto op : op_map)
    {
        op->GetNetworkConfig(network_config, handle);
    }

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);
    KernelInvoke kernel;
    if(!kernels.empty())
    {
        kernel = kernels.front();
    }
    else
    {
        std::string compile_config;
        for(auto op : op_map)
        {
            op->GetCompileParms(compile_config, handle);
        }
        auto ops_head = op_map[0]; // ins_order[0]];
        // TODO: If the first op is Conv
        if(ops_head->kind() == miopenFusionOpConvForward)
        {
            auto ki =
                std::dynamic_pointer_cast<ConvForwardOpDescriptor>(ops_head)->GetKernelInfo(handle);
            auto program_name = ki.kernel_file;
            auto kernel_name  = ki.kernel_name;
            const auto parms  = ki.comp_options + compile_config;
            const auto& vld   = ki.l_wk;
            const auto& vgd   = ki.g_wk;

            kernel = handle.AddKernel(
                algorithm_name, network_config, program_name, kernel_name, vld, vgd, parms);
        }
        else if(ops_head->kind() == miopenFusionOpBatchNormInference)
        {
/*            auto ki =
                std::dynamic_pointer_cast<BatchNormInferenceFusionOpDescriptor>(ops_head)->GetKernelInfo(handle);
*/        }

// TODO: If the first op is batch norm!
// else
// {
// }
    }
    // Construct the kernel args
    std::vector<any_t> args;
    args.push_back(any_t(input));
    args.push_back(any_t(output));
    for(auto op : op_map)
    {
        auto keys = op->GetArgs();
        for(auto key : keys)
        {
            auto it = op_args.args_map.find(key);
            if(it != op_args.args_map.end())
                args.push_back(any_t(it->second));
            else
                MIOPEN_THROW("Arg not found in Map");
        }
    }
    kernel(args);
    return miopenStatusSuccess;
}

} // namespace miopen
