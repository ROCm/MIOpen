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

namespace miopen {

FusionPlanDescriptor::~FusionPlanDescriptor()
{
    for(auto el : op_map)
        delete el.second.get();
}

miopenStatus_t FusionPlanDescriptor::AddOp(std::shared_ptr<FusionOpDescriptor> desc)
{
    desc->SetIdx(op_count);
    if(ins_order.empty())
        desc->SetInputDesc(input_desc);
    else
        desc->SetInputDesc(output_desc);
    desc->GetOutputDesc(output_desc);
    ins_order.push_back(op_count);
    op_map[op_count] = desc;
    op_count++;
    if(lu.Advance(desc->name()) == miopenStatusSuccess)
        is_valid = true;
    else
        is_valid = false;
    return miopenStatusSuccess;
}

TensorDescriptor FusionPlanDescriptor::DeriveOutputDescriptor()
{
    TensorDescriptor i_desc = input_desc;
    TensorDescriptor o_desc;
    if(fusion_dir == miopenVerticalFusion)
    {
        for(auto op_id : ins_order)
        {
            auto fod = op_map[op_id];
            fod->SetInputDesc(i_desc);
            fod->GetOutputDesc(o_desc);
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
        if(op.second->name() == miopenFusionOpConv)
        {
            auto ptr = std::dynamic_pointer_cast<ConvForwardOpDescriptor>(op.second);
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
    // args.ins_arg("alpha" + id, boost::spirit::hold_any(*f_alpha));
    // args.ins_arg("beta" + id, boost::spirit::hold_any(*(static_cast<const float*>(beta))));
    auto w_any = boost::spirit::hold_any(const_cast<void*>(w));
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
    auto alpha_any      = boost::spirit::hold_any(*(static_cast<const float*>(alpha)));
    auto beta_any       = boost::spirit::hold_any(*(static_cast<const float*>(beta)));
    auto activAlpha_any = boost::spirit::hold_any(activAlpha);
    auto activBeta_any  = boost::spirit::hold_any(activBeta);
    auto activGamma_any = boost::spirit::hold_any(activGamma);
    args.ins_arg("alpha" + id, alpha_any);
    args.ins_arg("beta" + id, beta_any);
    args.ins_arg("activAlpha" + id, activAlpha_any);
    args.ins_arg("activBeta" + id, activBeta_any);
    args.ins_arg("activGamma" + id, activGamma_any);
    return miopenStatusSuccess;
}

std::vector<std::string> ActivFusionOpDescriptor::GetArgs() const
{
    std::vector<std::string> keys;
    auto id = std::to_string(GetIdx());
    keys.push_back("alpha" + id);
    keys.push_back("beta" + id);
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
    //    args.ins_arg("alpha" + id, boost::spirit::hold_any(*static_cast<const float*>(alpha)));
    //    args.ins_arg("beta" + id, boost::spirit::hold_any(*static_cast<const float*>(beta)));
    auto bdata_any = boost::spirit::hold_any(const_cast<void*>(bdata));

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
miopenStatus_t FusionOpLU::Advance(miopenFusionOp_t op)
{
    size_t cur_idx_tmp = cur_idx;
    for(size_t idx = cur_idx; idx < lut.size(); idx++)
    {
        if(lut[idx] == op)
        {
            cur_idx = idx + 1;
            lut_hit.push_back(1);
            return miopenStatusSuccess;
        }
        else
            lut_hit.push_back(0);
    }
    lut_hit.resize(cur_idx_tmp);
    return miopenStatusInvalidValue;
}

auto FusionOpLU::GetPaths()
{
    std::vector<miopenFusionOp_t> vec;
    for(size_t idx = 0; lut_hit.size(); idx++)
    {
        if(lut_hit[idx] == 1)
            vec.push_back(lut[idx]);
    }
    return vec;
}

std::string FusionPlanDescriptor::GetKernelName(Handle& handle)
{
    auto conv_op = op_map[ins_order[0]];
    if(conv_op->name() == miopenFusionOpConv)
    {
        auto ki =
            std::dynamic_pointer_cast<ConvForwardOpDescriptor>(conv_op)->GetKernelInfo(handle);
        return ki.kernel_name;
    }
    else
    {
        MIOPEN_THROW("Unsupported starting op in Fusion Plan");
    }
}
// not clear where to put this

template <typename Ret, typename... Args>
Ret callfunc(std::function<Ret(Args...)> func, std::vector<boost::spirit::hold_any> anyargs);

template <typename Ret>
Ret callfunc(std::function<Ret()> func, std::vector<boost::spirit::hold_any> anyargs)
{
    if(anyargs.size() > 0)
        throw std::runtime_error("oops, argument list too long");
    return func();
}

template <typename Ret, typename Arg0, typename... Args>
Ret callfunc(std::function<Ret(Arg0, Args...)> func, std::vector<boost::spirit::hold_any> anyargs)
{
    if(anyargs.size() == 0)
        throw std::runtime_error("oops, argument list too short");
    Arg0 arg0 = boost::spirit::any_cast<Arg0>(anyargs[0]);
    anyargs.erase(anyargs.begin());
    std::function<Ret(Args... args)> lambda =
        ([=](Args... args) -> Ret { return func(arg0, args...); });
    return callfunc(lambda, anyargs);
}

template <typename Ret, typename... Args>
std::function<boost::spirit::hold_any(std::vector<boost::spirit::hold_any>)>
    adaptfunc(Ret (*func)(Args...))
{
    std::function<Ret(Args...)> stdfunc = func;
    std::function<boost::spirit::hold_any(std::vector<boost::spirit::hold_any>)> result =
        ([=](std::vector<boost::spirit::hold_any> anyargs) -> boost::spirit::hold_any {
            return boost::spirit::hold_any(callfunc(stdfunc, anyargs));
        });
    return result;
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
        MIOPEN_THROW("The execution plan is not valid");
    // TODO: The fusion plan is keeping track of the insertion order,
    // should we move this to the Graph ?
    for(auto nd : ins_order)
    {
        auto op = op_map[nd];
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
        for(auto nd : ins_order)
        {
            auto op = op_map[nd];
            op->GetCompileParms(compile_config, handle);
        }
        auto ops_head = op_map[ins_order[0]];
        // TODO: If the first op is Conv
        if(ops_head->name() == miopenFusionOpConv)
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
        // TODO: If the first op is batch norm!
        // else
        // {
        // }
    }
    // Construct the kernel args
    std::vector<boost::spirit::hold_any> args;
    args.push_back(boost::spirit::hold_any(const_cast<void*>(input)));
    args.push_back(boost::spirit::hold_any(static_cast<void*>(output)));
    for(auto nd : ins_order)
    {
        auto op   = op_map[nd];
        auto keys = op->GetArgs();
        for(auto key : keys)
        {
            boost::spirit::hold_any arg;
            auto it = op_args.args_map.find(key);
            if(it != op_args.args_map.end())
                arg = it->second;
            else
                MIOPEN_THROW("Arg not found in Map");
            args.push_back(arg);
        }
    }
    kernel(args);
    return miopenStatusSuccess;
}

} // namespace miopen
