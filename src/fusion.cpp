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

bool FusionPlanDescriptor::isValid()
{
    TensorDescriptor o_desc = this->DeriveOutputDescriptor();
    return true;
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
    return miopenStatusSuccess;
}

TensorDescriptor FusionPlanDescriptor::DeriveOutputDescriptor()
{
    TensorDescriptor i_desc = input_desc;
    TensorDescriptor o_desc;
    if(fusion_dir == miopenVertical)
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
                                                const Data_t w)
{
    // args.append_arg(*(static_cast<const float*>(alpha)));
    const float* f_alpha = static_cast<const float*>(alpha);
    args.append_arg(boost::spirit::hold_any(*f_alpha));
    args.append_arg(boost::spirit::hold_any(*(static_cast<const float*>(beta))));
    args.append_arg(boost::spirit::hold_any(static_cast<void*>(w)));
    return miopenStatusSuccess;
}

// Activ Forward
miopenStatus_t ActivFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc)
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}

} // namespace miopen
