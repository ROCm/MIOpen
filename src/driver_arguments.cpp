/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <miopen/driver_arguments.hpp>
#include <miopen/fusion_plan.hpp>

namespace miopen {
namespace debug {

miopenProblemDirection_t CmdArgToDirection(ConvDirection direction)
{
    switch(direction)
    {
    case ConvDirection::Fwd: return miopenProblemDirectionForward;
    case ConvDirection::Bwd: return miopenProblemDirectionBackward;
    case ConvDirection::WrW: return miopenProblemDirectionBackwardWeights;
    };
    MIOPEN_THROW(miopenStatusInternalError);
}

void ConvDataType(std::stringstream& ss, const miopen::TensorDescriptor& desc)
{
    if(desc.GetType() == miopenHalf)
    {
        ss << "convfp16";
    }
    else if(desc.GetType() == miopenBFloat16)
    {
        ss << "convbfp16";
    }
    else if(desc.GetType() == miopenInt8)
    {
        ss << "convint8";
    }
    else
    {
        ss << "conv";
    }
}

void BnDataType(std::stringstream& ss, const miopen::TensorDescriptor& desc)
{
    if(desc.GetType() == miopenHalf)
    {
        ss << "bnormfp16";
    }
    else
    {
        ss << "bnorm";
    }
}

void BnDriverInfo(std::stringstream& ss,
                  const BatchNormDirection_t& dir,
                  const void* resultRunningMean,
                  const void* resultRunningVariance,
                  const void* resultSaveMean,
                  const void* resultSaveInvVariance)
{
    if(dir != Backward)
    {
        ss << " --forw " << (dir == ForwardInference ? "2" : "1") << " -b 0";
    }
    else
    {
        ss << " --forw 0 -b 1";
    }
    if((resultRunningMean != nullptr) && (resultRunningVariance != nullptr))
    {
        ss << " -s 1";
    }
    if((resultSaveMean != nullptr) && (resultSaveInvVariance != nullptr))
    {
        ss << " -r 1";
    }
}

std::string ConvArgsForMIOpenDriver(const miopen::TensorDescriptor& xDesc,
                                    const miopen::TensorDescriptor& wDesc,
                                    const miopen::ConvolutionDescriptor& convDesc,
                                    const miopen::TensorDescriptor& yDesc,
                                    const miopenProblemDirection_t& dir,
                                    std::optional<uint64_t> immediate_mode_solver_id,
                                    bool print_for_conv_driver)
{
    const auto conv_dir = [&]() {
        switch(dir)
        {
        case miopenProblemDirectionForward: return ConvDirection::Fwd;
        case miopenProblemDirectionBackward: return ConvDirection::Bwd;
        case miopenProblemDirectionBackwardWeights: return ConvDirection::WrW;
        case miopenProblemDirectionInference:
            MIOPEN_THROW(miopenStatusInternalError);
            return ConvDirection::Fwd;
        }
    }();

    std::stringstream ss;
    if(print_for_conv_driver)
        ConvDataType(ss, xDesc);

    /// \todo Dimensions (N, C, H, W, K..) are always parsed as if layout is NC(D)HW.
    /// For other layouts, invalid values are printed.

    if(convDesc.GetSpatialDimension() == 2)
    {
        ss << " -n " << xDesc.GetLengths()[0] //
           << " -c " << xDesc.GetLengths()[1] //
           << " -H " << xDesc.GetLengths()[2] //
           << " -W " << xDesc.GetLengths()[3] //
           << " -k "
           << (convDesc.mode == miopenTranspose        //
                   ? wDesc.GetLengths()[1]             //
                   : wDesc.GetLengths()[0])            //
           << " -y " << wDesc.GetLengths()[2]          //
           << " -x " << wDesc.GetLengths()[3]          //
           << " -p " << convDesc.GetConvPads()[0]      //
           << " -q " << convDesc.GetConvPads()[1]      //
           << " -u " << convDesc.GetConvStrides()[0]   //
           << " -v " << convDesc.GetConvStrides()[1]   //
           << " -l " << convDesc.GetConvDilations()[0] //
           << " -j " << convDesc.GetConvDilations()[1];
        std::string x_layout = xDesc.GetLayout("NCHW");
        std::string w_layout = wDesc.GetLayout("NCHW");
        std::string y_layout = yDesc.GetLayout("NCHW");
        if(x_layout != "NCHW")
        {
            ss << " --in_layout " << x_layout;
        }
        if(w_layout != "NCHW")
        {
            ss << " --fil_layout " << w_layout;
        }
        if(y_layout != "NCHW")
        {
            ss << " --out_layout " << y_layout;
        }
    }
    else if(convDesc.GetSpatialDimension() == 3)
    {
        ss << " -n " << xDesc.GetLengths()[0]     //
           << " -c " << xDesc.GetLengths()[1]     //
           << " --in_d " << xDesc.GetLengths()[2] //
           << " -H " << xDesc.GetLengths()[3]     //
           << " -W " << xDesc.GetLengths()[4]     //
           << " -k "
           << (convDesc.mode == miopenTranspose                   //
                   ? wDesc.GetLengths()[1]                        //
                   : wDesc.GetLengths()[0])                       //
           << " --fil_d " << wDesc.GetLengths()[2]                //
           << " -y " << wDesc.GetLengths()[3]                     //
           << " -x " << wDesc.GetLengths()[4]                     //
           << " --pad_d " << convDesc.GetConvPads()[0]            //
           << " -p " << convDesc.GetConvPads()[1]                 //
           << " -q " << convDesc.GetConvPads()[2]                 //
           << " --conv_stride_d " << convDesc.GetConvStrides()[0] //
           << " -u " << convDesc.GetConvStrides()[1]              //
           << " -v " << convDesc.GetConvStrides()[2]              //
           << " --dilation_d " << convDesc.GetConvDilations()[0]  //
           << " -l " << convDesc.GetConvDilations()[1]            //
           << " -j " << convDesc.GetConvDilations()[2]            //
           << " --spatial_dim 3";
        std::string x_layout = xDesc.GetLayout("NCDHW");
        std::string w_layout = wDesc.GetLayout("NCDHW");
        std::string y_layout = yDesc.GetLayout("NCDHW");
        if(x_layout != "NCDHW")
        {
            ss << " --in_layout " << x_layout;
        }
        if(w_layout != "NCDHW")
        {
            ss << " --fil_layout " << w_layout;
        }
        if(y_layout != "NCDHW")
        {
            ss << " --out_layout " << y_layout;
        }
    }
    if(print_for_conv_driver)
        ss << " -m " << (convDesc.mode == 1 ? "trans" : "conv"); // clang-format off
    ss << " -g " << convDesc.group_count;
    if(print_for_conv_driver)
        ss << " -F " << std::to_string(static_cast<int>(conv_dir)) << " -t 1"; // clang-format on
    if(immediate_mode_solver_id.has_value())
    {
        ss << " -S " << *immediate_mode_solver_id;
    }

    return ss.str();
}

std::string BnormArgsForMIOpenDriver(miopenTensorDescriptor_t xDesc,
                                     miopenBatchNormMode_t bn_mode,
                                     const void* resultRunningMean,
                                     const void* resultRunningVariance,
                                     const void* resultSaveMean,
                                     const void* resultSaveInvVariance,
                                     const BatchNormDirection_t& dir,
                                     bool print_for_bn_driver)
{
    int size = {0};
    miopenGetTensorDescriptorSize(xDesc, &size);
    std::stringstream ss;
    if(print_for_bn_driver)
        BnDataType(ss, miopen::deref(xDesc));

    ss << " -n " << miopen::deref(xDesc).GetLengths()[0] // clang-format off
            << " -c " << miopen::deref(xDesc).GetLengths()[1];
        if(size == 5)
        {
            ss << " -D " << miopen::deref(xDesc).GetLengths()[2]
            << " -H " << miopen::deref(xDesc).GetLengths()[3]
            << " -W " << miopen::deref(xDesc).GetLengths()[4];
        }
        else
        {
            ss << " -H " << miopen::deref(xDesc).GetLengths()[2]
            << " -W " << miopen::deref(xDesc).GetLengths()[3];
        }
            ss << " -m " << bn_mode; // clang-format on
    if(print_for_bn_driver)
    {
        BnDriverInfo(ss,
                     dir,
                     resultRunningMean,
                     resultRunningVariance,
                     resultSaveMean,
                     resultSaveInvVariance);
    }
    return ss.str();
}

int GetFusionMode(const miopenFusionPlanDescriptor_t& fusePlanDesc)
{
    int fusion_mode = -1;

    if(miopen::deref(fusePlanDesc).op_map.size() == 4 &&
       (miopen::deref(fusePlanDesc).op_map[0]->kind() == miopenFusionOpConvForward) &&
       (miopen::deref(fusePlanDesc).op_map[1]->kind() == miopenFusionOpBiasForward) &&
       (miopen::deref(fusePlanDesc).op_map[2]->kind() == miopenFusionOpBatchNormInference) &&
       (miopen::deref(fusePlanDesc).op_map[3]->kind() == miopenFusionOpActivForward))
    {
        fusion_mode = 0;
    }
    else if(miopen::deref(fusePlanDesc).op_map.size() == 3 &&
            (miopen::deref(fusePlanDesc).op_map[0]->kind() == miopenFusionOpConvForward) &&
            (miopen::deref(fusePlanDesc).op_map[1]->kind() == miopenFusionOpBatchNormInference) &&
            (miopen::deref(fusePlanDesc).op_map[2]->kind() == miopenFusionOpActivForward))
    {
        fusion_mode = 1;
    }
    else if(miopen::deref(fusePlanDesc).op_map.size() == 2 &&
            (miopen::deref(fusePlanDesc).op_map[0]->kind() == miopenFusionOpBatchNormInference) &&
            (miopen::deref(fusePlanDesc).op_map[1]->kind() == miopenFusionOpActivForward))
    {
        fusion_mode = 2;
    }
    else if(miopen::deref(fusePlanDesc).op_map.size() == 2 &&
            (miopen::deref(fusePlanDesc).op_map[0]->kind() == miopenFusionOpConvForward) &&
            (miopen::deref(fusePlanDesc).op_map[1]->kind() == miopenFusionOpBatchNormInference))
    {
        fusion_mode = 3;
    }
    else if(miopen::deref(fusePlanDesc).op_map.size() == 3 &&
            (miopen::deref(fusePlanDesc).op_map[0]->kind() == miopenFusionOpConvForward) &&
            (miopen::deref(fusePlanDesc).op_map[1]->kind() == miopenFusionOpBiasForward) &&
            (miopen::deref(fusePlanDesc).op_map[2]->kind() == miopenFusionOpActivForward))
    {
        fusion_mode = 4;
    }
    else if(miopen::deref(fusePlanDesc).op_map.size() == 2 &&
            (miopen::deref(fusePlanDesc).op_map[0]->kind() == miopenFusionOpConvForward) &&
            (miopen::deref(fusePlanDesc).op_map[1]->kind() == miopenFusionOpActivForward))
    {
        fusion_mode = 5;
    }
    else if(miopen::deref(fusePlanDesc).op_map.size() == 2 &&
            (miopen::deref(fusePlanDesc).op_map[0]->kind() == miopenFusionOpConvForward) &&
            (miopen::deref(fusePlanDesc).op_map[1]->kind() == miopenFusionOpBiasForward))
    {
        fusion_mode = 6;
    }

    if(fusion_mode < 0)
    {
        MIOPEN_LOG_E("Unknown fusion plan : " << fusion_mode);
    }

    return fusion_mode;
}

} // namespace debug
} // namespace miopen
