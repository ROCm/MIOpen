/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/rocm_features.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/mlir_common.hpp>

#include <sstream>
#include <string>

namespace miopen {
namespace solver {
namespace mlir {

static std::string InsertGToLayout(const std::string& layout, char dim)
{
    std::string layout_with_g = layout;
    std::size_t index         = layout.find(dim);
    if(index == std::string::npos)
        MIOPEN_THROW(std::string("Failed to find dim '") + dim + "' in the layout " + layout);
    return layout_with_g.insert(index, 1, 'G');
}

static const char* DTypeName(miopenDataType_t ty)
{
    switch(ty)
    {
    case miopenHalf: return "fp16";
    case miopenFloat: return "fp32";
    case miopenDouble: return "fp64";
    case miopenBFloat16: return "bf16";
    case miopenInt32: return "i32";
    case miopenInt8: return "i8";
    case miopenInt8x4: return "i8x4";
    }
    MIOPEN_THROW(miopenStatusInternalError, "Value outside of datatype enum");
}

static std::string GetIsaName(const miopen::TargetProperties& target)
{
#if ROCM_FEATURE_TARGETID_OFF
    const char* const ecc_suffix = (target.Sramecc() && *target.Sramecc()) ? ":sramecc+" : "";
    return {"amdgcn-amd-amdhsa:" + target.Name() + ecc_suffix};
#else
    const LcOptionTargetStrings lots(target);
    return "amdgcn-amd-amdhsa:" + lots.targetId;
#endif
}

std::string GetKernelName(const ConvolutionContext& ctx, bool is_xdlops, int kernel_id)
{
    std::string version;
    std::string direction;
    if(ctx.problem.direction.IsForward())
    {
        version   = "_v4r4";
        direction = "_fwd";
    }
    else if(ctx.problem.direction.IsBackwardData())
    {
        version   = "_v4r1";
        direction = "_bwd";
    }
    else
    {
        version   = "_v4r4";
        direction = "_wrw";
    }

    std::string kernel_name = "mlir_gen_igemm_conv2d" + version + direction;

    if(is_xdlops)
        kernel_name += "_xdlops";

    return kernel_name + std::to_string(kernel_id);
}

static std::string GetOperation(const ConvolutionContext& ctx)
{
    if(ctx.problem.direction.IsForward())
    {
        return "conv2d";
    }
    else if(ctx.problem.direction.IsBackwardData())
    {
        return "conv2d_bwd_data";
    }
    else
    {
        return "conv2d_bwd_weight";
    }
}

/* Construct the options string passed to MLIR to cause it
to generate a given convolution.*/
std::string ConstructBuildOptions(const ConvolutionContext& ctx, bool is_xdlops, int kernel_id)
{
    // Arguments for mlir-miopen-driver.
    using PDI = ProblemInterpreter;

    std::string operation   = GetOperation(ctx);
    std::string kernel_name = GetKernelName(ctx, is_xdlops, kernel_id);

    std::string in_layout  = InsertGToLayout(PDI::GetInputLayout(ctx.problem), 'C');
    std::string fil_layout = InsertGToLayout(PDI::GetFilterLayout(ctx.problem), 'N');
    std::string out_layout = InsertGToLayout(PDI::GetOutputLayout(ctx.problem), 'C');

    std::ostringstream mlir_handle;

    if(is_xdlops)
    {
        mlir_handle << " --x2 1";
    }

    const auto in_type  = PDI::GetInputDataType(ctx.problem);
    const auto fil_type = ctx.problem.weights_data_type;
    auto out_type       = PDI::GetOutputDataType(ctx.problem);

    // In case this is int8 convolution, ignore the output type and always request int32_t as
    // default output type. This is because MLIR invoker does casttensor on output if a non-int32_t
    // is requested.
    if(in_type == miopenInt8 && fil_type == miopenInt8)
    {
        out_type = miopenInt32;
    }

    // clang-format off
    mlir_handle
        << " --operation " << operation
        << " --kernel_id " << kernel_id
        << " --num_cu " << ctx.GetStream().GetMaxComputeUnits()
        << " --arch " << GetIsaName(ctx.GetStream().GetTargetProperties())
        << " --groupsize " << PDI::GetGroupCountG(ctx.problem)
        << " --fil_layout " << fil_layout
        << " --fil_type " << DTypeName(fil_type)
        << " --in_layout " << in_layout
        << " --out_layout " << out_layout
        << " --in_type " << DTypeName(in_type)
        << " --out_type " << DTypeName(out_type)
        << " --batchsize " << PDI::GetBatchN(ctx.problem)
        << " --in_channels " << PDI::GetInputChannelC(ctx.problem)
        << " --out_channels " << PDI::GetOutputChannelK(ctx.problem)
        << " --in_h " << PDI::GetInputHeightHi(ctx.problem)
        << " --in_w " << PDI::GetInputWidthWi(ctx.problem)
        << " --out_h " << PDI::GetOutputHeightHo(ctx.problem)
        << " --out_w " << PDI::GetOutputWidthWo(ctx.problem)
        << " --fil_h " << PDI::GetFilterHeightY(ctx.problem)
        << " --fil_w " << PDI::GetFilterWidthX(ctx.problem)
        << " --dilation_h " << PDI::GetAdjustedConvolutionDilationH(ctx.problem)
        << " --dilation_w " << PDI::GetAdjustedConvolutionDilationW(ctx.problem)
        << " --conv_stride_h " << PDI::GetAdjustedConvolutionStrideH(ctx.problem)
        << " --conv_stride_w " << PDI::GetAdjustedConvolutionStrideW(ctx.problem)
        << " --padding_h " << PDI::GetInputLeftPadH(ctx.problem)
        << " --padding_w " << PDI::GetInputLeftPadW(ctx.problem)
        << " --kernel_name " << kernel_name;
    // clang-format on
    return mlir_handle.str();
}

} // namespace mlir
} // namespace solver
} // namespace miopen
