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
    if(ctx.direction.IsForward())
    {
        version   = "_v4r4";
        direction = "_fwd";
    }
    else if(ctx.direction.IsBackwardData())
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
    if(ctx.direction.IsForward())
    {
        return "conv2d";
    }
    else if(ctx.direction.IsBackwardData())
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
std::string ConstructBuildOptions(const ConvolutionContext& ctx,
                                  bool is_xdlops,
                                  int kernel_id)
{
    // Arguments for mlir-miopen-driver.
    using CI = ConvolutionContextInterpreter;

    std::string operation   = GetOperation(ctx);
    std::string kernel_name = GetKernelName(ctx, is_xdlops, kernel_id);

    std::string in_layout  = InsertGToLayout(CI::GetInputLayout(ctx), 'C');
    std::string fil_layout = InsertGToLayout(CI::GetFilterLayout(ctx), 'N');
    std::string out_layout = InsertGToLayout(CI::GetOutputLayout(ctx), 'C');

    std::ostringstream mlir_handle;

    if(is_xdlops)
    {
        mlir_handle << " --x2 1";
    }

    // clang-format off
    mlir_handle
        << " --operation " << operation
        << " --kernel_id " << kernel_id
        << " --num_cu " << ctx.GetStream().GetMaxComputeUnits()
        << " --arch " << GetIsaName(ctx.GetStream().GetTargetProperties())
        << " --groupsize " << CI::GetGroupCountG(ctx)
        << " --fil_layout " << fil_layout
        << " --fil_type " << DTypeName(ctx.weights_data_type)
        << " --in_layout " << in_layout
        << " --out_layout " << out_layout
        << " --in_type " << DTypeName(CI::GetInputDataType(ctx))
        << " --out_type " << DTypeName(CI::GetOutputDataType(ctx))
        << " --batchsize " << CI::GetBatchN(ctx)
        << " --in_channels " << CI::GetInputChannelC(ctx)
        << " --out_channels " << CI::GetOutputChannelK(ctx)
        << " --in_h " << CI::GetInputHeightHi(ctx)
        << " --in_w " << CI::GetInputWidthWi(ctx)
        << " --out_h " << CI::GetOutputHeightHo(ctx)
        << " --out_w " << CI::GetOutputWidthWo(ctx)
        << " --fil_h " << CI::GetFilterHeightY(ctx)
        << " --fil_w " << CI::GetFilterWidthX(ctx)
        << " --dilation_h " << CI::GetAdjustedConvolutionDilationH(ctx)
        << " --dilation_w " << CI::GetAdjustedConvolutionDilationW(ctx)
        << " --conv_stride_h " << CI::GetAdjustedConvolutionStrideH(ctx)
        << " --conv_stride_w " << CI::GetAdjustedConvolutionStrideW(ctx)
        << " --padding_h " << CI::GetInputLeftPadH(ctx)
        << " --padding_w " << CI::GetInputLeftPadW(ctx)
        << " --kernel_name " << kernel_name;
    // clang-format on
    return mlir_handle.str();
}

std::string ConstructBuildOptions(const ConvolutionContext& ctx,
                                  const std::string& config,
                                  bool is_xdlops,
                                  int kernel_id)
{
    std::ostringstream mlir_handle;

    // clang-format off
    mlir_handle
        << ConstructBuildOptions(ctx, is_xdlops, kernel_id)
        << " --perf_config " << config;
    // clang-format on

    return mlir_handle.str();
}

} // namespace mlir
} // namespace solver
} // namespace miopen
