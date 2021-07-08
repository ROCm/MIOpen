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

#include "miopen/miopen.h"
#include <miopen/errors.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/mlir_common.hpp>
#include <string>

namespace miopen {
namespace solver {
namespace mlir {

std::string InsertGToLayout(const std::string& layout, char dim)
{
    std::string layout_with_g = layout;
    std::size_t index         = layout.find(dim);
    if(index == std::string::npos)
        MIOPEN_THROW(std::string("Failed to find dim '") + dim + "' in the layout " + layout);
    return layout_with_g.insert(index, 1, 'G');
}

const char* DTypeName(miopenDataType_t ty) {
    switch (ty) {
    case miopenHalf:
        return "fp16";
    case miopenFloat:
        return "fp32";
    case miopenDouble:
        return "fp64";
    case miopenBFloat16:
        return "bf16";
    case miopenInt32:
        return "i32";
    case miopenInt8:
        return "i8";
    case miopenInt8x4:
        return "i8x4";
    }
    assert(false); // All cases should be covered before here
}

/* Construct the options string passed to MLIR to cause it
to generate a given convolution.

Returns an empty string on unsupported convolutions */
std::string ConstructBuildOptions(const ConvolutionContext& ctx,
                                  const std::string& operation,
                                  const std::string& kernel_name,
                                  bool is_xdlops,
                                  int kernel_id)
{
    // Arguments for mlir-miopen-driver.
    // clang-format off
    using CI = ConvolutionContextInterpreter;

    std::string in_layout = InsertGToLayout(CI::GetInputLayout(ctx), 'C');
    std::string fil_layout = InsertGToLayout(CI::GetFilterLayout(ctx), 'N');
    std::string out_layout = InsertGToLayout(CI::GetOutputLayout(ctx), 'C');

    std::string mlir_handle;
    if (!ctx.Is2d()) {
        // Future: Remove this once MLIr supports 3D convolutions
        return mlir_handle;
    }

    if (is_xdlops)
        mlir_handle += std::string(" --x2 1");

    mlir_handle +=
        std::string(" --operation ") + operation +
        std::string(" --kernel_id ") + std::to_string(kernel_id) +
        std::string(" --num_cu ") + std::to_string(ctx.GetStream().GetMaxComputeUnits()) +
        std::string(" --arch ") + ctx.GetStream().GetDeviceName() +
        std::string(" --groupsize ") + std::to_string(CI::GetGroupCountG(ctx)) +
        std::string(" --fil_layout ") + fil_layout +
        std::string(" --fil_type ") + DTypeName(ctx.weights_data_type) +
        std::string(" --in_layout ") + in_layout +
        std::string(" --in_type ") + DTypeName(CI::GetInputDataType(ctx)) +
        std::string(" --out_layout ") + out_layout +
        std::string(" --out_type ") + DTypeName(CI::GetOutputDataType(ctx)) +
        std::string(" --batchsize ") + std::to_string(CI::GetBatchN(ctx)) +
        std::string(" --in_channels ") + std::to_string(CI::GetInputChannelC(ctx)) +
        std::string(" --out_channels ") + std::to_string(CI::GetOutputChannelK(ctx)) +
        std::string(" --in_h ") + std::to_string(CI::GetInputHeightHi(ctx)) +
        std::string(" --in_w ") + std::to_string(CI::GetInputWidthWi(ctx)) +
        std::string(" --out_h ") + std::to_string(CI::GetOutputHeightHo(ctx)) +
        std::string(" --out_w ") + std::to_string(CI::GetOutputWidthWo(ctx)) +
        std::string(" --fil_h ") + std::to_string(CI::GetFilterHeightY(ctx)) +
        std::string(" --fil_w ") + std::to_string(CI::GetFilterWidthX(ctx)) +
        std::string(" --dilation_h ") + std::to_string(CI::GetAdjustedConvolutionDilationH(ctx)) +
        std::string(" --dilation_w ") + std::to_string(CI::GetAdjustedConvolutionDilationW(ctx)) +
        std::string(" --conv_stride_h ") + std::to_string(CI::GetAdjustedConvolutionStrideH(ctx)) +
        std::string(" --conv_stride_w ") + std::to_string(CI::GetAdjustedConvolutionStrideW(ctx)) +
        std::string(" --padding_h ") + std::to_string(CI::GetInputLeftPadH(ctx)) +
        std::string(" --padding_w ") + std::to_string(CI::GetInputLeftPadW(ctx)) +
        std::string(" --kernel_name ") + kernel_name;
    // clang-format on
    return mlir_handle;
}

} // namespace mlir
} // namespace solver
} // namespace miopen
