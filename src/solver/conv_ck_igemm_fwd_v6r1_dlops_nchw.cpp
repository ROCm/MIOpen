/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include <miopen/conv/invokers/impl_gemm.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

#include <cstddef>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW)

namespace miopen {
namespace solver {

PerformanceConvCkIgemmFwdV6r1DlopsNchw::PerformanceConvCkIgemmFwdV6r1DlopsNchw(
    int a0,
    int a1,
    int a2,
    int a3,
    int a4,
    int a5,
    int a6,
    int a7,
    int a8,
    int a9,
    int a10,
    int a11,
    int a12,
    std::initializer_list<int> a13,
    std::initializer_list<int> a14,
    std::initializer_list<int> a15,
    std::initializer_list<int> a16,
    std::initializer_list<int> a17,
    std::initializer_list<int> a18,
    std::initializer_list<int> a19,
    std::initializer_list<int> a20,
    int a21
    )
    : BlockSize{a0},
      GN0{a1},
      GK1{a2},
      GM1PerBlockGM11{a3},
      GN1PerBlockGN11{a4},
      GK0PerBlock{a5},
      BM1PerThreadBM11{a6},
      BN1PerThreadBN11{a7},
      BK0PerThread{a8},
      BM10BN10ThreadClusterBM100{a9},
      BM10BN10ThreadClusterBN100{a10},
      BM10BN10ThreadClusterBM101{a11},
      BM10BN10ThreadClusterBN101{a12},
      ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1{a13},
      ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1{a14},
      ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1{a15},
      ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1{a16},
      BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1{a17},
      BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1{a18},
      BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1{a19},
      BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1{a20},
      CThreadTransferDstScalarPerVector{a21}
{
}

std::tuple<int, bool>
PerformanceConvCkIgemmFwdV6r1DlopsNchw::CalculateGridSize(const ConvolutionContext& ctx) const
{
    int GridSize = 0;

    const int N  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const int K  = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const int C  = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const int Ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const int Wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const int Y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const int X  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

    if(!(N % GN0 == 0 && C % GK1 == 0))
        return std::make_tuple(0, false);

    const int N0 = GN0;
    const int N1 = N / N0;

    const int C0 = GK1;
    const int C1 = C / C0;

    const int GM0 = 1;
    const int GM1 = K;

    // GN0 is tunable
    const int GN1 =  N1 * Ho * Wo;

    // GK1 is tuanble
    const int GK0 = C1 * Y * X;

    const int GM11 = GM1PerBlockGM11;
    const int GN11 = GN1PerBlockGN11;

    const int GM10 = GM1 / GM11;
    const int GN10 = GN1 / GN11;

    GridSize = GM10 * GN10;

    return std::make_tuple(GridSize, true);
}

bool ConvCkIgemmFwdV6r1DlopsNchw::IsApplicable(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW{}))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.IsLayoutDefault())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.Is2d())
        return false;
    if(!(ctx.IsFp32() or ctx.IsFp16()))
        return false;
    if(ctx.group_counts != 1)
        return false;

    return true;
}

ConvSolution ConvCkIgemmFwdV6r1DlopsNchw::GetSolution(const ConvolutionContext& ctx,
                                                      const bool disableConfigOverrideFromEnv) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    // default config
    PerformanceConvCkIgemmFwdV6r1DlopsNchw config;

    assert(config.IsValid(ctx));

    const int N             = ConvolutionContextInterpreter::GetBatchN(ctx);
    const int K             = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const int C             = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const int Y             = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const int X             = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    const int Hi            = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    const int Wi            = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    const int Ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const int Wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const int ConvStrideH   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const int ConvStrideW   = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);
    const int ConvDilationH = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx);
    const int ConvDilationW = ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx);
    const int InLeftPadH    = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    const int InLeftPadW    = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
    const int InRightPadH   = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
    const int InRightPadW   = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

    const int N0 = config.GN0;
    const int N1 = N / N0;

    const int C0 = config.GK1;
    const int C1 = C / C0;

    const int GM0 = 1;
    const int GM1 = K;

    // GN0 is tunable
    const int GN1 =  N1 * Ho * Wo;

    // GK1 is tuanble
    const int GK0 = C1 * Y * X;

    const int GM11 = config.GM1PerBlockGM11;
    const int GN11 = config.GN1PerBlockGN11;

    const int GM10 = GM1 / GM11;
    const int GN10 = GN1 / GN11;

    const int grid_size = GM10 * GN10;

    construction_parameters.l_wk.push_back(config.BlockSize);
    construction_parameters.l_wk.push_back(1);
    construction_parameters.l_wk.push_back(1);

    construction_parameters.g_wk.push_back(config.BlockSize * grid_size);
    construction_parameters.g_wk.push_back(1);
    construction_parameters.g_wk.push_back(1);

    construction_parameters.kernel_file =
        "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp";

    construction_parameters.kernel_name =
        "dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw_prepare";

    const bool hasMainKBlockLoop = ((GK0 + config.GK0PerBlock) / (2 * config.GK0PerBlock) > 1);
    const bool hasDoubleTailKBlockLoop = ((GK0 / config.GK0PerBlock) % 2 == 0);

    // clang-format off
    construction_parameters.comp_options =
        " -DCK_PARAM_IN_WEI_DATATYPE=" + 
            std::string("70") + 
        " -DCK_PARAM_ACC_DATATYPE=" + 
            std::string("70") + 
        " -DCK_PARAM_OUT_DATATYPE=" + 
            std::string("70") + 
        " -DCK_PARAM_BlockSize=" +
            std::to_string(config.BlockSize) +
        " -DCK_PARAM_GN0=" +
            std::to_string(config.GN0) +
        " -DCK_PARAM_GK1=" +
            std::to_string(config.GK1) +
        " -DCK_PARAM_GM1PerBlockGM11=" +
            std::to_string(config.GM1PerBlockGM11) +
        " -DCK_PARAM_GN1PerBlockGN11=" +
            std::to_string(config.GN1PerBlockGN11) +
        " -DCK_PARAM_GK0PerBlock=" + 
            std::to_string(config.GK0PerBlock) +
        " -DCK_PARAM_BM1PerThreadBM11=" +
            std::to_string(config.BM1PerThreadBM11) +
        " -DCK_PARAM_BN1PerThreadBN11=" +
            std::to_string(config.BN1PerThreadBN11) +
        " -DCK_PARAM_BK0PerThread=" +
            std::to_string(config.BK0PerThread) +
        " -DCK_PARAM_BM10BN10ThreadClusterBM100=" +
            std::to_string(config.BM10BN10ThreadClusterBM100) +
        " -DCK_PARAM_BM10BN10ThreadClusterBN100=" +
            std::to_string(config.BM10BN10ThreadClusterBN100) +
        " -DCK_PARAM_BM10BN10ThreadClusterBM101=" +
            std::to_string(config.BM10BN10ThreadClusterBM101) +
        " -DCK_PARAM_BM10BN10ThreadClusterBN101=" +
            std::to_string(config.BM10BN10ThreadClusterBN101) +
        " -DCK_PARAM_ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1=" +
            std::to_string(config.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
            std::to_string(config.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
            std::to_string(config.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
            std::to_string(config.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
            std::to_string(config.ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1[4]) +
        " -DCK_PARAM_ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1=" +
            std::to_string(config.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
            std::to_string(config.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
            std::to_string(config.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
            std::to_string(config.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
            std::to_string(config.ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1[4]) +
        " -DCK_PARAM_ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1=" +
            std::to_string(config.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) +  "," +
            std::to_string(config.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
            std::to_string(config.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
            std::to_string(config.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
            std::to_string(config.ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]) +
        " -DCK_PARAM_ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1=" +
            std::to_string(config.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[0]) + "," +
            std::to_string(config.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[1]) + "," +
            std::to_string(config.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[2]) + "," +
            std::to_string(config.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[3]) + "," +
            std::to_string(config.ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1[4]) +
        " -DCK_PARAM_BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1=" +
            std::to_string(config.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
            std::to_string(config.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
            std::to_string(config.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
            std::to_string(config.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
            std::to_string(config.BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1[4]) +
        " -DCK_PARAM_BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1=" +
            std::to_string(config.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
            std::to_string(config.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
            std::to_string(config.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
            std::to_string(config.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
            std::to_string(config.BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1[4]) +
        " -DCK_PARAM_BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1=" +
            std::to_string(config.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
            std::to_string(config.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
            std::to_string(config.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
            std::to_string(config.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
            std::to_string(config.BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]) +
        " -DCK_PARAM_BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1=" +
            std::to_string(config.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[0]) + "," +
            std::to_string(config.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[1]) + "," +
            std::to_string(config.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[2]) + "," +
            std::to_string(config.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[3]) + "," +
            std::to_string(config.BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1[4]) +
        " -DCK_PARAM_CThreadTransferDstScalarPerVector=" +
            std::to_string(config.CThreadTransferDstScalarPerVector) +
        " -DCK_PARAM_HAS_MAIN_KBLOCK_LOOP=" + 
            std::to_string(hasMainKBlockLoop) +
        " -DCK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP=" + 
            std::to_string(hasDoubleTailKBlockLoop) +
        get_ck_common_compiler_flag(ctx) +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);

    // workspace is used to save transformed tensor descriptors
    result.workspce_sz = std::size_t(4096L);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& data_ctx = primitive_params.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            auto kernel0         = handle.Run(kernels[0]);
            auto kernel1         = handle.Run(kernels[1]);

            float elapsed = 0;
            kernel0(N,
                    C,
                    Hi,
                    Wi,
                    K,
                    Y,
                    X,
                    ConvStrideH,
                    ConvStrideW,
                    ConvDilationH,
                    ConvDilationW,
                    InLeftPadH,
                    InLeftPadW,
                    InRightPadH,
                    InRightPadW,
                    data_ctx.workSpace);

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
            }

            kernel1(tensors.w, tensors.in, tensors.out, data_ctx.workSpace);

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    return result;
}

} // namespace solver
} // namespace miopen
