/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <cstddef>
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include "implicitgemm_util.hpp"
#include <miopen/gcn_asm_utils.hpp>

namespace miopen {
namespace solver {

// 3 possible configs:
//{  16, 128,  16,   2,   4,   4,   4,   4,   4,   4,  16,   1,  16,   1,    4,  64},
//{  16, 128,  16,   2,   4,   4,   4,   4,   4,   4,  16,   1,  16,   1,   16,  16},
//{   8,  32,   4,   2,   2,   2,   2,   4,   4,   2,   4,   2,   8,   1,    4,  16}

static inline std::vector<string>& GetImplicitGemmWrwGTCDynamicXdlopsKernelNameList()
{
    // retrieve dynamic igemm wrw pass's possible kernel name
    // clang-format off
    static const std::vector<string> kernel_name_list = {
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt128x128x16_wt32x32_ws1x1_wr2x2_ta2x1x4x1_1x8x1x32_tb2x1x4x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx4_ex0_bt128x128x16_wt32x32_ws1x1_wr2x2_ta1x4x2x1_1x4x1x64_tb1x4x2x1_1x4x1x64",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt128x128x16_wt32x32_ws1x1_wr2x2_ta2x1x4x1_1x8x1x32_tb2x1x4x1_1x8x1x32_atadd",   
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt128x128x16_wt32x32_ws1x1_wr2x2_ta1x1x8x1_1x16x1x16_tb1x1x8x1_1x16x1x16_atadd", 
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt128x128x16_wt32x32_ws1x1_wr2x2_ta1x1x8x1_1x16x1x16_tb1x1x8x1_1x16x1x16",       
        "igemm_wrw_gtcx_nchw_fp32_bx4_ex0_bt128x128x16_wt32x32_ws1x1_wr2x2_ta1x4x2x1_1x4x1x64_tb1x4x2x1_1x4x1x64_atadd",   
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt128x128x8_wt32x32_ws1x1_wr2x2_ta1x1x4x1_1x8x1x32_tb1x1x4x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt128x128x8_wt32x32_ws1x1_wr2x2_ta1x1x4x1_1x8x1x32_tb1x1x4x1_1x8x1x32_atadd",    
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x128x16_wt64x32_ws1x1_wr2x2_ta1x1x16x1_1x16x1x16_tb1x1x8x1_1x16x1x16",      
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x128x8_wt64x32_ws1x1_wr2x2_ta1x1x8x1_1x8x1x32_tb1x1x4x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x128x16_wt64x32_ws1x1_wr2x2_ta2x1x8x1_1x8x1x32_tb2x1x4x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x128x16_wt64x32_ws1x1_wr2x2_ta1x1x16x1_1x16x1x16_tb1x1x8x1_1x16x1x16_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x64x4_wt64x16_ws1x1_wr2x2_ta1x1x4x1_1x4x1x64_tb1x1x1x1_1x4x1x64_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x64x4_wt64x16_ws1x1_wr2x2_ta1x1x4x1_1x4x1x64_tb1x1x1x1_1x4x1x64",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x64x8_wt64x16_ws1x1_wr2x2_ta1x1x8x1_1x8x1x32_tb1x1x2x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x64x8_wt64x16_ws1x1_wr2x2_ta1x1x8x1_1x8x1x32_tb1x1x2x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x64x16_wt64x16_ws1x1_wr2x2_ta1x1x16x1_1x16x1x16_tb1x1x4x1_1x16x1x16_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx4_ex0_bt256x64x16_wt64x16_ws1x1_wr2x2_ta1x4x4x1_1x4x1x64_tb1x4x1x1_1x4x1x64_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x64x16_wt64x16_ws1x1_wr2x2_ta1x1x16x1_1x16x1x16_tb1x1x4x1_1x16x1x16",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x64x16_wt16x16_ws1x1_wr2x2_ta2x1x2x1_1x8x1x32_tb2x1x2x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x64x16_wt16x16_ws1x1_wr2x2_ta1x1x4x1_1x16x1x16_tb1x1x4x1_1x16x1x16_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x64x8_wt16x16_ws1x1_wr2x2_ta1x1x2x1_1x8x1x32_tb1x1x2x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x64x16_wt16x16_ws1x1_wr2x2_ta2x1x2x1_1x8x1x32_tb2x1x2x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x64x16_wt16x16_ws1x1_wr2x2_ta1x1x4x1_1x16x1x16_tb1x1x4x1_1x16x1x16",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x64x8_wt16x16_ws1x1_wr2x2_ta1x1x2x1_1x8x1x32_tb1x1x2x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt4x64x16_wt4x64_ws1x1_wr1x1_ta1x1x1x1_1x16x1x4_tb1x1x16x1_1x16x1x4_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt4x64x16_wt4x64_ws1x1_wr1x1_ta1x1x1x1_1x16x1x4_tb1x1x16x1_1x16x1x4",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt32x32x8_wt16x16_ws1x1_wr1x1_ta1x1x1x1_1x8x1x32_tb1x1x1x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt32x32x8_wt16x16_ws1x1_wr1x1_ta1x1x1x1_1x8x1x32_tb1x1x1x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x32x8_wt32x8_ws1x2_wr1x1_ta1x1x2x1_1x8x1x32_tb1x1x1x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x32x8_wt32x8_ws1x2_wr1x1_ta1x1x2x1_1x8x1x32_tb1x1x1x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x32x16_wt32x8_ws1x2_wr1x1_ta1x1x4x1_1x16x1x16_tb1x1x2x1_1x16x1x16",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x32x16_wt32x8_ws1x2_wr1x1_ta1x1x4x1_1x16x1x16_tb1x1x2x1_1x16x1x16_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x32x16_wt64x4_ws1x2_wr2x2_ta1x1x16x1_1x16x1x16_tb1x1x2x1_1x16x1x16",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x32x16_wt64x4_ws1x2_wr2x2_ta1x1x16x1_1x16x1x16_tb1x1x2x1_1x16x1x16_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x32x8_wt64x4_ws1x2_wr2x2_ta1x1x8x1_1x8x1x32_tb1x1x1x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt256x32x8_wt64x4_ws1x2_wr2x2_ta1x1x8x1_1x8x1x32_tb1x1x1x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt16x32x16_wt8x32_ws1x1_wr1x1_ta1x1x2x1_1x16x1x8_tb1x1x4x1_1x16x1x8",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt16x32x16_wt8x32_ws1x1_wr1x1_ta1x1x2x1_1x16x1x8_tb1x1x4x1_1x16x1x8_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt16x32x8_wt8x32_ws1x1_wr1x1_ta1x1x1x1_1x8x1x16_tb1x1x2x1_1x8x1x16",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt16x32x8_wt8x32_ws1x1_wr1x1_ta1x1x1x1_1x8x1x16_tb1x1x2x1_1x8x1x16_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x256x8_wt16x64_ws1x1_wr2x2_ta1x1x2x1_1x8x1x32_tb1x1x8x1_1x8x1x32_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x256x8_wt16x64_ws1x1_wr2x2_ta1x1x2x1_1x8x1x32_tb1x1x8x1_1x8x1x32",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x256x16_wt16x64_ws1x1_wr2x2_ta1x1x4x1_1x16x1x16_tb1x1x16x1_1x16x1x16_atadd",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x256x16_wt16x64_ws1x1_wr2x2_ta1x1x4x1_1x16x1x16_tb1x1x16x1_1x16x1x16",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x16x16_wt64x4_ws1x1_wr1x1_ta1x1x4x1_1x16x1x16_tb1x1x1x1_1x16x1x16",
        "igemm_wrw_gtcx_nchw_fp32_bx1_ex1_bt64x16x16_wt64x4_ws1x1_wr1x1_ta1x1x4x1_1x16x1x16_tb1x1x1x1_1x16x1x16_atadd"
    };
    return kernel_name_list;
}

//static inline 

static inline int
GetImplicitGemmWrwGTCDynamicXdlopsGemmkSplits(const conv::ProblemDescription& conv_problem,
                                              const int& GemmKPerBlock)
{
    int n            = conv_problem.GetInBatchSize();
    int ho           = conv_problem.GetInHeight();
    int wo           = conv_problem.GetInWidth();
    int gemmk        = n * ho * wo;
    int gemmk_splits = 1;
    int n_per_group;
    for(int i = 0; i < 6; i++)
    {
        if(0 == n % (1 << i))
        {
            n_per_group = n >> i;
            if(0 == (gemmk % (n_per_group * GemmKPerBlock)))
                gemmk_splits = i;
            else
                break;
        }
        else
            break;
    }
    // gemmk_splits = 0;
    return gemmk_splits;
}

static inline float CallImplicitGemmWrwDynamic(const miopen::Handle& handle,
                                               const conv::ProblemDescription& conv_problem,
                                               ConstData_t src,
                                               ConstData_t dst,
                                               Data_t wei,
                                               const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    // clang-format off
    int hi           = conv_problem.GetOutHeight();
    int wi           = conv_problem.GetOutWidth();
    int n            = conv_problem.GetOutChannels();
    int k            = conv_problem.GetInChannels();
    int c            = conv_problem.GetInBatchSize();
    int ho           = conv_problem.GetWeightsHeight();
    int wo           = conv_problem.GetWeightsWidth();
    int dilation_h   = conv_problem.GetInHeight() > 1 ? conv_problem.GetKernelStrideH() : 1;
    int dilation_w   = conv_problem.GetInWidth() > 1 ? conv_problem.GetKernelStrideW() : 1;
    int stride_h     = conv_problem.GetWeightsHeight() > 1? conv_problem.GetDilationH() : 1;
    int stride_w     = conv_problem.GetWeightsWidth() > 1? conv_problem.GetDilationW() : 1;
    int pad_h        = conv_problem.GetPadH();
    int pad_w        = conv_problem.GetPadW();
    int y            = conv_problem.GetInHeight();
    int x            = conv_problem.GetInWidth();
    int gemmk_groups = 0;
    int GemmKPerBlock;

    if((k % 128 == 0) && ((n * ho * wo) % 128 == 0))
        GemmKPerBlock = 16;
    else
        GemmKPerBlock = 4;

    gemmk_groups = GetImplicitGemmWrwGTCDynamicXdlopsGemmkGroups(conv_problem, GemmKPerBlock);

    MIOPEN_LOG_I2(kernel.GetName() << " with groups for reduction: " << (1 << gemmk_groups) << " GemmKPerBlock: " << GemmKPerBlock);

    // clang-format on
    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(dst);
    opArgs.emplace_back(wei);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(gemmk_groups);
    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed += handle.GetKernelTime();

    return elapsed;
}

// find wrw dynamic kernel by a simple algo
// check wether this kernel can be applicable
static inline bool FindImplicitGemmWrwGTCDynamicXdlopsKernel(const ConvolutionContext& ctx,
                                                        std::string& kernel_name,
                                                        int& block_size,
                                                        int& grid_size)
{
    int n     = ctx.batch_sz;
    int k     = ctx.n_inputs;
    int c     = ctx.n_outputs;
    int ho    = ctx.in_height;
    int wo    = ctx.in_width;
    int y     = ctx.kernel_size_h;
    int x     = ctx.kernel_size_w;
    int GemmN = c * y * x;
    int GemmM = k;
    int GemmK = n * ho * wo;
    int GemmNRepeat;
    int GemmNPerThreadSubC;
    int GemmN0YXPerBlock;
    int GemmMPerBlock;
    int GemmKPerBlock;
    int GemmKGroups;

    if((GemmM % 128 == 0) && (GemmN % 128 == 0))
    {
        GemmNRepeat        = 2;
        GemmNPerThreadSubC = 4;
        GemmN0YXPerBlock   = 16;
        GemmMPerBlock      = 128;
        GemmKPerBlock      = 16;

        if(c % (GemmNRepeat * GemmNPerThreadSubC) != 0)
            return false;
        if(GemmN % (GemmNRepeat * GemmNPerThreadSubC * GemmN0YXPerBlock) != 0)
            return false;
        if(GemmM % GemmMPerBlock != 0)
            return false;

        int log2_gemmk_groups =
            GetImplicitGemmWrwGTCDynamicXdlopsGemmkSplits(ctx.conv_problem, GemmKPerBlock);
        GemmKGroups = 1 << log2_gemmk_groups;
        if(GemmK % (GemmKGroups * GemmKPerBlock) != 0)
            return false;

        block_size = 256;
        grid_size  = (GemmM / GemmMPerBlock) *
                    (GemmN / (GemmNRepeat * GemmNPerThreadSubC * GemmN0YXPerBlock)) * GemmKGroups;

        if((ho * wo) % 4 == 0)
            kernel_name = "igemm_v4r1_dynamic_wrw_128x128x16_8x8_4x4x4x4x4x4_16x1x16x1_4x64";
        else
            kernel_name = "igemm_v4r1_dynamic_wrw_128x128x16_8x8_4x4x4x4x4x4_16x1x16x1_16x16";

        return true;
    }
    else if((GemmM % 32 == 0) && (GemmN % 32 == 0))
    {
        GemmNRepeat        = 2;
        GemmNPerThreadSubC = 2;
        GemmN0YXPerBlock   = 8;
        GemmMPerBlock      = 32;
        GemmKPerBlock      = 4;

        if(c % (GemmNRepeat * GemmNPerThreadSubC) != 0)
            return false;
        if(GemmN % (GemmNRepeat * GemmNPerThreadSubC * GemmN0YXPerBlock) != 0)
            return false;
        if(GemmM % GemmMPerBlock != 0)
            return false;

        int log2_gemmk_groups =
            GetImplicitGemmWrwGTCDynamicXdlopsGemmkSplits(ctx.conv_problem, GemmKPerBlock);
        GemmKGroups = 1 << log2_gemmk_groups;
        if(GemmK % (GemmKGroups * GemmKPerBlock) != 0)
            return false;

        block_size = 64;
        grid_size  = (GemmM / GemmMPerBlock) *
                    (GemmN / (GemmNRepeat * GemmNPerThreadSubC * GemmN0YXPerBlock)) * GemmKGroups;

        kernel_name = "igemm_v4r1_dynamic_wrw_32x32x4_4x4_2x2x4x2x4x2_4x2x8x1_4x16";

        return true;
    }
    else
        return false;
}

bool ConvAsmImplicitGemmGTCDynamicWrwXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx908")))
        return false;

    if(!IsApplicableXdlops(ctx))
        return false;

    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(ctx.group_counts != 1)
        return false;

    std::string kernel_name;
    int block_size;
    int grid_size;

    return FindImplicitGemmWrwGTCDynamicXdlopsKernel(ctx, kernel_name, block_size, grid_size);
}

ConvSolution ConvAsmImplicitGemmGTCDynamicWrwXdlops::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;

    KernelInfo kernel;
    std::ostringstream options;

    int block_size;
    int grid_size;
    std::string kernel_name;
    bool ret = FindImplicitGemmWrwGTCDynamicXdlopsKernel(ctx, kernel_name, block_size, grid_size);

    if(!ret)
        MIOPEN_THROW("this kernel should not run with igemm dynamic!");

    int k = ctx.n_inputs;
    int c = ctx.n_outputs;
    int y = ctx.kernel_size_h;
    int x = ctx.kernel_size_w;
    int GemmKPerBlock;
    int GemmN = c * y * x;

    if((k % 128 == 0) && (GemmN % 128 == 0))
        GemmKPerBlock = 16;
    else
        GemmKPerBlock = 4;
    int gemmk_groups  = GetImplicitGemmWrwGTCDynamicXdlopsGemmkSplits(ctx.conv_problem, GemmKPerBlock);

    result.workspce_sz = GetWorkspaceSize(ctx);

    kernel.kernel_file = "igemm_wrw_gtc_gfx908.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
    * grid dims is in unit of work item.
    * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
    */
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    kernel.comp_options = options.str();

    MIOPEN_LOG_I2(kernel.kernel_file + ":" + kernel.kernel_name);

    result.construction_params.push_back(kernel);

    const auto& conv_problem = ctx.conv_problem;

    result.invoker_factory = [conv_problem](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const boost::any& primitive_parameters) {
            const auto data_ctx = boost::any_cast<conv::WrWInvokeParams>(primitive_parameters);
            const auto& tensors = data_ctx.tensors;
            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k_wrw) { return handle.Run(k_wrw); });
            float elapsed = 0;
            elapsed       = CallImplicitGemmWrwDynamic(
                handle, conv_problem, tensors.x, tensors.dy, tensors.dw, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    return result;
}

} // namespace solver
} // namespace miopen
