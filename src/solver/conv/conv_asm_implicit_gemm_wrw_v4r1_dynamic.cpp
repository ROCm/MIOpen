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
#include <miopen/conv/solvers.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/gcn_asm_utils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

// 3 possible configs:
//{  16, 128,  16,   2,   4,   4,   4,   4,   4,   4,  16,   1,  16,   1,    4,  64},
//{  16, 128,  16,   2,   4,   4,   4,   4,   4,   4,  16,   1,  16,   1,   16,  16},
//{   8,  32,   4,   2,   2,   2,   2,   4,   4,   2,   4,   2,   8,   1,    4,  16}

static inline int GetImplicitGemmWrwV4R1DynamicGemmkGroups(const ProblemDescription& problem,
                                                           const int& GemmKPerBlock)
{
    int n            = problem.GetInBatchSize();
    int ho           = problem.GetInHeight();
    int wo           = problem.GetInWidth();
    int gemmk        = n * ho * wo;
    int gemmk_groups = 1;
    int n_per_group;
    for(int i = 0; i < 6; i++)
    {
        if(0 == n % (1 << i))
        {
            n_per_group = n >> i;
            if(0 == (gemmk % (n_per_group * GemmKPerBlock)))
                gemmk_groups = i;
            else
                break;
        }
        else
            break;
    }
    // gemmk_groups = 0;
    return gemmk_groups;
}

static inline float CallImplicitGemmWrwDynamic(const miopen::Handle& handle,
                                               const ProblemDescription& problem,
                                               ConstData_t src,
                                               ConstData_t dst,
                                               Data_t wei,
                                               Data_t wei_workspace,
                                               const std::vector<KernelInvoke>& kernels)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    // clang-format off
    int hi           = problem.GetOutHeight();
    int wi           = problem.GetOutWidth();
    int n            = problem.GetOutChannels();
    int k            = problem.GetInChannels();
    int c            = problem.GetInBatchSize();
    int ho           = problem.GetWeightsHeight();
    int wo           = problem.GetWeightsWidth();
    int dilation_h   = problem.GetInHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int dilation_w   = problem.GetInWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int stride_h     = problem.GetWeightsHeight() > 1? problem.GetDilationH() : 1;
    int stride_w     = problem.GetWeightsWidth() > 1? problem.GetDilationW() : 1;
    int pad_h        = problem.GetPadH();
    int pad_w        = problem.GetPadW();
    int y            = problem.GetInHeight();
    int x            = problem.GetInWidth();
    int gemmk_groups = 0;
    int GemmKPerBlock;

    if((k % 128 == 0) && ((n * ho * wo) % 128 == 0))
        GemmKPerBlock = 16;
    else
        GemmKPerBlock = 4;

    gemmk_groups = GetImplicitGemmWrwV4R1DynamicGemmkGroups(problem, GemmKPerBlock);

    MIOPEN_LOG_I2(kernel.GetName() << " with groups for reduction: " << (1 << gemmk_groups) << " GemmKPerBlock: " << GemmKPerBlock);

    // clang-format on
    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(dst);
    if(gemmk_groups > 0)
        opArgs.emplace_back(wei_workspace);
    else
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

    // reduction section
    if(gemmk_groups > 0)
    {
        auto kernel_reduction = kernels[1];
        int reduction_groups  = 1 << gemmk_groups;
        MIOPEN_LOG_I(kernel_reduction.GetName() << " with groups: " << reduction_groups);
        std::vector<OpKernelArg> opArgs_reduction;
        int reduction_per_thread = 4;
        int in_stride            = n * k * ho * wo;
        opArgs_reduction.emplace_back(wei);
        opArgs_reduction.emplace_back(wei_workspace);
        opArgs_reduction.emplace_back(reduction_per_thread);
        opArgs_reduction.emplace_back(in_stride);
        opArgs_reduction.emplace_back(reduction_groups);
        kernel_reduction(opArgs_reduction);
        if(handle.IsProfilingEnabled())
            elapsed += handle.GetKernelTime();
    }

    return elapsed;
}

// find wrw dynamic kernel by a simple algo
// check wether this kernel can be applicable
static inline bool FindImplicitGemmWrwV4R1DynamicKernel(const ProblemDescription& problem,
                                                        std::string& kernel_name,
                                                        int& block_size,
                                                        int& grid_size)
{
    int n     = problem.GetBatchSize();
    int k     = problem.GetInChannels();
    int c     = problem.GetOutChannels();
    int ho    = problem.GetInHeight();
    int wo    = problem.GetInWidth();
    int y     = problem.GetWeightsHeight();
    int x     = problem.GetWeightsWidth();
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

        int log2_gemmk_groups = GetImplicitGemmWrwV4R1DynamicGemmkGroups(problem, GemmKPerBlock);
        GemmKGroups           = 1 << log2_gemmk_groups;
        if(GemmK % (GemmKGroups * GemmKPerBlock) != 0)
            return false;

        block_size = 256;
        grid_size  = (GemmM / GemmMPerBlock) *
                    (GemmN / (GemmNRepeat * GemmNPerThreadSubC * GemmN0YXPerBlock)) * GemmKGroups;

        if((ho * wo) % 16 == 0)
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

        int log2_gemmk_groups = GetImplicitGemmWrwV4R1DynamicGemmkGroups(problem, GemmKPerBlock);
        GemmKGroups           = 1 << log2_gemmk_groups;
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

size_t ConvAsmImplicitGemmV4R1DynamicWrw::GetWorkspaceSize(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    int k            = problem.GetInChannels();
    int c            = problem.GetOutChannels();
    int y            = problem.GetWeightsHeight();
    int x            = problem.GetWeightsWidth();
    int ele_size     = 0;
    int gemmk_groups = 0;
    int extra_groups = 0;
    int GemmKPerBlock;
    int GemmN = c * y * x;

    if((k % 128 == 0) && (GemmN % 128 == 0))
        GemmKPerBlock = 16;
    else
        GemmKPerBlock = 4;

    if(problem.IsFp32())
        ele_size = sizeof(float);
    else
        ele_size = 2;

    gemmk_groups = GetImplicitGemmWrwV4R1DynamicGemmkGroups(problem, GemmKPerBlock);

    if(gemmk_groups == 0)
        extra_groups = 0;
    else
        extra_groups = 1 << gemmk_groups;
    return static_cast<std::size_t>(k) * c * y * x * ele_size * extra_groups;
}

static int GetGemmkGroups(const ProblemDescription& problem)
{
    const int k      = problem.GetInChannels();
    const int c      = problem.GetOutChannels();
    const int y      = problem.GetWeightsHeight();
    const int x      = problem.GetWeightsWidth();
    const auto GemmN = c * y * x;

    int GemmKPerBlock = 4;
    if((k % 128 == 0) && (GemmN % 128 == 0))
        GemmKPerBlock = 16;

    return GetImplicitGemmWrwV4R1DynamicGemmkGroups(problem, GemmKPerBlock);
}

bool ConvAsmImplicitGemmV4R1DynamicWrw::IsApplicable(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1))
        return false;

    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx900") || StartsWith(device_name, "gfx906")))
        return false;

    if(!ctx.use_asm_kernels)
        return false;

    if(GetGemmkGroups(problem) > 0) // GetSolution() adds HIP kernels in this case.
    {
        if(!ctx.use_hip_kernels)
            return false;
    }

    if(!problem.IsDirectionBackwardWrW())
        return false;

    if(!problem.Is2d())
        return false;

    if(!problem.IsFp32())
        return false;

    if(problem.HasNonPackedTensors())
        return false;

    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    if(problem.IsTensorsCasted())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(problem.GetGroupCount() != 1)
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;
    std::string kernel_name;
    int block_size;
    int grid_size;

    return FindImplicitGemmWrwV4R1DynamicKernel(problem, kernel_name, block_size, grid_size);
}

ConvSolution ConvAsmImplicitGemmV4R1DynamicWrw::GetSolution(const ExecutionContext& ctx,
                                                            const ProblemDescription& problem) const
{
    ConvSolution result;

    KernelInfo kernel;
    std::ostringstream options;

    int block_size;
    int grid_size;
    std::string kernel_name;
    bool ret = FindImplicitGemmWrwV4R1DynamicKernel(problem, kernel_name, block_size, grid_size);

    if(!ret)
        MIOPEN_THROW("this kernel should not run with igemm dynamic!");

    result.workspace_sz = GetWorkspaceSize(ctx, problem);

    kernel.kernel_file = "igemm_v4r1_wrw_dynamic.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
     * grid dims is in unit of work item.
     * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
     */
    kernel.g_wk.push_back(static_cast<std::size_t>(grid_size) * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    kernel.comp_options = options.str();

    MIOPEN_LOG_I2(kernel.kernel_file << ":" << kernel.kernel_name);

    result.construction_params.push_back(kernel);

    if(GetGemmkGroups(problem) > 0)
    {
        KernelInfo kernel_reduction;
        int reduction_per_thread     = 4;
        kernel_reduction.kernel_file = "wrw_reduction_hip.cpp";
        kernel_reduction.kernel_name = "wrw_reduction_hip";
        kernel_reduction.g_wk.clear();
        int block_size_reduction = 256;
        int grid_size_redcution =
            problem.GetOutChannels() * problem.GetInChannels() * problem.GetWeightsHeight() *
            problem.GetWeightsWidth() /
            static_cast<std::size_t>(reduction_per_thread * block_size_reduction);
        kernel_reduction.g_wk.push_back(static_cast<std::size_t>(grid_size_redcution) *
                                        block_size_reduction);
        kernel_reduction.g_wk.push_back(1);
        kernel_reduction.g_wk.push_back(1);
        kernel_reduction.l_wk.clear();
        kernel_reduction.l_wk.push_back(block_size_reduction);
        kernel_reduction.l_wk.push_back(1);
        kernel_reduction.l_wk.push_back(1);

        kernel_reduction.comp_options =
            std::string("-Wno-old-style-cast ") + std::string("-Wno-cast-align");

        result.construction_params.push_back(kernel_reduction);
    }

    result.invoker_factory = [problem](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            MIOPEN_LOG_I("wrw workspace size: " << data_ctx.workSpaceSize);
            const auto& workSpace = data_ctx.workSpace;
            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k_wrw) { return handle.Run(k_wrw); });
            float elapsed = 0;
            elapsed       = CallImplicitGemmWrwDynamic(
                handle, problem, tensors.x, tensors.dy, tensors.dw, workSpace, ks);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
