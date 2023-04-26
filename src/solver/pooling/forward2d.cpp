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

#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

namespace solver {

namespace pooling {

namespace {

int inline get_kernel_size_h(const miopen::pooling::ProblemDescription& p)
{
    return p.GetPooling().lens[0];
}

int inline get_kernel_size_w(const miopen::pooling::ProblemDescription& p)
{
    return p.GetPooling().lens[1];
}

int inline get_kernel_stride_h(const miopen::pooling::ProblemDescription& p)
{
    return p.GetPooling().strides[0];
}

int inline get_kernel_stride_w(const miopen::pooling::ProblemDescription& p)
{
    return p.GetPooling().strides[1];
}

int inline get_out_height(const miopen::pooling::ProblemDescription& p)
{
    return p.GetYDesc().GetLengths()[2];
}

int inline get_out_width(const miopen::pooling::ProblemDescription& p)
{
    return p.GetYDesc().GetLengths()[3];
}

int inline get_out_pix_tile0(const miopen::pooling::ProblemDescription&) { return 1; }

int inline get_out_pix_tile1(const miopen::pooling::ProblemDescription& p)
{
    int rv =                          //
        get_out_height(p) <= 8    ? 1 //
        : get_out_height(p) <= 32 ? 4 //
                                  : 8;
    if(get_out_height(p) > 16 && get_out_height(p) % 32 > 16)
        rv = std::min(16, std::max(1, prePow2(rv * get_kernel_stride_h(p))));
    return rv;
}
} // namespace

bool PoolingForward2d::IsApplicable(const ExecutionContext&,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Forward &&
           problem.GetXDesc().GetSize() == 4 &&
           problem.GetXDesc().GetType() == problem.GetYDesc().GetType() &&
           (problem.GetXDesc().GetType() == miopenFloat ||
            problem.GetXDesc().GetType() == miopenHalf) &&
           problem.GetXDesc().GetLayout("NCHW") == "NCHW" &&
           problem.GetYDesc().GetLayout("NCHW") == "NCHW";
}

ConvSolution PoolingForward2d::GetSolution(const ExecutionContext&,
                                           const miopen::pooling::ProblemDescription& p) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPooling.cl";
        kernel.kernel_name = "mloPoolingG";

        int batch_sz, n_outputs;
        std::tie(batch_sz, n_outputs, std::ignore, std::ignore) =
            miopen::tien<4>(p.GetYDesc().GetLengths(), 1);

        const auto wsp_index = p.GetPooling().GetWorkspaceIndexMode();

        int grp_tile0 = get_out_width(p) <= 8 ? 8 : (get_out_width(p) % 32 <= 16 ? 16 : 32);
        int grp_tile1 = get_out_height(p) <= 8    ? 8
                        : get_out_height(p) < 16  ? 16
                        : get_out_height(p) <= 32 ? 32
                        : get_out_height(p) <= 64 ? 64
                                                  : 128;
        grp_tile1 /= get_out_pix_tile1(p);
        while(grp_tile0 * grp_tile1 > 256 && grp_tile0 > 1)
            grp_tile0 >>= 1;

        int pooling_method = (p.GetPooling().GetMode() == miopenPoolingMax)
                                 ? MLO_POOLING_OP_MAX
                                 : ((p.GetPooling().GetMode() == miopenPoolingAverage)
                                        ? MLO_POOLING_OP_AVE
                                        : MLO_POOLING_OP_AVE_INCLUSIVE);

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", pooling_method},
            {"MLO_POOLING_KERNEL_SZ1", get_kernel_size_h(p)},
            {"MLO_POOLING_STRIDE1", get_kernel_stride_h(p)},
            {"MLO_POOLING_KERNEL_SZ0", get_kernel_size_w(p)},
            {"MLO_POOLING_STRIDE0", get_kernel_stride_w(p)},
            {"MLO_POOLING_N_HORIZ_OUT_PIX", get_out_pix_tile0(p)},
            {"MLO_POOLING_N_VERT_OUT_PIX", get_out_pix_tile1(p)},
            {"MLO_POOLING_GROUP_SZ0", grp_tile0},
            {"MLO_POOLING_GROUP_SZ1", grp_tile1},
            {"MLO_POOLING_INDEX_TYPE", get_pooling_index_type_name(p.GetPooling().GetIndexType())},
            {"MLO_POOLING_INDEX_MAX",
             get_pooling_index_type_max_name(p.GetPooling().GetIndexType())},
        };

        if(p.SaveIndex())
        {
            build_params << KernelBuildParameters{
                {"MLO_POOLING_SAVE_INDEX"},
                {"USE_IMG_INDEX", (wsp_index == miopenPoolingWorkspaceIndexImage ? 1 : 0)},
            };
        }

        build_params << GetDataTypeKBP(p.GetXDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk.push_back(grp_tile0);
        kernel.l_wk.push_back(grp_tile1);
        kernel.l_wk.push_back(1);

        int g_wk_width  = ((get_out_width(p) + grp_tile0 * get_out_pix_tile0(p) - 1) /
                          (grp_tile0 * get_out_pix_tile0(p)));
        int g_wk_height = ((get_out_height(p) + grp_tile1 * get_out_pix_tile1(p) - 1) /
                           (grp_tile1 * get_out_pix_tile1(p)));

        kernel.g_wk.push_back(static_cast<std::size_t>(g_wk_width) * grp_tile0);
        kernel.g_wk.push_back(static_cast<std::size_t>(g_wk_height) * grp_tile1);
        kernel.g_wk.push_back(static_cast<std::size_t>(n_outputs) * batch_sz);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();

            kernel(params.x,
                   params.y,
                   params.workspace,
                   static_cast<int>(params.pooling.pads[0]),
                   static_cast<int>(params.pooling.pads[1]),
                   static_cast<int>(params.xDesc.GetLengths()[1]),
                   static_cast<int>(params.xDesc.GetLengths()[2]),
                   static_cast<int>(params.xDesc.GetLengths()[3]),
                   static_cast<int>(params.yDesc.GetLengths()[2]),
                   static_cast<int>(params.yDesc.GetLengths()[3]),
                   static_cast<int>(params.xDesc.GetStrides()[0]),
                   static_cast<int>(params.xDesc.GetStrides()[1]),
                   static_cast<int>(params.xDesc.GetStrides()[2]),
                   static_cast<int>(params.yDesc.GetStrides()[0]),
                   static_cast<int>(params.yDesc.GetStrides()[1]),
                   static_cast<int>(params.yDesc.GetStrides()[2]));
        };
    };

    return result;
}

std::size_t
PoolingForward2d::GetWorkspaceSize(const ExecutionContext&,
                                   const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
