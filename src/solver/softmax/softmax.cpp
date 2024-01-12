/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/softmax/solvers.hpp>

#include <miopen/softmax/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/softmax.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/float_equal.hpp>

namespace miopen {

namespace solver {

namespace softmax {

bool SoftmaxForward::IsApplicable(const ExecutionContext& context,
            	              const miopen::softmax::ProblemDescription& problem) const
{
    return true;
}


ConvSolution SoftmaxForward::GetSolution(const ExecutionContext& context,
                                     const miopen::softmax::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto xDesc = problem.GetXDesc();
    auto yDesc = problem.GetYDesc();
    auto alpha = problem.GetAlpha();
    auto beta = problem.GetBeta();
    auto mode = problem.GetMode();
    auto algorithm = problem.GetAlgorithm();

    int n, c, h, w;
    // using workgroup size of 256 by default
    int grid_size, spatial_dim, vector_size, num_batch;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool usefp16, usefp32;

    size_t workgroups;
    int batch_size;
    int u_batch_size;

    int in_nstr, in_cstr, in_hstr;
    std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = tien<4>(xDesc.GetStrides());

    int out_nstr, out_cstr, out_hstr;
    std::tie(out_nstr, out_cstr, out_hstr, std::ignore) = tien<4>(yDesc.GetStrides());

    miopen::softmax::getParams(yDesc, mode, n, c, h, w, grid_size, spatial_dim, vector_size, num_batch, usefp16, usefp32, vld, vgd, workgroups, batch_size, u_batch_size);

    auto alpha_fp = *(static_cast<const float*>(alpha));
    auto beta_fp  = *(static_cast<const float*>(beta));

    KernelBuildParameters build_params = KernelBuildParameters{{"NUM_BATCH", num_batch}};

    if (num_batch > 1)
    {
        build_params.Define("BATCH_SIZE", batch_size);
        build_params.Define("U_BATCH_SIZE", u_batch_size);
    }

    build_params.Define("MIOPEN_USE_FP16", static_cast<int>(usefp16));
    build_params.Define("MIOPEN_USE_FP32", static_cast<int>(usefp32));

    if(algorithm == MIOPEN_SOFTMAX_LOG)
        build_params.Define("USE_SOFTMAX_LOG", 1);
    else if(algorithm == MIOPEN_SOFTMAX_FAST)
        build_params.Define("USE_SOFTMAX_FAST", 1);
    else
        build_params.Define("USE_SOFTMAX_ACCURATE", 1);

    if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        build_params.Define("USE_SOFTMAX_MODE_INSTANCE", 1);
    else
        build_params.Define("USE_SOFTMAX_MODE_CHANNEL", 1);

    build_params.Define("RUN_FORWARD", 1);

    build_params.Define("IS_INPUT_PACKED", static_cast<int>(xDesc.IsPacked()));
    build_params.Define("IS_OUTPUT_PACKED", static_cast<int>(yDesc.IsPacked()));

    if(!float_equal(alpha_fp, 1.0))
        build_params.Define("USE_ALPHA", 1);

    if(!float_equal(beta_fp, 0))
        build_params.Define("USE_BETA", 1);

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

    kernel.kernel_file = "MIOpenSoftmax.cl";
    kernel.kernel_name = "SoftmaxForward";
   
    for (unsigned int i = 0; i < 2; ++i)
    {
        kernel.l_wk.push_back(vld[i]);
        kernel.g_wk.push_back(vgd[i]);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::softmax::InvokeParams>();

            kernel(params.x,
                    params.y,
                    vector_size,
                    grid_size,
                    spatial_dim,
                    h,
                    w,
                    in_nstr,
                    in_cstr,
                    in_hstr,
                    out_nstr,
                    out_cstr,
                    out_hstr,
                    params.x_offset,
                    params.y_offset,
                    alpha_fp,
                    beta_fp);
        };
    };

    result.construction_params.push_back(kernel);

    return result;
}

} // namespace softmax

} // namespace solver

} // namespace miopen
