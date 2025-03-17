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

#include <miopen/env.hpp>
#include <miopen/softmax/solvers.hpp>

#include <miopen/softmax/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/softmax.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/float_equal.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_OCL_SOFTMAX)

namespace miopen {

namespace {
constexpr int nextPow2(int v)
{
    if(v == 1)
    {
        return (v << 1);
    }
    else
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
}

void getParams(const TensorDescriptor& in_desc,
               miopenSoftmaxMode_t in_mode,
               int& out_n,
               int& out_c,
               int& out_h,
               int& out_w,
               int& out_grid_size,
               int& out_spatial_dim,
               int& out_vector_size,
               int& out_num_batch,
               bool& out_usefp16,
               bool& out_usefp32,
               std::vector<size_t>& out_vld,
               std::vector<size_t>& out_vgd,
               size_t& out_workgroups,
               int& out_batch_size,
               int& out_u_batch_size)
{
    std::tie(out_n, out_c, out_h, out_w) = tien<4>(in_desc.GetLengths());

    // using workgroup size of 256 by default
    out_grid_size   = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? out_n : out_n * out_h * out_w;
    out_spatial_dim = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? 1 : out_h * out_w;
    out_vector_size = in_mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? out_c * out_h * out_w : out_c;
    // num_spatial_dims or pixels each workgroup can compute

    /// \todo Magic numbers
    out_num_batch = out_vector_size < 256 ? nextPow2(256 / out_vector_size) : 1;

    out_vld = {256, 1, 1};

    out_usefp16 = false;
    out_usefp32 = true;
    if(in_desc.GetType() == miopenHalf)
    {
        out_usefp16 = true;
        out_usefp32 = false;
    }

    if(out_num_batch == 1)
    {
        out_workgroups = std::min(out_grid_size, 64 * 40 * 8);
        out_vgd        = {out_workgroups * out_vld[0], 1, 1};

        out_batch_size   = 0;
        out_u_batch_size = 0;
    }
    else
    {
        out_batch_size = 256 / out_num_batch;
        // num_channels each threads iterates over to cover all the channels
        out_u_batch_size =
            (out_vector_size > out_batch_size) ? nextPow2(out_vector_size / out_batch_size) : 1;

        out_workgroups = (out_grid_size % out_num_batch == 0) ? (out_grid_size / out_num_batch)
                                                              : (out_grid_size / out_num_batch + 1);
        out_vgd        = {out_workgroups * out_vld[0], 1, 1};
    }
}
} // namespace

namespace solver {

namespace softmax {

bool Softmax::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::softmax::ProblemDescription& problem) const
{
    return !env::disabled(MIOPEN_DEBUG_OCL_SOFTMAX);
}

std::size_t
Softmax::GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                          [[maybe_unused]] const miopen::softmax::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution Softmax::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                  const miopen::softmax::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto xDesc = problem.GetXDesc();
    auto yDesc = problem.GetYDesc();

    auto dxDesc = problem.GetdXDesc();
    auto dyDesc = problem.GetdYDesc();

    auto alpha     = problem.GetAlpha();
    auto beta      = problem.GetBeta();
    auto mode      = problem.GetMode();
    auto algorithm = problem.GetAlgorithm();

    bool isForward = problem.IsForward();

    int n, c, h, w;
    // using workgroup size of 256 by default
    int grid_size, spatial_dim, vector_size, num_batch;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool usefp16, usefp32;

    size_t workgroups;
    int batch_size;
    int u_batch_size;

    getParams(yDesc,
              mode,
              n,
              c,
              h,
              w,
              grid_size,
              spatial_dim,
              vector_size,
              num_batch,
              usefp16,
              usefp32,
              vld,
              vgd,
              workgroups,
              batch_size,
              u_batch_size);

    if(num_batch > 1)
    {
        if(isForward)
        {
            /// \todo Magic numbers
            if((u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
                MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");
        }
        else
        {
            /// \todo Magic numbers
            if((2 * u_batch_size + 1) * 256 > 65536 && yDesc.GetType() == miopenHalf)
                MIOPEN_THROW(miopenStatusBadParm, "Exceed local memory capacity");
        }
    }

    KernelBuildParameters build_params = KernelBuildParameters{{"NUM_BATCH", num_batch}};

    if(num_batch > 1)
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

    build_params.Define("RUN_FORWARD", isForward ? 1 : 0);

    if(isForward)
    {
        build_params.Define("IS_INPUT_PACKED", static_cast<int>(xDesc.IsPacked()));
        build_params.Define("IS_OUTPUT_PACKED", static_cast<int>(yDesc.IsPacked()));
    }
    else
    {
        build_params.Define("IS_OUTPUT_PACKED", static_cast<int>(yDesc.IsPacked()));
        build_params.Define("IS_DOUTPUT_PACKED", static_cast<int>(dyDesc.IsPacked()));
        build_params.Define("IS_DINPUT_PACKED", static_cast<int>(dxDesc.IsPacked()));
    }

    if(!float_equal(alpha, 1.0))
        build_params.Define("USE_ALPHA", 1);

    if(!float_equal(beta, 0))
        build_params.Define("USE_BETA", 1);

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

    kernel.kernel_file = "MIOpenSoftmax.cl";

    kernel.kernel_name = isForward ? "SoftmaxForward" : "SoftmaxBackward";

    for(unsigned int i = 0; i < 2; ++i)
    {
        kernel.l_wk.push_back(vld[i]);
        kernel.g_wk.push_back(vgd[i]);
    }

    if(isForward)
    {
        int in_nstr, in_cstr, in_hstr;
        std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = tien<4>(xDesc.GetStrides());

        int out_nstr, out_cstr, out_hstr;
        std::tie(out_nstr, out_cstr, out_hstr, std::ignore) = tien<4>(yDesc.GetStrides());

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::softmax::InvokeParams>();

                kernel(params.x,
                       params.forward_y,
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
                       params.xdx_offset,
                       params.y_offset,
                       alpha,
                       beta);
            };
        };
    }
    else
    {
        int din_nstr, din_cstr, din_hstr;
        std::tie(din_nstr, din_cstr, din_hstr, std::ignore) = tien<4>(dxDesc.GetStrides());

        int dout_nstr, dout_cstr, dout_hstr;
        std::tie(dout_nstr, dout_cstr, dout_hstr, std::ignore) = tien<4>(dyDesc.GetStrides());

        int out_nstr, out_cstr, out_hstr;
        std::tie(out_nstr, out_cstr, out_hstr, std::ignore) = tien<4>(yDesc.GetStrides());

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::softmax::InvokeParams>();

                kernel(params.backward_y,
                       params.dy,
                       params.dx,
                       vector_size,
                       grid_size,
                       spatial_dim,
                       h,
                       w,
                       out_nstr,
                       out_cstr,
                       out_hstr,
                       dout_nstr,
                       dout_cstr,
                       dout_hstr,
                       din_nstr,
                       din_cstr,
                       din_hstr,
                       params.y_offset,
                       params.dy_offset,
                       params.xdx_offset,
                       alpha,
                       beta);
            };
        };
    }

    result.construction_params.push_back(kernel);

    return result;
}

} // namespace softmax

} // namespace solver

} // namespace miopen
