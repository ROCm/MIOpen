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

#include <miopen/activ/solvers.hpp>

#include <miopen/activ/invoke_params.hpp>
#include <miopen/activ/problem_description.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>

#include <cassert>

namespace miopen {

namespace solver {

namespace activ {

bool ActivBwdSolver0::IsApplicable(const ExecutionContext&,
                                   const miopen::activ::ProblemDescription& problem) const
{
    if(problem.GetDirection() != miopen::activ::Direction::Backward)
        return false;

    const auto& xDesc  = problem.GetXDesc();
    const auto& yDesc  = problem.GetYDesc();
    const auto& dxDesc = problem.GetDXDesc();
    const auto& dyDesc = problem.GetDYDesc();

    const auto x_elem_sz  = xDesc.GetElementSize();
    const auto y_elem_sz  = yDesc.GetElementSize();
    const auto dx_elem_sz = dxDesc.GetElementSize();
    const auto dy_elem_sz = dyDesc.GetElementSize();

    if(!(x_elem_sz == y_elem_sz && dx_elem_sz == dy_elem_sz && x_elem_sz == dx_elem_sz))
        return false;

    if(xDesc.IsPacked() && yDesc.IsPacked() && dxDesc.IsPacked() && dyDesc.IsPacked())
        return true;

    const auto& x_lens  = xDesc.GetLengths();
    const auto& y_lens  = yDesc.GetLengths();
    const auto& dx_lens = dxDesc.GetLengths();
    const auto& dy_lens = dyDesc.GetLengths();

    const auto& x_strides  = xDesc.GetStrides();
    const auto& y_strides  = yDesc.GetStrides();
    const auto& dx_strides = dxDesc.GetStrides();
    const auto& dy_strides = dyDesc.GetStrides();

    const auto x_stride2D  = x_strides[x_lens.size() - 2];
    const auto y_stride2D  = y_strides[y_lens.size() - 2];
    const auto dx_stride2D = dx_strides[dx_lens.size() - 2];
    const auto dy_stride2D = dy_strides[dy_lens.size() - 2];

    const auto x_width2D  = x_lens[x_lens.size() - 1];
    const auto y_width2D  = y_lens[y_lens.size() - 1];
    const auto dx_width2D = dx_lens[dx_lens.size() - 1];
    const auto dy_width2D = dy_lens[dy_lens.size() - 1];

    // clang-format off
    return x_lens.size() == y_lens.size() && dx_lens.size() == dy_lens.size() && x_lens.size() == dx_lens.size()
        && ((x_width2D != x_stride2D) || (y_width2D != y_stride2D)
            || (dx_width2D != dx_stride2D) || (dy_width2D != dy_stride2D))
        && (x_lens.size() == 2
            || (x_lens.size() == 3
                && x_lens[0] == 1 && y_lens[0] == 1 && dx_lens[0] == 1 && dy_lens[0] == 1)
            || (x_lens.size() == 4
                && x_lens[0] == 1 && x_lens[1] == 1
                && y_lens[0] == 1 && y_lens[1] == 1
                && dy_lens[0] == 1 && dy_lens[1] == 1
                && dx_lens[0] == 1 && dx_lens[1] == 1)
            || (x_lens.size() == 5
                && x_lens[0] == 1 && x_lens[1] == 1 && x_lens[2] == 1
                && y_lens[0] == 1 && y_lens[1] == 1 && y_lens[2] == 1
                && dy_lens[0] == 1 && dy_lens[1] == 1 && dy_lens[2] == 1
                && dx_lens[0] == 1 && dx_lens[1] == 1 && dx_lens[2] == 1));
    // clang-format on
}

ConvSolution ActivBwdSolver0::GetSolution(const ExecutionContext&,
                                          const miopen::activ::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& xDesc  = problem.GetXDesc();
    const auto& yDesc  = problem.GetYDesc();
    const auto& dxDesc = problem.GetDXDesc();
    const auto& dyDesc = problem.GetDYDesc();

    const auto& x_lens  = xDesc.GetLengths();
    const auto& dx_lens = dxDesc.GetLengths();

    const auto dx_width2D = dx_lens[dx_lens.size() - 1];
    const auto height     = x_lens[x_lens.size() - 2];

    const auto packed =
        xDesc.IsPacked() && yDesc.IsPacked() && dxDesc.IsPacked() && dyDesc.IsPacked();

    const auto x_elem_sz = xDesc.GetElementSize();

    const auto read_len = (packed) ? x_elem_sz : dx_width2D;

    const auto read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    const auto MAP_RD    = read_len / read_unit;
    const auto READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);

    auto compiler_options = KernelBuildParameters{
        {"LITE"},
        {"MIOPEN_READ_UNIT", read_unit},
        {"MIOPEN_READ_TYPE", READ_TYPE},
        {"MIOPEN_NRN_OP_ID", problem.GetActivDesc().GetMode()},
    };

    if(xDesc.GetType() == miopenFloat)
    {
        compiler_options.Define("MIOPEN_USE_FP16", 0);
        compiler_options.Define("MIOPEN_USE_FP32", 1);
    }
    else if(xDesc.GetType() == miopenHalf)
    {
        compiler_options.Define("MIOPEN_USE_FP16", 1);
        compiler_options.Define("MIOPEN_USE_FP32", 0);
    }

    {
        auto kernel = KernelInfo{};

        kernel.comp_options = compiler_options.GenerateFor(kbp::OpenCL{});
        kernel.kernel_file  = "MIOpenNeuron.cl";
        kernel.kernel_name  = (packed) ? "MIOpenActiveBwdLite" : "MIOpenActiveBwd2DLite";

        kernel.l_wk.push_back(256);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);

        // first dimension looks similar but for the packed it is a full image for the
        // non-packed 2D it's width
        kernel.g_wk.push_back(MAP_RD);
        kernel.g_wk.push_back(packed ? 1 : height);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kernel_handle = kernels.front();

        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle.Run(kernel_handle);
            decltype(auto) params = raw_params.CastTo<miopen::activ::BwdInvokeParams>();

            const auto& dx = params.dx;
            const auto& dy = params.dy;
            const auto& x  = params.x;
            const auto& y  = params.y;

            const auto& dxOffset = params.dx_offset;
            const auto& dyOffset = params.dy_offset;
            const auto& xOffset  = params.x_offset;
            const auto& yOffset  = params.y_offset;

            visit_float(params.x_desc.GetType(), [&](auto as_float) {
                decltype(auto) f_activ_alpha = as_float(params.alpha);
                decltype(auto) f_activ_beta  = as_float(params.beta);
                decltype(auto) f_activ_gamma = as_float(params.gamma);
                decltype(auto) f_diff_scale  = as_float(params.beta * params.gamma);

                if(packed)
                {
                    kernel(dx,
                           dy,
                           x,
                           y,
                           f_diff_scale,
                           f_activ_gamma,
                           f_activ_beta,
                           f_activ_alpha,
                           static_cast<long long>(dxOffset),
                           static_cast<long long>(dyOffset),
                           static_cast<long long>(xOffset),
                           static_cast<long long>(yOffset));
                }
                else
                {
                    const auto& x_lens_  = params.x_desc.GetLengths();
                    const auto& y_lens_  = params.y_desc.GetLengths();
                    const auto& dx_lens_ = params.dx_desc.GetLengths();
                    const auto& dy_lens_ = params.dy_desc.GetLengths();

                    const auto& x_strides_  = params.x_desc.GetStrides();
                    const auto& y_strides_  = params.y_desc.GetStrides();
                    const auto& dx_strides_ = params.dx_desc.GetStrides();
                    const auto& dy_strides_ = params.dy_desc.GetStrides();

                    const auto x_stride2D_  = x_strides_[x_lens_.size() - 2];
                    const auto y_stride2D_  = y_strides_[y_lens_.size() - 2];
                    const auto dx_stride2D_ = dx_strides_[dx_lens_.size() - 2];
                    const auto dy_stride2D_ = dy_strides_[dy_lens_.size() - 2];

                    kernel(dx,
                           dy,
                           x,
                           y,
                           f_diff_scale,
                           f_activ_gamma,
                           f_activ_beta,
                           f_activ_alpha,
                           static_cast<long long>(dxOffset),
                           static_cast<long long>(dyOffset),
                           static_cast<long long>(xOffset),
                           static_cast<long long>(yOffset),
                           static_cast<unsigned int>(dx_stride2D_),
                           static_cast<unsigned int>(dy_stride2D_),
                           static_cast<unsigned int>(x_stride2D_),
                           static_cast<unsigned int>(y_stride2D_));
                }
            });
        };
    };

    return result;
}

} // namespace activ

} // namespace solver

} // namespace miopen
