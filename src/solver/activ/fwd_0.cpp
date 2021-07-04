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

namespace miopen {

namespace solver {

namespace activ {

bool ActivFwdSolver0::IsApplicable(const ExecutionContext&,
                                   const miopen::activ::ProblemDescription& problem) const
{
    if(problem.GetDirection() != miopen::activ::Direction::Forward)
        return false;

    // short cut for packed tensors and 2D tensors with stride != width
    const auto& x_lens = problem.GetXDesc().GetLengths();
    const auto& y_lens = problem.GetYDesc().GetLengths();

    const auto& x_strides = problem.GetXDesc().GetStrides();
    const auto& y_strides = problem.GetYDesc().GetStrides();

    const auto x_elem_sz = problem.GetXDesc().GetElementSize();
    const auto y_elem_sz = problem.GetYDesc().GetElementSize();

    const auto x_stride2D = static_cast<unsigned int>(
        (x_lens.size() == 2) ? x_strides[0] : (x_lens.size() == 3)
                                                  ? x_strides[1]
                                                  : (x_lens.size() == 4) ? x_strides[2]
                                                                         : x_strides[3]);
    const auto y_stride2D = static_cast<unsigned int>(
        (y_lens.size() == 2) ? y_strides[0] : (y_lens.size() == 3)
                                                  ? y_strides[1]
                                                  : (y_lens.size() == 4) ? y_strides[2]
                                                                         : y_strides[3]);

    const auto x_width2D =
        ((x_lens.size() == 2) ? x_lens[1] : (x_lens.size() == 3) ? x_lens[2] : (x_lens.size() == 4)
                                                                                   ? x_lens[3]
                                                                                   : x_lens[4]);

    const auto y_width2D =
        ((y_lens.size() == 2) ? y_lens[1] : (y_lens.size() == 3) ? y_lens[2] : (y_lens.size() == 4)
                                                                                   ? y_lens[3]
                                                                                   : y_lens[4]);

    const auto t2D =
        (x_lens.size() == y_lens.size() &&
         ((x_width2D != x_stride2D) || (y_width2D != y_stride2D)) &&
         (x_lens.size() == 2 || (x_lens.size() == 3 && x_lens[0] == 1 && y_lens[0] == 1) ||
          (x_lens.size() == 4 && x_lens[0] == 1 && x_lens[1] == 1 && y_lens[0] == 1 &&
           y_lens[1] == 1) ||
          (x_lens.size() == 5 && x_lens[0] == 1 && x_lens[1] == 1 && x_lens[2] == 1 &&
           y_lens[0] == 1 && y_lens[1] == 1 && y_lens[2] == 1)));
    const auto packed = problem.GetXDesc().IsPacked() && problem.GetYDesc().IsPacked();

    return x_elem_sz == y_elem_sz && (packed || t2D);
}

ConvSolution ActivFwdSolver0::GetSolution(const ExecutionContext&,
                                          const miopen::activ::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    // short cut for packed tensors and 2D tensors with stride != width
    const auto x_lens = problem.GetXDesc().GetLengths();
    const auto y_lens = problem.GetYDesc().GetLengths();

    const auto x_elem_sz = problem.GetXDesc().GetElementSize();

    const auto x_width2D =
        ((x_lens.size() == 2) ? x_lens[1] : (x_lens.size() == 3) ? x_lens[2] : (x_lens.size() == 4)
                                                                                   ? x_lens[3]
                                                                                   : x_lens[4]);

    const auto packed    = problem.GetXDesc().IsPacked() && problem.GetYDesc().IsPacked();
    const auto read_len  = (packed) ? x_elem_sz : x_width2D;
    const auto read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;

    const auto READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);

    const auto height =
        (x_lens.size() == 2) ? x_lens[0] : (x_lens.size() == 3) ? x_lens[1] : (x_lens.size() == 4)
                                                                                  ? x_lens[2]
                                                                                  : x_lens[3];

    auto build_params = KernelBuildParameters{
        {"LITE"},
        {"MIOPEN_READ_UNIT", read_unit},
        {"MIOPEN_READ_TYPE", READ_TYPE},
        {"MIOPEN_NRN_OP_ID", problem.GetActivDesc().GetMode()},
    };

    if(problem.GetXDesc().GetType() == miopenFloat)
    {
        build_params.Define("MIOPEN_USE_FP16", 0);
        build_params.Define("MIOPEN_USE_FP32", 1);
    }
    else if(problem.GetXDesc().GetType() == miopenHalf)
    {
        build_params.Define("MIOPEN_USE_FP16", 1);
        build_params.Define("MIOPEN_USE_FP32", 0);
    }

    {
        auto kernel_info         = KernelInfo{};
        kernel_info.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel_info.l_wk.push_back(256);
        kernel_info.l_wk.push_back(1);
        kernel_info.l_wk.push_back(1);

        const auto MAP_RD = read_len / read_unit;

        kernel_info.g_wk.push_back(MAP_RD);
        kernel_info.g_wk.push_back(packed ? 1 : height);
        kernel_info.g_wk.push_back(1);

        kernel_info.kernel_file = "MIOpenNeuron.cl";
        kernel_info.kernel_name = (packed) ? "MIOpenActiveFwdLite" : "MIOpenActiveFwd2DLite";

        result.construction_params.push_back(kernel_info);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::activ::InvokeParams>();

            visit_float(params.x_desc.GetType(), [&](auto as_float) {
                const auto alpha = as_float(params.alpha);
                const auto beta  = as_float(params.beta);
                const auto gamma = as_float(params.gamma);

                if(packed)
                {
                    kernel(params.x,
                           params.y,
                           gamma,
                           beta,
                           alpha,
                           static_cast<long long>(params.x_offset),
                           static_cast<long long>(params.y_offset));
                }
                else
                {
                    const auto x_lens_ = params.x_desc.GetLengths();
                    const auto y_lens_ = params.y_desc.GetLengths();

                    const auto x_strides = params.x_desc.GetStrides();
                    const auto y_strides = params.y_desc.GetStrides();

                    const auto x_stride2D = static_cast<unsigned int>(
                        (x_lens_.size() == 2) ? x_strides[0] : (x_lens_.size() == 3)
                                                                   ? x_strides[1]
                                                                   : (x_lens_.size() == 4)
                                                                         ? x_strides[2]
                                                                         : x_strides[3]);
                    const auto y_stride2D = static_cast<unsigned int>(
                        (y_lens_.size() == 2) ? y_strides[0] : (y_lens_.size() == 3)
                                                                   ? y_strides[1]
                                                                   : (y_lens_.size() == 4)
                                                                         ? y_strides[2]
                                                                         : y_strides[3]);

                    kernel(params.x,
                           params.y,
                           gamma,
                           beta,
                           alpha,
                           static_cast<long long>(params.x_offset),
                           static_cast<long long>(params.y_offset),
                           x_stride2D,
                           y_stride2D);
                }
            });
        };
    };

    return result;
}

} // namespace activ

} // namespace solver

} // namespace miopen
