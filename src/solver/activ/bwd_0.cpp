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

    const auto& xDesc = problem.GetXDesc();
    const auto& yDesc = problem.GetXDesc();
    const auto& dxDesc = problem.GetDXDesc();
    const auto& dyDesc = problem.GetDXDesc();

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

    const auto x_stride2D = x_strides[x_lens.size() - 2];
    const auto y_stride2D = y_strides[y_lens.size() - 2];
    const auto dx_stride2D = dx_strides[dx_lens.size() - 2];
    const auto dy_stride2D = dy_strides[dy_lens.size() - 2];

    const auto x_width2D = x_lens[x_lens.size() - 1];
    const auto y_width2D = y_lens[y_lens.size() - 1];
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
    MIOPEN_THROW("Not implemented");

    auto result = ConvSolution{miopenStatusSuccess};

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
        };
    };

    return result;
}

} // namespace activ

} // namespace solver

} // namespace miopen
