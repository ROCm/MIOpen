/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/solution.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/any_solver.hpp>

namespace miopen {

void Solution::Run(Handle& handle,
                   const std::unordered_map<miopenTensorName_t, RunInput>& inputs,
                   Data_t workspace,
                   std::size_t workspace_size)
{
    if(workspace_size < workspace_required)
        MIOPEN_THROW(miopenStatusBadParm,
                     GetSolver().ToString() + " requires at least " +
                         std::to_string(workspace_required) + " workspace, while " +
                         std::to_string(workspace_size) + " was provided");

    if(problem.GetOperatorDescriptor().GetPrimitive() != solver::Primitive::Convolution)
        MIOPEN_THROW(miopenStatusNotImplemented);

    auto problem_ = problem;

    const auto get_input_checked = [&](auto name, const std::string& name_str) {
        const auto& found = inputs.find(name);
        if(found == inputs.end())
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Problem is missing " + name_str + " tensor descriptor.");
        auto ret = found->second;
        if(!ret.descriptor.has_value())
            ret.descriptor = problem.GetTensorDescriptorChecked(name, name_str);
        return ret;
    };

    auto x       = get_input_checked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto w = get_input_checked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    auto y       = get_input_checked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    {
        const auto& conv_desc =
            dynamic_cast<const ConvolutionDescriptor&>(problem.GetOperatorDescriptor());

        if(conv_desc.mode == miopenTranspose)
        {
            problem_ = {};
            problem_.SetOperatorDescriptor(&conv_desc);

            switch(problem.GetDirection())
            {
            case miopenProblemDirectionForward:
                problem_.SetDirection(miopenProblemDirectionBackward);
                break;
            case miopenProblemDirectionBackward:
                problem_.SetDirection(miopenProblemDirectionForward);
                break;
            case miopenProblemDirectionBackwardWeight:
                problem_.SetDirection(miopenProblemDirectionBackwardWeight);
                break;
            }

            std::swap(x, y);

            problem_.RegisterTensorDescriptor(miopenTensorConvolutionX, *x.descriptor);
            problem_.RegisterTensorDescriptor(miopenTensorConvolutionW, *w.descriptor);
            problem_.RegisterTensorDescriptor(miopenTensorConvolutionY, *y.descriptor);
        }
    }

    const auto conv_problem = problem_.AsConvolution();

    const auto invoke_ctx = [&]() -> AnyInvokeParams {
        switch(problem_.GetDirection())
        {
        case miopenProblemDirectionForward:
            return conv::DataInvokeParams(
                {*x.descriptor, x.buffer, *w.descriptor, w.buffer, *y.descriptor, y.buffer},
                workspace,
                workspace_size,
                conv_problem.GetConv().attribute.gfx90aFp16alt.GetFwd());
        case miopenProblemDirectionBackward:
            return conv::DataInvokeParams(
                {*y.descriptor, y.buffer, *w.descriptor, w.buffer, *x.descriptor, x.buffer},
                workspace,
                workspace_size,
                conv_problem.GetConv().attribute.gfx90aFp16alt.GetFwd());
        case miopenProblemDirectionBackwardWeight:
            return conv::WrWInvokeParams{
                {*y.descriptor, y.buffer, *x.descriptor, x.buffer, *w.descriptor, w.buffer},
                workspace,
                workspace_size,
                conv_problem.GetConv().attribute.gfx90aFp16alt.GetWrW()};
        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }
    }();

    const auto net_cfg       = conv_problem.BuildConfKey();
    const auto found_invoker = handle.GetInvoker(net_cfg, solver);

    if(found_invoker)
    {
        (*found_invoker)(handle, invoke_ctx);
        return;
    }

    const auto conv_ctx = ConvolutionContext{conv_problem, {&handle}};

    decltype(auto) db        = GetDb(conv_ctx);
    const auto conv_solution = solver.GetSolver().FindSolution(conv_ctx, db, invoke_ctx);
    decltype(auto) invoker =
        handle.PrepareInvoker(*conv_solution.invoker_factory, conv_solution.construction_params);
    handle.RegisterInvoker(invoker, net_cfg, solver.ToString());
    invoker(handle, invoke_ctx);
}

} // namespace miopen
