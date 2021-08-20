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
#include <miopen/activ.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/activ/invoke_params.hpp>
#include <miopen/activ/problem_description.hpp>
#include <miopen/activ/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t ActivationDescriptor::Forward(Handle& handle,
                                             const void* alpha,
                                             const TensorDescriptor& xDesc,
                                             ConstData_t x,
                                             const void* beta,
                                             const TensorDescriptor& yDesc,
                                             Data_t y,
                                             size_t xOffset,
                                             size_t yOffset)
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }

    const auto problem = activ::ProblemDescription{*this, xDesc, yDesc};

    const auto invoke_params = [&]() {
        auto tmp     = activ::InvokeParams{};
        tmp.type     = InvokeType::Run;
        tmp.alpha    = GetAlpha();
        tmp.beta     = GetBeta();
        tmp.gamma    = GetGamma();
        tmp.x        = x;
        tmp.x_desc   = xDesc;
        tmp.y        = y;
        tmp.y_desc   = yDesc;
        tmp.x_offset = xOffset;
        tmp.y_offset = yOffset;
        return tmp;
    }();

    const auto algo           = AlgorithmName{"miopenActivationForward"};
    const auto network_config = problem.MakeNetworkConfig();

    if(const auto invoker = handle.GetInvoker(network_config, boost::none, algo))
    {
        (*invoker)(handle, invoke_params);
        return miopenStatusSuccess;
    }

    const auto ctx = ExecutionContext{&handle};
    const auto solvers =
        solver::SolverContainer<solver::activ::ActivFwdSolver0, solver::activ::ActivFwdSolver1>{};
    const auto slns = solvers.SearchForSolutions(ctx, problem, 1);

    if(slns.empty())
        MIOPEN_THROW(miopenStatusNotImplemented, "No solver found for activation forward.");

    const auto& sln = slns.front();
    if(!sln.invoker_factory)
        MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
    const auto invoker = handle.PrepareInvoker(*sln.invoker_factory, sln.construction_params);
    handle.RegisterInvoker(invoker, network_config, sln.solver_id, algo);
    invoker(handle, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t ActivationDescriptor::Backward(Handle& handle,
                                              const void* alpha,
                                              const TensorDescriptor& yDesc,
                                              ConstData_t y,
                                              const TensorDescriptor& dyDesc,
                                              ConstData_t dy,
                                              const TensorDescriptor& xDesc,
                                              ConstData_t x,
                                              const void* beta,
                                              const TensorDescriptor& dxDesc,
                                              Data_t dx,
                                              size_t yOffset,
                                              size_t dyOffset,
                                              size_t xOffset,
                                              size_t dxOffset)
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }

    const auto problem = activ::ProblemDescription{*this, xDesc, yDesc, dxDesc, dyDesc};

    const auto invoke_params = [&]() {
        auto tmp      = activ::BwdInvokeParams{};
        tmp.type      = InvokeType::Run;
        tmp.alpha     = GetAlpha();
        tmp.beta      = GetBeta();
        tmp.gamma     = GetGamma();
        tmp.x         = x;
        tmp.x_desc    = xDesc;
        tmp.y         = y;
        tmp.y_desc    = yDesc;
        tmp.dx        = dx;
        tmp.dx_desc   = dxDesc;
        tmp.dy        = dy;
        tmp.dy_desc   = dyDesc;
        tmp.x_offset  = xOffset;
        tmp.y_offset  = yOffset;
        tmp.dx_offset = dxOffset;
        tmp.dy_offset = dyOffset;
        return tmp;
    }();

    const auto algo           = AlgorithmName{"miopenActivationBackward"};
    const auto network_config = problem.MakeNetworkConfig();

    if(const auto invoker = handle.GetInvoker(network_config, boost::none, algo))
    {
        (*invoker)(handle, invoke_params);
        return miopenStatusSuccess;
    }

    const auto ctx     = ExecutionContext{&handle};
    const auto solvers = solver::SolverContainer<solver::activ::ActivBwdSolver0>{};
    const auto slns    = solvers.SearchForSolutions(ctx, problem, 1);

    if(slns.empty())
        MIOPEN_THROW(miopenStatusNotImplemented, "No solver found for activation forward.");

    const auto& sln = slns.front();
    if(!sln.invoker_factory)
        MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
    const auto invoker = handle.PrepareInvoker(*sln.invoker_factory, sln.construction_params);
    handle.RegisterInvoker(invoker, network_config, sln.solver_id, algo);
    invoker(handle, invoke_params);
    return miopenStatusSuccess;
}
} // namespace miopen
