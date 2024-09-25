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

#include <miopen/any_solver.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/kernel.hpp>

#include <miopen/mha/invoke_params.hpp>
#include <miopen/mha/problem_description.hpp>
#include <miopen/mha/solvers.hpp>
#include <miopen/softmax/invoke_params.hpp>
#include <miopen/softmax/problem_description.hpp>
#include <miopen/softmax/solvers.hpp>

#include <nlohmann/json.hpp>

#include <boost/hof/match.hpp>
#include "miopen/fusion/problem_description.hpp"
#include "miopen/fusion/context.hpp"

namespace miopen::debug {
// Todo: This should be updated when a separate driver command is implemented
void LogCmdConvolution(const miopen::TensorDescriptor& x,
                       const miopen::TensorDescriptor& w,
                       const miopen::ConvolutionDescriptor& conv,
                       const miopen::TensorDescriptor& y,
                       miopenProblemDirection_t dir,
                       std::optional<uint64_t> solver_id);
} // namespace miopen::debug

namespace miopen {

void Solution::Run(Handle& handle,
                   const std::unordered_map<miopenTensorArgumentId_t, RunInput>& inputs,
                   Data_t workspace,
                   std::size_t workspace_size)
{
    if(workspace_size < workspace_required)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     GetSolver().ToString() + " requires at least " +
                         std::to_string(workspace_required) + " workspace, while " +
                         std::to_string(workspace_size) + " was provided");
    }

    std::visit(boost::hof::match(
                   [&](const Problem& problem_) {
                       std::visit(
                           boost::hof::match(
                               [&](const ConvolutionDescriptor& op_desc) {
                                   RunImpl(handle, inputs, workspace, workspace_size, op_desc);
                               },
                               [&](const SoftmaxDescriptor& op_desc) {
                                   RunImpl(handle, inputs, workspace, workspace_size, op_desc);
                               },
                               [&](const ActivationDescriptor& /*op_desc*/) {
                                   MIOPEN_THROW(miopenStatusNotImplemented);
                               },
                               [&](const BiasDescriptor& /*op_desc*/) {
                                   MIOPEN_THROW(miopenStatusNotImplemented);
                               },
                               [&](const MhaDescriptor& op_desc) {
                                   RunImpl(handle, inputs, workspace, workspace_size, op_desc);
                               },
                               [&](const BatchnormDescriptor& /*op_desc*/) {
                                   MIOPEN_THROW(miopenStatusNotImplemented);
                               }),
                           problem_.GetOperatorDescriptor());
                   },
                   [&](const FusedProblem& problem_) {
                       RunImpl(handle, inputs, workspace, workspace_size, problem_);
                   }),
               problem.item);
}

void Solution::LogDriverCommand() const
{
    std::visit([&](const auto& problem_) { LogDriverCommand(problem_); }, problem.item);
}

void Solution::LogDriverCommand(const ConvolutionDescriptor& desc) const
{
    auto problem_ = std::get<Problem>(problem.item);
    const auto& x_desc =
        problem_.GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        problem_.GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    const auto& y_desc =
        problem_.GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");
    miopen::debug::LogCmdConvolution(
        x_desc, w_desc, desc, y_desc, problem_.GetDirection(), solver.Value());
}

void Solution::LogDriverCommand(const ActivationDescriptor& desc) const
{
    std::ignore = desc;
    std::get<Problem>(problem.item).LogDriverCommand();
    /// \todo: when possible, add some command for reproducing a specific case rather than the whole
    /// problem
}

void Solution::LogDriverCommand(const BatchnormDescriptor& desc) const
{
    std::ignore = desc;
    /// \todo: bnorm command logging
}

void Solution::LogDriverCommand(const Problem& problem_) const
{
    std::visit(boost::hof::match(
                   [&](const BiasDescriptor&) { /* \todo: think on how to log bias */ },
                   [&](const MhaDescriptor&) { /* \todo: think on how to log mha */ },
                   [&](const SoftmaxDescriptor&) { /* \todo: think on how to log softmax */ },
                   [&](const auto& op_desc) { LogDriverCommand(op_desc); }),
               problem_.GetOperatorDescriptor());
}

void Solution::LogDriverCommand(const FusedProblem& problem_) const
{
    std::ignore = problem_;
    /// \todo: add logging of some command to reproduce current solution or at least problem
}

void Solution::RunImpl(Handle& handle,
                       const std::unordered_map<miopenTensorArgumentId_t, RunInput>& inputs,
                       Data_t workspace,
                       std::size_t workspace_size,
                       const ConvolutionDescriptor& conv_desc)
{
    const auto& problem_casted = std::get<Problem>(problem.item);

    const auto get_input_checked = [&](auto name, const std::string& name_str) {
        const auto& found = inputs.find(name);
        if(found == inputs.end())
        {
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Problem is missing " + name_str + " tensor descriptor.");
        }
        auto ret = found->second;
        if(!ret.descriptor.has_value())
            ret.descriptor = problem_casted.GetTensorDescriptorChecked(name, name_str);
        return ret;
    };

    auto x       = get_input_checked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto w = get_input_checked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    auto y       = get_input_checked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    const auto problem_ =
        conv_desc.mode == miopenTranspose ? Transpose(problem_casted, &x, w, &y) : problem_casted;

    if(problem_.GetDirection() == miopenProblemDirectionBackward &&
       y.descriptor->GetLengths()[1] != w.descriptor->GetLengths()[0])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(miopen::CheckNumericsEnabled())
    {
        if(problem_.GetDirection() != miopenProblemDirectionBackward)
            miopen::checkNumericsInput(handle, *x.descriptor, x.buffer);
        if(problem_.GetDirection() != miopenProblemDirectionBackwardWeights)
            miopen::checkNumericsInput(handle, *w.descriptor, w.buffer);
        if(problem_.GetDirection() != miopenProblemDirectionForward)
            miopen::checkNumericsInput(handle, *y.descriptor, y.buffer);
    }

    Problem::ValidateGroupCount(*x.descriptor, *w.descriptor, conv_desc);

    const auto invoke_ctx =
        MakeInvokeParams(problem_, conv_desc, x, w, y, workspace, workspace_size);

    const auto checkNumericsOutput_ = [&]() {
        if(miopen::CheckNumericsEnabled())
        {
            if(problem_.GetDirection() == miopenProblemDirectionBackward)
                miopen::checkNumericsOutput(handle, *x.descriptor, x.buffer);
            if(problem_.GetDirection() == miopenProblemDirectionBackwardWeights)
                miopen::checkNumericsOutput(handle, *w.descriptor, w.buffer);
            if(problem_.GetDirection() == miopenProblemDirectionForward)
                miopen::checkNumericsOutput(handle, *y.descriptor, y.buffer);
        }
    };

    if(invoker)
    {
        (*invoker)(handle, invoke_ctx);
        checkNumericsOutput_();
        return;
    }

    const auto conv_problem = problem_.AsConvolution();

    if(!kernels.empty())
    {
        auto ctx = ExecutionContext{&handle};
        conv_problem.SetupFloats(ctx);
        const auto invoker_factory =
            GetSolver().GetSolver().GetInvokeFactory(ctx, conv_problem, perf_cfg.value_or(""));
        auto kernel_handles = std::vector<Kernel>{std::begin(kernels), std::end(kernels)};

        invoker = invoker_factory(kernel_handles);
        (*invoker)(handle, invoke_ctx);
        checkNumericsOutput_();
        return;
    }

    const auto net_cfg       = conv_problem.BuildConfKey();
    const auto found_invoker = handle.GetInvoker(net_cfg, GetSolver());

    if(found_invoker)
    {
        invoker = *found_invoker;
        (*found_invoker)(handle, invoke_ctx);
        checkNumericsOutput_();
        return;
    }

    auto conv_ctx = ExecutionContext{&handle};
    conv_problem.SetupFloats(conv_ctx);

    decltype(auto) db        = GetDb(conv_ctx);
    const auto conv_solution = GetSolver().GetSolver().FindSolution(
        conv_ctx, conv_problem, db, invoke_ctx, perf_cfg.value_or(""));

    invoker =
        handle.PrepareInvoker(*conv_solution.invoker_factory, conv_solution.construction_params);
    handle.RegisterInvoker(*invoker, net_cfg, GetSolver().ToString());
    (*invoker)(handle, invoke_ctx);
    checkNumericsOutput_();
}

void Solution::RunImpl(Handle& handle,
                       const std::unordered_map<miopenTensorArgumentId_t, RunInput>& inputs,
                       Data_t workspace,
                       std::size_t workspace_size,
                       [[maybe_unused]] const MhaDescriptor& mha_desc)
{
    const Problem& problem_casted = std::get<Problem>(problem.item);

    const auto get_input_checked = [&](auto name, const std::string& name_str) {
        const auto& found = inputs.find(name);
        if(found == inputs.end())
        {
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Problem is missing " + name_str + " tensor descriptor.");
        }
        auto ret = found->second;
        if(!ret.descriptor.has_value())
            ret.descriptor = problem_casted.GetTensorDescriptorChecked(name, name_str);
        return ret;
    };

    const mha::ProblemDescription problem_description = problem_casted.AsMha();

    auto k = get_input_checked(miopenTensorMhaK, "miopenTensorMhaK");
    auto q = get_input_checked(miopenTensorMhaQ, "miopenTensorMhaQ");
    auto v = get_input_checked(miopenTensorMhaV, "miopenTensorMhaV");
    auto o = get_input_checked(miopenTensorMhaO, "miopenTensorMhaO");

    auto descaleK = get_input_checked(miopenTensorMhaDescaleK, "miopenTensorMhaDescaleK");
    auto descaleQ = get_input_checked(miopenTensorMhaDescaleQ, "miopenTensorMhaDescaleQ");
    auto descaleV = get_input_checked(miopenTensorMhaDescaleV, "miopenTensorMhaDescaleV");
    auto descaleS = get_input_checked(miopenTensorMhaDescaleS, "miopenTensorMhaDescaleS");
    auto scaleS   = get_input_checked(miopenTensorMhaScaleS, "miopenTensorMhaScaleS");

    auto m    = get_input_checked(miopenTensorMhaM, "miopenTensorMhaM");
    auto zInv = get_input_checked(miopenTensorMhaZInv, "miopenTensorMhaZInv");

    auto dropoutProbability =
        get_input_checked(miopenTensorMhaDropoutProbability, "miopenTensorMhaDropoutProbability");
    auto dropoutSeed = get_input_checked(miopenTensorMhaDropoutSeed, "miopenTensorMhaDropoutSeed");
    auto dropoutOffset =
        get_input_checked(miopenTensorMhaDropoutOffset, "miopenTensorMhaDropoutOffset");

    // reading bias buffer as an optional parameter
    Data_t biasBuffer  = nullptr;
    const auto& biasIt = inputs.find(miopenTensorMhaBias);
    if(biasIt != inputs.end())
    {
        biasBuffer = biasIt->second.buffer;
    }

    // reading a mask as an optional parameter
    miopenMhaMask_t mask = miopenMhaMaskNone;
    const auto& maskIt   = inputs.find(miopenTensorMhaMask);
    if(maskIt != inputs.end())
    {
        mask = *(static_cast<miopenMhaMask_t*>(maskIt->second.buffer));
    }

    const auto invoke_ctx = [&]() -> AnyInvokeParams {
        switch(problem_casted.GetDirection())
        {
        case miopenProblemDirectionForward: {

            auto scaleO = get_input_checked(miopenTensorMhaScaleO, "miopenTensorMhaScaleO");

            auto amaxO = get_input_checked(miopenTensorMhaAmaxO, "miopenTensorMhaAmaxO");
            auto amaxS = get_input_checked(miopenTensorMhaAmaxS, "miopenTensorMhaAmaxS");

            mha::MhaDataForward dataForward = {k.buffer,
                                               q.buffer,
                                               v.buffer,
                                               descaleK.buffer,
                                               descaleQ.buffer,
                                               descaleV.buffer,
                                               descaleS.buffer,
                                               scaleS.buffer,
                                               scaleO.buffer,
                                               dropoutProbability.buffer,
                                               dropoutSeed.buffer,
                                               dropoutOffset.buffer,
                                               biasBuffer,
                                               mask,
                                               o.buffer,
                                               amaxO.buffer,
                                               amaxS.buffer,
                                               m.buffer,
                                               zInv.buffer};

            return mha::InvokeParams(dataForward, workspace, workspace_size);
        }
        case miopenProblemDirectionBackward: {

            auto doData   = get_input_checked(miopenTensorMhaDO, "miopenTensorMhaDO");
            auto descaleO = get_input_checked(miopenTensorMhaDescaleO, "miopenTensorMhaDescaleO");
            auto descaleDO =
                get_input_checked(miopenTensorMhaDescaleDO, "miopenTensorMhaDescaleDO");
            auto descaleDS =
                get_input_checked(miopenTensorMhaDescaleDS, "miopenTensorMhaDescaleDS");

            auto scaleDS = get_input_checked(miopenTensorMhaScaleDS, "miopenTensorMhaScaleDS");
            auto scaleDQ = get_input_checked(miopenTensorMhaScaleDQ, "miopenTensorMhaScaleDQ");
            auto scaleDK = get_input_checked(miopenTensorMhaScaleDK, "miopenTensorMhaScaleDK");
            auto scaleDV = get_input_checked(miopenTensorMhaScaleDV, "miopenTensorMhaScaleDV");

            auto dq = get_input_checked(miopenTensorMhaDQ, "miopenTensorMhaDQ");
            auto dk = get_input_checked(miopenTensorMhaDK, "miopenTensorMhaDK");
            auto dv = get_input_checked(miopenTensorMhaDV, "miopenTensorMhaDV");

            auto amaxDQ = get_input_checked(miopenTensorMhaAmaxDQ, "miopenTensorMhaAmaxDQ");
            auto amaxDK = get_input_checked(miopenTensorMhaAmaxDK, "miopenTensorMhaAmaxDK");
            auto amaxDV = get_input_checked(miopenTensorMhaAmaxDV, "miopenTensorMhaAmaxDV");
            auto amaxDS = get_input_checked(miopenTensorMhaAmaxDS, "miopenTensorMhaAmaxDS");

            mha::MhaDataBackward dataBackward = {k.buffer,           q.buffer,
                                                 v.buffer,           o.buffer,
                                                 doData.buffer,      m.buffer,
                                                 zInv.buffer,        descaleK.buffer,
                                                 descaleQ.buffer,    descaleV.buffer,
                                                 descaleS.buffer,    descaleO.buffer,
                                                 descaleDO.buffer,   descaleDS.buffer,
                                                 scaleS.buffer,      scaleDS.buffer,
                                                 scaleDQ.buffer,     scaleDK.buffer,
                                                 scaleDV.buffer,     dropoutProbability.buffer,
                                                 dropoutSeed.buffer, dropoutOffset.buffer,
                                                 dq.buffer,          dk.buffer,
                                                 dv.buffer,          amaxDQ.buffer,
                                                 amaxDK.buffer,      amaxDV.buffer,
                                                 amaxDS.buffer};

            return mha::InvokeParams(dataBackward, workspace, workspace_size);
        }

        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }
    }();

    if(invoker)
    {
        (*invoker)(handle, invoke_ctx);
        return;
    }

    solver::mha::MhaForward mhaForward;
    solver::mha::MhaBackward mhaBackward;

    if(!kernels.empty())
    {
        const auto ctx          = ExecutionContext{&handle};
        const auto mha_solution = GetSolver() == mhaForward.SolverDbId()
                                      ? mhaForward.GetSolution(ctx, problem_description)
                                      : mhaBackward.GetSolution(ctx, problem_description);
        auto kernel_handles     = std::vector<Kernel>{std::begin(kernels), std::end(kernels)};

        invoker = (*mha_solution.invoker_factory)(kernel_handles);
        (*invoker)(handle, invoke_ctx);
        return;
    }

    const auto net_cfg = problem_description.MakeNetworkConfig();
    invoker            = handle.GetInvoker(net_cfg, GetSolver());

    if(invoker)
    {
        (*invoker)(handle, invoke_ctx);
        return;
    }

    auto ctx = ExecutionContext{&handle};

    const auto mha_solution = GetSolver() == mhaForward.SolverDbId()
                                  ? mhaForward.GetSolution(ctx, problem_description)
                                  : mhaBackward.GetSolution(ctx, problem_description);

    invoker =
        handle.PrepareInvoker(*mha_solution.invoker_factory, mha_solution.construction_params);
    handle.RegisterInvoker(*invoker, net_cfg, GetSolver().ToString());
    (*invoker)(handle, invoke_ctx);
}

void Solution::RunImpl(Handle& handle,
                       const std::unordered_map<miopenTensorArgumentId_t, RunInput>& inputs,
                       Data_t /*workspace*/,
                       std::size_t /*workspace_size*/,
                       const SoftmaxDescriptor& softmax_desc)
{

    const auto& problem_casted = std::get<Problem>(problem.item);

    const auto get_input_checked = [&](auto name, const std::string& name_str) {
        const auto& found = inputs.find(name);
        if(found == inputs.end())
        {
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Problem is missing " + name_str + " tensor descriptor.");
        }
        auto ret = found->second;
        if(!ret.descriptor.has_value())
            ret.descriptor = problem_casted.GetTensorDescriptorChecked(name, name_str);
        return ret;
    };

    const softmax::ProblemDescription problem_description = problem_casted.AsSoftmax();

    float alpha                        = softmax_desc.GetAlpha();
    float beta                         = softmax_desc.GetBeta();
    miopenSoftmaxAlgorithm_t algorithm = softmax_desc.GetAlgorithm();
    miopenSoftmaxMode_t mode           = softmax_desc.GetMode();

    const auto invoke_ctx = [&]() -> AnyInvokeParams {
        switch(problem_casted.GetDirection())
        {
        case miopenProblemDirectionForward: {
            auto x = get_input_checked(miopenTensorSoftmaxX, "miopenTensorSoftmaxX");
            auto y = get_input_checked(miopenTensorSoftmaxY, "miopenTensorSoftmaxY");

            return softmax::InvokeParams(
                &alpha, &beta, *x.descriptor, x.buffer, *y.descriptor, y.buffer, algorithm, mode);
        }
        case miopenProblemDirectionBackward: {
            auto y  = get_input_checked(miopenTensorSoftmaxY, "miopenTensorSoftmaxY");
            auto dy = get_input_checked(miopenTensorSoftmaxDY, "miopenTensorSoftmaxDY");
            auto dx = get_input_checked(miopenTensorSoftmaxDX, "miopenTensorSoftmaxDX");

            return softmax::InvokeParams(&alpha,
                                         &beta,
                                         *y.descriptor,
                                         y.buffer,
                                         *dy.descriptor,
                                         dy.buffer,
                                         *dx.descriptor,
                                         dx.buffer,
                                         algorithm,
                                         mode);
        }

        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }
    }();

    if(invoker)
    {
        (*invoker)(handle, invoke_ctx);
        return;
    }

    solver::softmax::Softmax regularSoftmax;
    solver::softmax::AttnSoftmax attnSoftmax;

    if(!kernels.empty())
    {
        const auto ctx              = ExecutionContext{&handle};
        const auto softmax_solution = GetSolver() == regularSoftmax.SolverDbId()
                                          ? regularSoftmax.GetSolution(ctx, problem_description)
                                          : attnSoftmax.GetSolution(ctx, problem_description);
        auto kernel_handles         = std::vector<Kernel>{std::begin(kernels), std::end(kernels)};

        invoker = (*softmax_solution.invoker_factory)(kernel_handles);
        (*invoker)(handle, invoke_ctx);
        return;
    }

    const auto net_cfg = problem_description.MakeNetworkConfig();
    invoker            = handle.GetInvoker(net_cfg, GetSolver());

    if(invoker)
    {
        (*invoker)(handle, invoke_ctx);
        return;
    }

    auto ctx = ExecutionContext{&handle};

    const auto softmax_solution = GetSolver() == regularSoftmax.SolverDbId()
                                      ? regularSoftmax.GetSolution(ctx, problem_description)
                                      : attnSoftmax.GetSolution(ctx, problem_description);

    invoker = handle.PrepareInvoker(*softmax_solution.invoker_factory,
                                    softmax_solution.construction_params);
    handle.RegisterInvoker(*invoker, net_cfg, GetSolver().ToString());
    (*invoker)(handle, invoke_ctx);
}

void Solution::RunImpl(Handle& handle,
                       const std::unordered_map<miopenTensorArgumentId_t, RunInput>& inputs,
                       Data_t /*workspace*/,
                       std::size_t /*workspace_size*/,
                       const FusedProblem& problem_)
{
    const auto buffer_getter = [&](auto id, auto&& descriptor) {
        const auto found = inputs.find(id);
        if(found == inputs.end())
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Problem is missing " + std::to_string(id) + " tensor descriptor.");
        if(found->second.descriptor.has_value() && *found->second.descriptor != descriptor)
            MIOPEN_THROW(miopenStatusNotImplemented,
                         "Providing new descriptors for a fused solution is not supported.");
        return found->second.buffer;
    };

    OperatorArgs op_args;
    const auto invoke_params = problem_.MakeInvokeParams(buffer_getter, op_args);

    if(invoker)
    {
        (*invoker)(handle, invoke_params);
        return;
    }

    const auto plan           = problem_.AsFusionPlan();
    const auto fusion_problem = FusionDescription{&plan};

    if(!kernels.empty())
    {
        const auto ctx = FusionContext{handle};
        const auto solution =
            MakeFusedSolution(ctx, solver, perf_cfg, fusion_problem, invoke_params);
        auto kernel_handles = std::vector<Kernel>{std::begin(kernels), std::end(kernels)};

        invoker = (*solution.invoker_factory)(kernel_handles);
        (*invoker)(handle, invoke_params);
        return;
    }

    const auto net_cfg = fusion_problem.MakeNetworkConfig();

    invoker = handle.GetInvoker(net_cfg, GetSolver());
    if(invoker)
    {
        (*invoker)(handle, invoke_params);
        return;
    }

    const auto ctx      = FusionContext{handle};
    const auto solution = MakeFusedSolution(ctx, solver, perf_cfg, fusion_problem, invoke_params);
    invoker = handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    handle.RegisterInvoker(*invoker, net_cfg, GetSolver().ToString());
    (*invoker)(handle, invoke_params);
}

AnyInvokeParams Solution::MakeInvokeParams(const Problem& problem_,
                                           const ConvolutionDescriptor& conv_desc,
                                           const RunInput& x,
                                           const RunInput& w,
                                           const RunInput& y,
                                           Data_t workspace,
                                           size_t workspace_size)
{
    switch(problem_.GetDirection())
    {
    case miopenProblemDirectionForward:
        return conv::DataInvokeParams(
            {*x.descriptor, x.buffer, *w.descriptor, w.buffer, *y.descriptor, y.buffer},
            workspace,
            workspace_size,
            conv_desc.attribute.gfx90aFp16alt.GetFwd());
    case miopenProblemDirectionBackward:
        return conv::DataInvokeParams(
            {*y.descriptor, y.buffer, *w.descriptor, w.buffer, *x.descriptor, x.buffer},
            workspace,
            workspace_size,
            conv_desc.attribute.gfx90aFp16alt.GetBwd());
    case miopenProblemDirectionBackwardWeights:
        return conv::WrWInvokeParams{
            {*y.descriptor, y.buffer, *x.descriptor, x.buffer, *w.descriptor, w.buffer},
            workspace,
            workspace_size,
            conv_desc.attribute.gfx90aFp16alt.GetWrW()};
    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

Problem Solution::Transpose(const Problem& problem, RunInput* x, const RunInput& w, RunInput* y)
{
    auto transposed = problem.MakeTransposed();

    std::swap(*x, *y);

    if(x->descriptor)
        transposed.RegisterTensorDescriptor(miopenTensorConvolutionX, *x->descriptor);
    if(w.descriptor)
        transposed.RegisterTensorDescriptor(miopenTensorConvolutionW, *w.descriptor);
    if(y->descriptor)
        transposed.RegisterTensorDescriptor(miopenTensorConvolutionY, *y->descriptor);

    return transposed;
}

namespace fields {
namespace header {
inline constexpr const char* Validation = "validation";
inline constexpr const char* Version    = "version";
} // namespace header
inline constexpr const char* Header    = "header";
inline constexpr const char* Time      = "time";
inline constexpr const char* Workspace = "workspace";
inline constexpr const char* Solver    = "solver";
inline constexpr const char* Problem   = "problem";
inline constexpr const char* PerfCfg   = "perf_cfg";
inline constexpr const char* Binaries  = "binaries";
inline constexpr const char* Kernels   = "kernels";
namespace kernels {
inline constexpr const char* Name           = "name";
inline constexpr const char* File           = "file";
inline constexpr const char* Program        = "program";
inline constexpr const char* LocalWorkDims  = "local_work_dims";
inline constexpr const char* GlobalWorkDims = "global_work_dims";
} // namespace kernels
} // namespace fields

void to_json(nlohmann::json& json, const Solution::SerializationMetadata& metadata)
{
    json = nlohmann::json{
        {fields::header::Validation, metadata.validation_number},
        {fields::header::Version, metadata.version},
    };
}
void from_json(const nlohmann::json& json, Solution::SerializationMetadata& metadata)
{
    json.at(fields::header::Validation).get_to(metadata.validation_number);
    json.at(fields::header::Version).get_to(metadata.version);
}

struct SerializedSolutionKernelInfo
{
    int program;
    std::vector<size_t> local_work_dims;
    std::vector<size_t> global_work_dims;
    std::string kernel_name;
    fs::path program_name;

    friend void to_json(nlohmann::json& json, const SerializedSolutionKernelInfo& kernel_info)
    {
        json = nlohmann::json{
            {fields::kernels::Program, kernel_info.program},
            {fields::kernels::Name, kernel_info.kernel_name},
            {fields::kernels::File, kernel_info.program_name.string()},
            {fields::kernels::LocalWorkDims, kernel_info.local_work_dims},
            {fields::kernels::GlobalWorkDims, kernel_info.global_work_dims},
        };

        MIOPEN_LOG_I2("Serialized solution kernel info <" << kernel_info.program_name << ":"
                                                          << kernel_info.kernel_name << ", binary "
                                                          << kernel_info.program << ">");
    }

    friend void from_json(const nlohmann::json& json, SerializedSolutionKernelInfo& kernel_info)
    {
        json.at(fields::kernels::Program).get_to(kernel_info.program);
        json.at(fields::kernels::Name).get_to(kernel_info.kernel_name);
        kernel_info.program_name = json.at(fields::kernels::File).get<std::string>();
        json.at(fields::kernels::LocalWorkDims).get_to(kernel_info.local_work_dims);
        json.at(fields::kernels::GlobalWorkDims).get_to(kernel_info.global_work_dims);

        MIOPEN_LOG_I2("Deserialized solution kernel info <"
                      << kernel_info.program_name << ":" << kernel_info.kernel_name << ", binary "
                      << kernel_info.program << ">");
    }
};

void to_json(nlohmann::json& json, const Solution& solution)
{
    json = nlohmann::json{
        {fields::Header, Solution::SerializationMetadata::Current()},
        {fields::Time, solution.time},
        {fields::Workspace, solution.workspace_required},
        {fields::Solver, solution.solver.ToString()},
        {fields::Problem, solution.problem},
    };

    if(solution.perf_cfg.has_value())
        json[fields::PerfCfg] = *solution.perf_cfg;

    if(solution.kernels.empty())
    {
        MIOPEN_LOG_I2("Solution lacks kernels information. This would slowdown the first "
                      "miopenRunSolution call after miopenLoadSolution.");
        return;
    }

    {
        const auto& first_program = solution.kernels.front().program;
        if(!first_program.IsCodeObjectInMemory() && !first_program.IsCodeObjectInFile())
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Subsequent serialization of a deserialized solution is not supported.");
    }

    auto programs         = std::vector<Program>{};
    auto prepared_kernels = std::vector<SerializedSolutionKernelInfo>{};

    std::transform(solution.kernels.begin(),
                   solution.kernels.end(),
                   std::back_inserter(programs),
                   [](const Solution::KernelInfo& sol) { return sol.program; });

    constexpr auto sorter = [](auto&& l, auto&& r) { return l.impl.get() < r.impl.get(); };
    std::sort(programs.begin(), programs.end(), sorter);
    programs.erase(std::unique(programs.begin(), programs.end()), programs.end());

    for(const auto& kernel : solution.kernels)
    {
        const auto program_it        = std::find(programs.begin(), programs.end(), kernel.program);
        auto prepared_kernel         = SerializedSolutionKernelInfo{};
        prepared_kernel.program      = std::distance(programs.begin(), program_it);
        prepared_kernel.kernel_name  = kernel.kernel_name;
        prepared_kernel.program_name = kernel.program_name;
        prepared_kernel.global_work_dims = kernel.global_work_dims;
        prepared_kernel.local_work_dims  = kernel.local_work_dims;
        prepared_kernels.emplace_back(std::move(prepared_kernel));
    }

    json[fields::Kernels] = prepared_kernels;
    auto programs_json    = nlohmann::json{};

    for(const auto& program : programs)
    {
        auto binary = nlohmann::json::binary_t{};

        if(program.IsCodeObjectInMemory())
        {
            // With disabled cache programs after build would be attached as a char vector. Same for
            // the sqlite cache.

            const auto& chars = program.GetCodeObjectBlob();
            binary.resize(chars.size());
            std::memcpy(binary.data(), chars.data(), chars.size());

            MIOPEN_LOG_I2("Serialized binary to solution blob, " << chars.size() << " bytes");
        }
        else if(program.IsCodeObjectInFile())
        {
            // Programs that have been loaded from file cache are internally interpreted
            // as read from file with a correct path.

            using Iterator      = std::istream_iterator<uint8_t>;
            constexpr auto mode = std::ios::binary | std::ios::ate;
            const auto path     = program.GetCodeObjectPathname();
            auto file           = std::ifstream(path, mode);
            const auto filesize = file.tellg();

            file.unsetf(std::ios::skipws);
            file.seekg(0, std::ios::beg);
            binary.reserve(filesize);
            binary.insert(binary.begin(), Iterator{file}, Iterator{});

            MIOPEN_LOG_I2("Serialized binary to solution blob, " << std::to_string(filesize)
                                                                 << " bytes");
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError);
        }

        programs_json.emplace_back(std::move(binary));
    }

    json[fields::Binaries] = std::move(programs_json);
}

void from_json(const nlohmann::json& json, Solution& solution)
{
    {
        const auto header = json.at(fields::Header).get<Solution::SerializationMetadata>();
        constexpr const auto check_header = Solution::SerializationMetadata::Current();

        if(header.validation_number != check_header.validation_number)
        {
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Invalid buffer has been passed to the solution deserialization.");
        }
        if(header.version != check_header.version)
        {
            MIOPEN_THROW(
                miopenStatusVersionMismatch,
                "Data from wrong version has been passed to the solution deserialization.");
        }
    }

    json.at(fields::Time).get_to(solution.time);
    json.at(fields::Workspace).get_to(solution.workspace_required);
    solution.solver = json.at(fields::Solver).get<std::string>();
    json.at(fields::Problem).get_to(solution.problem);

    const auto perf_cfg_json = json.find(fields::PerfCfg);
    solution.perf_cfg        = perf_cfg_json != json.end()
                                   ? std::optional{perf_cfg_json->get<std::string>()}
                                   : std::nullopt;

    solution.kernels.clear();
    if(const auto binaries_json = json.find(fields::Binaries); binaries_json != json.end())
    {
        auto programs = std::vector<HIPOCProgram>{};

        for(const auto& bin : *binaries_json)
        {
            const auto& binary = bin.get_ref<const nlohmann::json::binary_t&>();
            MIOPEN_LOG_I2("Derializing binary from solution blob, " << binary.size() << " bytes");
            programs.emplace_back(HIPOCProgram{"", binary});
        }

        auto kernel_infos =
            json.at(fields::Kernels).get<std::vector<SerializedSolutionKernelInfo>>();
        solution.kernels.reserve(kernel_infos.size());

        for(auto&& serialized_kernel_info : kernel_infos)
        {
            auto kernel_info             = Solution::KernelInfo{};
            kernel_info.program          = programs[serialized_kernel_info.program];
            kernel_info.local_work_dims  = std::move(serialized_kernel_info.local_work_dims);
            kernel_info.global_work_dims = std::move(serialized_kernel_info.global_work_dims);
            kernel_info.kernel_name      = std::move(serialized_kernel_info.kernel_name);
            kernel_info.program_name     = std::move(serialized_kernel_info.program_name);
            solution.kernels.emplace_back(std::move(kernel_info));
        }
    }
}
} // namespace miopen
