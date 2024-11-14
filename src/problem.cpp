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

#include <miopen/problem.hpp>

#include <miopen/activ/problem_description.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solver_finders.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/convolution.hpp>
#include <miopen/mha/problem_description.hpp>
#include <miopen/mha/solvers.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/softmax/problem_description.hpp>
#include <miopen/softmax/solvers.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/handle.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/solution.hpp>
#include <miopen/search_options.hpp>
#include <miopen/tensor_ops.hpp>

#include <nlohmann/json.hpp>

#include <boost/hof/match.hpp>

namespace miopen::debug {
/// \todo: This should be updated when a separate driver command is implemented
void LogCmdFindConvolution(const miopen::TensorDescriptor& x,
                           const miopen::TensorDescriptor& w,
                           const miopen::ConvolutionDescriptor& conv,
                           const miopen::TensorDescriptor& y,
                           miopenProblemDirection_t dir,
                           std::optional<uint64_t> solver_id);

/// \todo: This should be updated when a separate driver command is implemented
void LogCmdActivation(const miopen::TensorDescriptor& x_desc,
                      const miopen::ActivationDescriptor& activ_desc,
                      bool fwd);
} // namespace miopen::debug

namespace miopen {

namespace detail {

// Selected only with empty VariantArgs
template <int i, template <class Type> class Visitor, class... VariantArgs>
struct VisitTypeImpl
{
    template <class... Args>
    void operator()(int, Args...)
    {
        MIOPEN_THROW(miopenStatusInvalidValue);
    }
};

template <int i, template <class Type> class Visitor, class VariantArg, class... VariantArgs>
struct VisitTypeImpl<i, Visitor, VariantArg, VariantArgs...>
{
    template <class... Args>
    void operator()(int id, Args... args)
    {
        if(i == id)
        {
            Visitor<VariantArg>{args...}();
            return;
        }

        VisitTypeImpl<i + 1, Visitor, VariantArgs...>{}(id, args...);
    }
};

template <template <class Type> class Visitor, class... VariantArgs>
struct VisitType;

template <template <class Type> class Visitor, class... VariantArgs>
struct VisitType<Visitor, std::variant<VariantArgs...>>
{
    template <class... Args>
    void operator()(int id, Args... args)
    {
        detail::VisitTypeImpl<0, Visitor, VariantArgs...>{}(id, args...);
    }
};

} // namespace detail

template <template <class Type> class Visitor, class Variant, class... Args>
void VisitType(int id, Args... args)
{
    detail::VisitType<Visitor, Variant>{}(id, args...);
}

static Data_t AllocateTensor(Handle& handle,
                             const FindOptions& options,
                             std::vector<Allocator::ManageDataPtr>& owned,
                             std::vector<std::uint64_t>& owned_scalars,
                             miopenTensorArgumentId_t id,
                             const TensorDescriptor& descriptor)
{
    const auto preallocated = options.preallocated_tensors.find(id);

    if(preallocated != options.preallocated_tensors.end())
        return preallocated->second;

    if((id & miopenTensorArgumentIsScalar) == miopenTensorArgumentIsScalar)
        return &owned_scalars.emplace_back(0);

    const auto element_size = get_data_size(descriptor.GetType());
    auto buffer             = handle.Create(descriptor.GetElementSpace() * element_size);

    const auto allocated = buffer.get();
    owned.emplace_back(std::move(buffer));
    return allocated;
}

static void SortFindResults(const FindOptions& options, std::vector<Solution>& results)
{
    std::sort(results.begin(),
              results.end(),
              [&]() -> std::function<bool(const Solution&, const Solution&)> {
                  switch(options.results_order)
                  {
                  case miopenFindResultsOrderByTime:
                      return [](auto&& l, auto&& r) { return l.GetTime() < r.GetTime(); };
                  case miopenFindResultsOrderByWorkspaceSize:
                      return [](auto&& l, auto&& r) {
                          return l.GetWorkspaceSize() < r.GetWorkspaceSize();
                      };
                  }
                  MIOPEN_THROW(miopenStatusNotImplemented);
              }());
}

std::vector<Solution>
Problem::FindSolutions(Handle& handle, const FindOptions& options, std::size_t max_solutions) const
{
    auto owned_buffers = std::vector<Allocator::ManageDataPtr>{};
    auto owned_scalars = std::vector<std::uint64_t>{};
    auto buffers       = std::unordered_map<miopenTensorArgumentId_t, Data_t>{};

    const auto allocate = [&](auto id, auto&& descriptor) {
        auto buffer = AllocateTensor(handle, options, owned_buffers, owned_scalars, id, descriptor);
        buffers.emplace(id, buffer);
        return buffer;
    };

    for(const auto& pair : tensor_descriptors)
        allocate(pair.first, pair.second);

    auto ret = std::visit(
        boost::hof::match(
            [&](const ConvolutionDescriptor& op_desc) {
                if(op_desc.mode == miopenTranspose)
                    return MakeTransposed().FindSolutionsImpl(
                        handle, options, max_solutions, buffers, op_desc);
                else
                    return FindSolutionsImpl(handle, options, max_solutions, buffers, op_desc);
            },
            [&](const SoftmaxDescriptor& op_desc) {
                return FindSolutionsImpl(handle, options, max_solutions, buffers, op_desc);
            },
            [&](const ActivationDescriptor& /*op_desc*/) -> std::vector<Solution> {
                MIOPEN_THROW(miopenStatusNotImplemented);
            },
            [&](const MhaDescriptor& op_desc) {
                return FindSolutionsImpl(handle, options, max_solutions, buffers, op_desc);
            },
            [&](const BiasDescriptor& /*op_desc*/) -> std::vector<Solution> {
                MIOPEN_THROW(miopenStatusNotImplemented);
            },
            [&](const BatchnormDescriptor& /*op_desc*/) -> std::vector<Solution> {
                MIOPEN_THROW(miopenStatusNotImplemented);
            }),
        operator_descriptor);

    owned_buffers.resize(0);
    SortFindResults(options, ret);
    return ret;
}

const TensorDescriptor&
Problem::GetTensorDescriptorChecked(miopenTensorArgumentId_t name,
                                    [[maybe_unused]] const std::string& name_str) const
{
    const auto found = tensor_descriptors.find(name);
    if(found == tensor_descriptors.end())
    {
        MIOPEN_THROW(miopenStatusInvalidValue,
                     "Problem is missing " + name_str + " tensor descriptor.");
    }
    return found->second;
}

const TensorDescriptor& Problem::GetTensorDescriptor(miopenTensorArgumentId_t name,
                                                     const TensorDescriptor& default_value) const
{
    const auto found = tensor_descriptors.find(name);
    if(found == tensor_descriptors.end())
        return default_value;
    return found->second;
}

Problem Problem::MakeTransposed() const
{
    auto transposed = Problem{};
    transposed.SetOperatorDescriptor(GetOperatorDescriptor());

    switch(GetDirection())
    {
    case miopenProblemDirectionForward:
        transposed.SetDirection(miopenProblemDirectionBackward);
        break;
    case miopenProblemDirectionBackward:
        transposed.SetDirection(miopenProblemDirectionForward);
        break;
    case miopenProblemDirectionBackwardWeights:
        transposed.SetDirection(miopenProblemDirectionBackwardWeights);
        break;
    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }

    transposed.tensor_descriptors.reserve(tensor_descriptors.size());
    for(const auto& descriptor : tensor_descriptors)
        transposed.tensor_descriptors.emplace(descriptor.first, descriptor.second);

    const auto transpose_tensors = boost::hof::match(
        [&](const ConvolutionDescriptor& op_desc) { return transposed.TransposeImpl(op_desc); },
        [](auto&&) { MIOPEN_THROW(miopenStatusNotImplemented); });

    std::visit(transpose_tensors, operator_descriptor);

    return transposed;
}

void Problem::TransposeImpl(const ConvolutionDescriptor& /*conv_desc*/)
{
    std::swap(tensor_descriptors.at(miopenTensorConvolutionX),
              tensor_descriptors.at(miopenTensorConvolutionY));
}

AnyInvokeParams Problem::MakeConvInvokeParams(const TensorDescriptor& x_desc,
                                              Data_t x,
                                              const TensorDescriptor& w_desc,
                                              Data_t w,
                                              const TensorDescriptor& y_desc,
                                              Data_t y,
                                              Data_t workspace,
                                              size_t workspace_size) const
{
    const auto& conv_desc = std::get<ConvolutionDescriptor>(operator_descriptor);

    switch(GetDirection())
    {
    case miopenProblemDirectionForward:
        return conv::DataInvokeParams({x_desc, x, w_desc, w, y_desc, y},
                                      workspace,
                                      workspace_size,
                                      conv_desc.attribute.gfx90aFp16alt.GetFwd());
    case miopenProblemDirectionBackward:
        return conv::DataInvokeParams({y_desc, y, w_desc, w, x_desc, x},
                                      workspace,
                                      workspace_size,
                                      conv_desc.attribute.gfx90aFp16alt.GetBwd());
    case miopenProblemDirectionBackwardWeights:
        return conv::WrWInvokeParams{{y_desc, y, x_desc, x, w_desc, w},
                                     workspace,
                                     workspace_size,
                                     conv_desc.attribute.gfx90aFp16alt.GetWrW()};
    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

conv::ProblemDescription Problem::AsConvolution() const
{
    const auto& conv_desc = std::get<ConvolutionDescriptor>(operator_descriptor);

    const auto& x_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    const auto& y_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    const auto conv_dir = static_cast<conv::Direction>(direction);
    return conv_dir == conv::Direction::Forward
               ? conv::ProblemDescription(x_desc, w_desc, y_desc, conv_desc, conv_dir)
               : conv::ProblemDescription(y_desc, w_desc, x_desc, conv_desc, conv_dir);
}

activ::ProblemDescription Problem::AsActivation() const
{
    const auto& activ_desc = std::get<ActivationDescriptor>(operator_descriptor);

    const auto& x_desc =
        GetTensorDescriptorChecked(miopenTensorActivationX, "miopenTensorActivationX");
    const auto& y_desc = GetTensorDescriptor(miopenTensorActivationY, x_desc);

    if(direction == miopenProblemDirectionForward)
    {
        return {activ_desc, x_desc, y_desc};
    }
    else
    {
        const auto& dx_desc = GetTensorDescriptor(miopenTensorActivationDX, x_desc);
        const auto& dy_desc = GetTensorDescriptor(miopenTensorActivationDY, dx_desc);

        return {activ_desc, x_desc, y_desc, dx_desc, dy_desc};
    }
}

mha::ProblemDescription Problem::AsMha() const
{
    const auto& mha_desc = std::get<MhaDescriptor>(operator_descriptor);

    float scale = mha_desc.GetScale();

    const auto& kDesc = GetTensorDescriptorChecked(miopenTensorMhaK, "miopenTensorMhaK");
    const auto& qDesc = GetTensorDescriptorChecked(miopenTensorMhaQ, "miopenTensorMhaQ");
    const auto& vDesc = GetTensorDescriptorChecked(miopenTensorMhaV, "miopenTensorMhaV");

    const auto& descaleKDesc =
        GetTensorDescriptorChecked(miopenTensorMhaDescaleK, "miopenTensorMhaDescaleK");
    const auto& descaleQDesc =
        GetTensorDescriptorChecked(miopenTensorMhaDescaleQ, "miopenTensorMhaDescaleQ");
    const auto& descaleVDesc =
        GetTensorDescriptorChecked(miopenTensorMhaDescaleV, "miopenTensorMhaDescaleV");
    const auto& descaleSDesc =
        GetTensorDescriptorChecked(miopenTensorMhaDescaleS, "miopenTensorMhaDescaleS");

    const auto& scaleSDesc =
        GetTensorDescriptorChecked(miopenTensorMhaScaleS, "miopenTensorMhaScaleS");

    const auto& dpDesc = GetTensorDescriptorChecked(miopenTensorMhaDropoutProbability,
                                                    "miopenTensorMhaDropoutProbability");
    const auto& dsDesc =
        GetTensorDescriptorChecked(miopenTensorMhaDropoutSeed, "miopenTensorMhaDropoutSeed");
    const auto& doffDesc =
        GetTensorDescriptorChecked(miopenTensorMhaDropoutOffset, "miopenTensorMhaDropoutOffset");

    const auto& oDesc    = GetTensorDescriptorChecked(miopenTensorMhaO, "miopenTensorMhaO");
    const auto& mDesc    = GetTensorDescriptorChecked(miopenTensorMhaM, "miopenTensorMhaM");
    const auto& zInvDesc = GetTensorDescriptorChecked(miopenTensorMhaZInv, "miopenTensorMhaZInv");

    if(GetDirection() == miopenProblemDirectionForward)
    {
        mha::MhaInputDescsForward mhaInputDescsForward = {
            kDesc,
            qDesc,
            vDesc,
            descaleKDesc,
            descaleQDesc,
            descaleVDesc,
            descaleSDesc,
            scaleSDesc,
            GetTensorDescriptorChecked(miopenTensorMhaScaleO, "miopenTensorMhaScaleO"),
            scale,
            dpDesc,
            dsDesc,
            doffDesc,
            GetTensorDescriptor(miopenTensorMhaBias, TensorDescriptor()),
            oDesc,
            GetTensorDescriptorChecked(miopenTensorMhaAmaxO, "miopenTensorMhaAmaxO"),
            GetTensorDescriptorChecked(miopenTensorMhaAmaxS, "miopenTensorMhaAmaxS"),
            mDesc,
            zInvDesc};

        return {mhaInputDescsForward};
    }
    else
    {
        mha::MhaInputDescsBackward mhaInputDescsBackward = {
            kDesc,
            qDesc,
            vDesc,
            oDesc,
            GetTensorDescriptorChecked(miopenTensorMhaDO, "miopenTensorMhaDO"),
            mDesc,
            zInvDesc,
            descaleKDesc,
            descaleQDesc,
            descaleVDesc,
            descaleSDesc,
            GetTensorDescriptorChecked(miopenTensorMhaDescaleO, "miopenTensorMhaDescaleO"),
            GetTensorDescriptorChecked(miopenTensorMhaDescaleDO, "miopenTensorMhaDescaleDO"),
            GetTensorDescriptorChecked(miopenTensorMhaDescaleDS, "miopenTensorMhaDescaleDS"),
            scaleSDesc,
            GetTensorDescriptorChecked(miopenTensorMhaScaleDS, "miopenTensorMhaScaleDS"),
            GetTensorDescriptorChecked(miopenTensorMhaScaleDQ, "miopenTensorMhaScaleDQ"),
            GetTensorDescriptorChecked(miopenTensorMhaScaleDK, "miopenTensorMhaScaleDK"),
            GetTensorDescriptorChecked(miopenTensorMhaScaleDV, "miopenTensorMhaScaleDV"),
            scale,
            dpDesc,
            dsDesc,
            doffDesc,
            GetTensorDescriptorChecked(miopenTensorMhaDQ, "miopenTensorMhaDQ"),
            GetTensorDescriptorChecked(miopenTensorMhaDK, "miopenTensorMhaDK"),
            GetTensorDescriptorChecked(miopenTensorMhaDV, "miopenTensorMhaDV"),
            GetTensorDescriptorChecked(miopenTensorMhaAmaxDQ, "miopenTensorMhaAmaxDQ"),
            GetTensorDescriptorChecked(miopenTensorMhaAmaxDK, "miopenTensorMhaAmaxDK"),
            GetTensorDescriptorChecked(miopenTensorMhaAmaxDV, "miopenTensorMhaAmaxDV"),
            GetTensorDescriptorChecked(miopenTensorMhaAmaxDS, "miopenTensorMhaAmaxDS")};

        return {mhaInputDescsBackward};
    }
}

softmax::ProblemDescription Problem::AsSoftmax() const
{
    const auto& softmax_desc = std::get<SoftmaxDescriptor>(operator_descriptor);

    float alpha = softmax_desc.GetAlpha();
    float beta  = softmax_desc.GetBeta();

    softmax::ProblemDescription problem_description =
        (GetDirection() == miopenProblemDirectionForward)
            ? softmax::ProblemDescription(
                  &alpha,
                  &beta,
                  GetTensorDescriptorChecked(miopenTensorSoftmaxX, "miopenTensorSoftmaxX"),
                  GetTensorDescriptorChecked(miopenTensorSoftmaxY, "miopenTensorSoftmaxY"),
                  softmax_desc.GetAlgorithm(),
                  softmax_desc.GetMode())
            : softmax::ProblemDescription(
                  &alpha,
                  &beta,
                  GetTensorDescriptorChecked(miopenTensorSoftmaxY, "miopenTensorSoftmaxY"),
                  GetTensorDescriptorChecked(miopenTensorSoftmaxDY, "miopenTensorSoftmaxDY"),
                  GetTensorDescriptorChecked(miopenTensorSoftmaxDX, "miopenTensorSoftmaxDX"),
                  softmax_desc.GetAlgorithm(),
                  softmax_desc.GetMode());
    return problem_description;
}

std::vector<Solution> Problem::FindSolutionsImpl(Handle& handle,
                                                 const FindOptions& options,
                                                 std::size_t max_solutions,
                                                 const Buffers& buffers,
                                                 const ConvolutionDescriptor& conv_desc) const
{
    if(tensor_descriptors.size() != 3)
    {
        MIOPEN_THROW(miopenStatusInvalidValue,
                     "Convolution problem should have exactly three tensor descriptors.");
    }

    auto x_desc = GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    auto y_desc = GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    auto x        = buffers.at(miopenTensorConvolutionX);
    const auto& w = buffers.at(miopenTensorConvolutionW);
    auto y        = buffers.at(miopenTensorConvolutionY);

    if(conv_desc.mode == miopenTranspose)
        std::swap(x, y);

    const auto conv_problem = AsConvolution();

    ValidateGroupCount(x_desc, w_desc, conv_desc);

    std::size_t workspace_size;
    Allocator::ManageDataPtr owned_workspace;
    Data_t workspace;

    if(options.preallocated_workspace)
    {
        workspace      = options.preallocated_workspace->buffer;
        workspace_size = options.preallocated_workspace->size;
    }
    else
    {
        auto tmp_ctx             = ExecutionContext{&handle};
        const auto workspace_max = conv_desc.GetWorkSpaceSize(tmp_ctx, conv_problem);
        workspace_size           = std::min(options.workspace_limit, workspace_max);
        owned_workspace          = workspace_size != 0 ? handle.Create(workspace_size) : nullptr;
        workspace                = owned_workspace.get();
    }

    auto ctx = ExecutionContext{&handle};
    conv_problem.SetupFloats(ctx);
    ctx.do_search = options.exhaustive_search;

    const auto invoke_ctx =
        MakeConvInvokeParams(x_desc, x, w_desc, w, y_desc, y, workspace, workspace_size);

    auto results =
        FindConvolution(ctx, conv_problem, invoke_ctx, max_solutions, options.attach_binaries);

    for(auto& result : results)
    {
        result.SetProblem({*this});

        if(result.GetKernels().empty())
        {
            // If find-db was used binaries and invoker have not been set.
            // This would make binaries not serialized and invoker not cached.
            // So we prepare them here.

            auto db = GetDb(ctx);
            const auto conv_solution =
                result.GetSolver().GetSolver().FindSolution(ctx, conv_problem, db, invoke_ctx);

            std::vector<Program> programs;
            auto invoker = handle.PrepareInvoker(*conv_solution.invoker_factory,
                                                 conv_solution.construction_params,
                                                 options.attach_binaries ? &programs : nullptr);
            result.SetInvoker(std::move(invoker), programs, conv_solution.construction_params);
        }
    }
    return results;
}

std::vector<Solution>
Problem::FindSolutionsImpl(Handle& handle,
                           [[maybe_unused]] const FindOptions& options,
                           std::size_t max_solutions,
                           [[maybe_unused]] const Buffers& buffers,
                           [[maybe_unused]] const SoftmaxDescriptor& softmax_desc) const
{
    auto ret = std::vector<Solution>();

    auto ctx = ExecutionContext{&handle};

    const softmax::ProblemDescription problem_description = AsSoftmax();

    const auto algo = AlgorithmName{"Softmax"};

    static solver::softmax::AttnSoftmax attnSoftmaxSolver;
    static solver::softmax::Softmax regularSoftmaxSolver;

    std::vector<solver::softmax::SoftmaxSolver*> solvers;

    solvers.push_back(&attnSoftmaxSolver);
    solvers.push_back(&regularSoftmaxSolver);

    for(auto solver : solvers)
    {
        if(!solver->IsApplicable(ctx, problem_description))
        {
            MIOPEN_LOG_I2(solver->SolverDbId() << ": Not applicable");
            continue;
        }

        auto solution = Solution();

        /// \todo time measurement will be done later. For now we set less time for attention
        /// softmax and slightly bigger for regular
        solution.SetTime(solver == &attnSoftmaxSolver ? 1.0f : 2.0f);
        solution.SetWorkspaceSize(solver->GetWorkspaceSize(ctx, problem_description));
        solution.SetSolver(solver->SolverDbId());
        solution.SetProblem({*this});

        MIOPEN_LOG_I("Found solution: " << solution.GetSolver().ToString() << " , "
                                        << solution.GetWorkspaceSize() << ", "
                                        << solution.GetTime());

        ret.emplace_back(std::move(solution));

        if(ret.size() >= max_solutions)
        {
            break;
        }
    }

    return ret;
}

std::vector<Solution>
Problem::FindSolutionsImpl(Handle& handle,
                           [[maybe_unused]] const FindOptions& options,
                           std::size_t max_solutions,
                           [[maybe_unused]] const Buffers& buffers,
                           [[maybe_unused]] const MhaDescriptor& mha_desc) const
{
    auto ret = std::vector<Solution>{};

    auto ctx = ExecutionContext{&handle};

    const mha::ProblemDescription problem_description = AsMha();

    const auto algo = AlgorithmName{"Mha"};

    static solver::mha::MhaForward mhaForwardSolver;
    static solver::mha::MhaBackward mhaBackwardSolver;

    std::vector<solver::mha::MhaSolver*> solvers = {&mhaForwardSolver, &mhaBackwardSolver};

    for(auto solver : solvers)
    {
        if(!solver->IsApplicable(ctx, problem_description))
        {
            MIOPEN_LOG_I2(solver->SolverDbId() << ": Not applicable");
            continue;
        }

        auto solution = Solution();

        /// \todo time measurement could be done later. For now we set less time for attention
        /// softmax and slightly bigger for regular
        solution.SetTime(1.0f);

        solution.SetWorkspaceSize(solver->GetWorkspaceSize(ctx, problem_description));
        solution.SetSolver(solver->SolverDbId());
        solution.SetProblem({*this});

        MIOPEN_LOG_I("Found solution: " << solution.GetSolver().ToString() << " , "
                                        << solution.GetWorkspaceSize() << ", "
                                        << solution.GetTime());

        ret.emplace_back(std::move(solution));

        if(ret.size() >= max_solutions)
        {
            break;
        }
    }

    return ret;
}

namespace {
inline bool IsValidFilterChannelNumber(const TensorDescriptor& x,
                                       const TensorDescriptor& w,
                                       const std::optional<miopenTensorLayout_t>& layout,
                                       const int groups)
{
    if(layout == miopenTensorNCHW      //
       || layout == miopenTensorNCHWc4 //
       || layout == miopenTensorNCHWc8)
    {
        return x.GetLengths()[1] / groups == w.GetLengths()[1];
    }

    if(layout == miopenTensorCHWNc4 //
       || layout == miopenTensorCHWNc8)
    {
        return x.GetLengths()[1] / groups == w.GetLengths()[0];
    }

    return true;
}

inline bool IsValidGroupCount(const TensorDescriptor& x,
                              const TensorDescriptor& w,
                              const std::optional<miopenTensorLayout_t>& layout,
                              const int groups)
{
    if(groups > 1) // Optimize for speed
    {
        if(x.GetLengths()[1] % groups != 0)
            return false;

        if(layout == miopenTensorNCHW      //
           || layout == miopenTensorNCHWc4 //
           || layout == miopenTensorNCHWc8)
            return w.GetLengths()[0] % groups == 0;

        if(layout == miopenTensorCHWNc4 //
           || layout == miopenTensorCHWNc8)
            return w.GetLengths()[3] % groups == 0;
    }
    return true;
}
} // namespace

void Problem::ValidateGroupCount(const TensorDescriptor& x,
                                 const TensorDescriptor& w,
                                 const ConvolutionDescriptor& conv)
{
    const auto layout = w.GetLayoutEnum();
    const auto groups = conv.group_count;
    assert(groups > 0);

    const auto ok_c = IsValidFilterChannelNumber(x, w, layout, groups);
    const auto ok_g = IsValidGroupCount(x, w, layout, groups);

    if(ok_c && ok_g)
        return;

    MIOPEN_LOG_W(w.GetLayout_str() << "w {" << w.ToString() << "}, " //
                                   << "x {" << x.ToString() << "}, " //
                                   << "groups = " << conv.group_count);
    if(!ok_c)
        MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    if(!ok_g)
        MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
}

void Problem::LogDriverCommand() const
{
    std::visit([&](const auto& op_desc) { LogDriverCommand(op_desc); }, operator_descriptor);
}

void Problem::LogDriverCommand(const ConvolutionDescriptor& conv_desc) const
{
    const auto& x_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    const auto& y_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");
    miopen::debug::LogCmdFindConvolution(x_desc, w_desc, conv_desc, y_desc, direction, 0);
}

void Problem::LogDriverCommand(const ActivationDescriptor& descriptor) const
{
    const auto& x_desc =
        GetTensorDescriptorChecked(miopenTensorActivationX, "miopenTensorActivationX");
    miopen::debug::LogCmdActivation(x_desc, descriptor, direction == miopenProblemDirectionForward);
}

void Problem::LogDriverCommand(const BiasDescriptor& descriptor) const
{
    /// \todo: log actual driver command
    std::ignore = descriptor;
}

void Problem::LogDriverCommand(const BatchnormDescriptor& descriptor) const
{
    /// \todo: log actual driver command
    std::ignore = descriptor;
}

void Problem::LogDriverCommand(const MhaDescriptor& descriptor) const
{
    /// \todo: log actual driver command
    std::ignore = descriptor;
}

void Problem::LogDriverCommand(const SoftmaxDescriptor& descriptor) const
{
    /// \todo: log actual driver command
    std::ignore = descriptor;
}

void to_json(nlohmann::json& json, const BiasDescriptor&) { json = nlohmann::json{}; }

void from_json(const nlohmann::json&, BiasDescriptor&) {}

void to_json(nlohmann::json& j, const BatchnormDescriptor& descriptor)
{
    j = nlohmann::json{
        {"mode", descriptor.mode},
        {"runningMeanVariance", descriptor.runningMeanVariance},
    };
}

void from_json(const nlohmann::json& j, BatchnormDescriptor& descriptor)
{
    j.at("mode").get_to(descriptor.mode);
    j.at("runningMeanVariance").get_to(descriptor.runningMeanVariance);
}

void to_json(nlohmann::json& json, const Problem& problem)
{
    json = nlohmann::json{
        {"direction", problem.direction},
        {"tensors", problem.tensor_descriptors},
        {"primitive", problem.operator_descriptor.index()},
    };

    auto operator_serialization = [&](auto&& op) { json["operator"] = op; };
    std::visit(operator_serialization, problem.operator_descriptor);
}

namespace detail {
template <class Descriptor>
struct OperatorDescriptorDeserializer
{
    const nlohmann::json* json;
    OperatorDescriptor* descriptor;

    inline void operator()() const { *descriptor = json->get<Descriptor>(); }
};
} // namespace detail

void from_json(const nlohmann::json& json, Problem& problem)
{
    json.at("direction").get_to(problem.direction);
    json.at("tensors").get_to(problem.tensor_descriptors);

    const auto primitive = json.at("primitive").get<int>();
    auto operator_json   = json.at("operator");

    VisitType<detail::OperatorDescriptorDeserializer, OperatorDescriptor>(
        primitive, &operator_json, &problem.operator_descriptor);
}

void to_json(nlohmann::json& json, const FusedProblem& problem)
{
    json = nlohmann::json{
        {"problems", problem.problems},
    };
}

void from_json(const nlohmann::json& json, FusedProblem& problem)
{
    json.at("problems").get_to(problem.problems);
}

void to_json(nlohmann::json& json, const ProblemContainer& problem)
{
    json = nlohmann::json{
        {"problem_type", problem.item.index()},
    };

    auto operator_serialization = [&](auto&& op) { json["value"] = op; };
    std::visit(operator_serialization, problem.item);
}

namespace detail {
template <class Problem>
struct ProblemDeserializer
{
    const nlohmann::json* json;
    ProblemContainer::Item* problem;

    void operator()() const { *problem = json->get<Problem>(); }
};
} // namespace detail

void from_json(const nlohmann::json& json, ProblemContainer& problem)
{
    const auto type = json.at("problem_type").get<int>();
    auto value      = json.at("value");

    VisitType<detail::ProblemDeserializer, ProblemContainer::Item>(type, &value, &problem.item);
}

void Problem::CalculateOutput()
{
    if(!HasInput())
        return;

    std::visit(
        boost::hof::match(
            [&](const ConvolutionDescriptor& conv) {
                const auto& in = GetInput();
                conv.GetForwardOutputTensor(in,
                                            GetTensorDescriptorChecked(miopenTensorConvolutionW,
                                                                       "miopenTensorConvolutionW"),
                                            in.GetType());
            },
            [&](const ActivationDescriptor&) {
                RegisterTensorDescriptor(GetOutputId(), GetInput());
            },

            [&](const MhaDescriptor&) { RegisterTensorDescriptor(GetOutputId(), GetInput()); },
            [&](const SoftmaxDescriptor&) { RegisterTensorDescriptor(GetOutputId(), GetInput()); },
            [&](const BiasDescriptor&) { RegisterTensorDescriptor(GetOutputId(), GetInput()); },
            [&](const BatchnormDescriptor&) {
                RegisterTensorDescriptor(GetOutputId(), GetInput());
            }),
        operator_descriptor);
}

miopenTensorArgumentId_t Problem::GetInputId() const
{
    return std::visit(boost::hof::match(
                          [&](const ConvolutionDescriptor&) {
                              return direction == miopenProblemDirectionForward
                                         ? miopenTensorConvolutionX
                                         : miopenTensorConvolutionY;
                          },
                          [&](const ActivationDescriptor&) {
                              return direction == miopenProblemDirectionForward
                                         ? miopenTensorActivationX
                                         : miopenTensorActivationDY;
                          },
                          [&](const BiasDescriptor&) {
                              return direction == miopenProblemDirectionForward ? miopenTensorBiasX
                                                                                : miopenTensorBiasY;
                          },
                          [&](const BatchnormDescriptor&) {
                              return direction == miopenProblemDirectionBackward
                                         ? miopenTensorBatchnormDY
                                         : miopenTensorBatchnormX;
                          },
                          [](const MhaDescriptor&) { return miopenTensorMhaK; },
                          [](const SoftmaxDescriptor&) { return miopenTensorSoftmaxX; }),
                      operator_descriptor);
}

miopenTensorArgumentId_t Problem::GetOutputId() const
{
    return std::visit(boost::hof::match(
                          [&](const ConvolutionDescriptor&) {
                              return direction == miopenProblemDirectionForward
                                         ? miopenTensorConvolutionY
                                         : miopenTensorConvolutionX;
                          },
                          [&](const ActivationDescriptor&) {
                              return direction == miopenProblemDirectionForward
                                         ? miopenTensorActivationY
                                         : miopenTensorActivationDX;
                          },
                          [&](const BiasDescriptor&) {
                              return direction == miopenProblemDirectionForward ? miopenTensorBiasY
                                                                                : miopenTensorBiasX;
                          },
                          [&](const BatchnormDescriptor&) {
                              return direction == miopenProblemDirectionBackward
                                         ? miopenTensorBatchnormDX
                                         : miopenTensorBatchnormY;
                          },
                          [](const MhaDescriptor&) { return miopenTensorMhaO; },
                          [](const SoftmaxDescriptor&) { return miopenTensorSoftmaxY; }),
                      operator_descriptor);
}

void FusedProblem::PropagateDescriptors()
{
    for(auto i = 0; i < problems.size(); ++i)
    {
        auto& cur = problems[i];

        if(i > 0 && !cur.HasInput())
        {
            auto& prev = problems[i - 1];
            if(prev.HasOutput())
                cur.RegisterTensorDescriptor(cur.GetInputId(), prev.GetOutput());
        }

        if(cur.HasInput() && !cur.HasOutput())
            cur.CalculateOutput();
    }
}

std::vector<Solution> FusedProblem::FindSolutions(Handle& handle,
                                                  const FindOptions& options,
                                                  std::size_t max_solutions) const
{
    auto solutions = [&]() {
        OperatorArgs params;
        auto owned_buffers = std::vector<Allocator::ManageDataPtr>{};
        auto owned_scalars = std::vector<std::uint64_t>{};

        const auto make_invoke_params = [&]() {
            auto buffer_allocator = [&](auto id, auto&& desc) {
                return AllocateTensor(handle, options, owned_buffers, owned_scalars, id, desc);
            };

            return MakeInvokeParams(buffer_allocator, params);
        };

        return AsFusionPlan().Find(handle, make_invoke_params, options);
    }();

    for(auto& solution : solutions)
    {
        solution.SetProblem({*this});
        MIOPEN_LOG_I("Found solution: " << solution.GetSolver().ToString() << " , "
                                        << solution.GetWorkspaceSize() << ", "
                                        << solution.GetTime());
    }

    SortFindResults(options, solutions);
    solutions.resize(std::min(solutions.size(), max_solutions));
    return solutions;
}

void FusedProblem::AddProblemToPlan(FusionPlanDescriptor& plan, const Problem& problem)
{
    std::visit(
        boost::hof::match(
            [&](const ConvolutionDescriptor& conv_desc) {
                plan.AddOp(std::make_shared<ConvForwardOpDescriptor>(
                    conv_desc,
                    problem.GetTensorDescriptorChecked(miopenTensorConvolutionW,
                                                       "miopenTensorConvolutionW")));
            },
            [&](const ActivationDescriptor& activ_desc) {
                if(problem.GetDirection() == miopenProblemDirectionForward)
                    plan.AddOp(std::make_shared<ActivFwdFusionOpDescriptor>(activ_desc.GetMode()));
                else
                    plan.AddOp(std::make_shared<ActivBwdFusionOpDescriptor>(activ_desc.GetMode()));
            },
            [&](const BiasDescriptor&) {
                plan.AddOp(std::make_shared<BiasFusionOpDescriptor>(
                    problem.GetTensorDescriptorChecked(miopenTensorBias, "miopenTensorBias")));
            },
            [&](const MhaDescriptor&) {
                // Not implemented
                assert(false);
                MIOPEN_THROW(miopenStatusNotImplemented, "Mha is not implemented for FusedProblem");
            },
            [&](const SoftmaxDescriptor&) {
                // Not implemented
                assert(false);
                MIOPEN_THROW(miopenStatusNotImplemented,
                             "Softmax is not implemented for FusedProblem");
            },
            [&](const BatchnormDescriptor& descriptor) {
                switch(problem.GetDirection())
                {
                case miopenProblemDirectionForward:
                    plan.AddOp(std::make_shared<BatchNormFwdTrainFusionOpDescriptor>(
                        descriptor.mode, descriptor.runningMeanVariance));
                    break;
                case miopenProblemDirectionBackward:
                    plan.AddOp(
                        std::make_shared<BatchNormBwdTrainFusionOpDescriptor>(descriptor.mode));
                    break;
                case miopenProblemDirectionInference: {
                    auto smv = problem.GetTensorDescriptorChecked(
                        miopenTensorBatchnormEstimatedMean, "miopenTensorBatchnormEstimatedMean");
                    plan.AddOp(std::make_shared<BatchNormInferenceFusionOpDescriptor>(
                        descriptor.mode, smv));
                    break;
                }
                default:
                    MIOPEN_THROW(miopenStatusBadParm,
                                 "Batchnorm only has forward, backward and inference directions");
                }
            }),

        problem.operator_descriptor);
}

fusion::FusionInvokeParams FusedProblem::MakeInvokeParams(
    const std::function<Data_t(miopenTensorArgumentId_t, const TensorDescriptor&)>& buffer_getter,
    OperatorArgs& operator_args) const
{
    auto buffers   = std::unordered_map<miopenTensorArgumentId_t, Data_t>{};
    auto& in_desc  = problems.front().GetInput();
    auto& out_desc = problems.back().GetOutput();

    const auto get_buffer = [&](auto id, auto&& descriptor) {
        if(const auto found = buffers.find(id); found != buffers.end())
            return found->second;
        auto buffer = buffer_getter(id, descriptor);
        buffers.emplace(id, buffer);
        return buffer;
    };

    // This is not used right now, but there is a PR using it already and it is an example on how to
    // get a scalar.
    const auto get_scalar = [&](auto id, auto type_marker) {
        // This is hacky because we lack separate way to pass them through API
        return *reinterpret_cast<std::decay_t<decltype(type_marker)>*>(
            get_buffer(id, TensorDescriptor()));
    };

    bool gfx90aaltimpl = false;
    auto in            = get_buffer(GetInputId(), in_desc);
    auto out           = get_buffer(GetOutputId(), out_desc);

    for(const auto& problem : problems)
    {
        for(const auto& pair : problem.tensor_descriptors)
            if(pair.first != problem.GetInputId() && pair.first != problem.GetOutputId())
                get_buffer(pair.first, pair.second);

        std::visit(
            boost::hof::match(
                [&](const ConvolutionDescriptor& conv_desc) {
                    gfx90aaltimpl = conv_desc.attribute.gfx90aFp16alt.GetFwd();

                    const auto wei_ptr = buffers.at(miopenTensorConvolutionW);
                    operator_args.params.emplace_back(
                        std::make_unique<miopen::fusion::ConvolutionOpInvokeParam>(wei_ptr));
                },
                [&](const ActivationDescriptor& activ_desc) {
                    const auto alpha = activ_desc.GetAlpha();
                    const auto beta  = activ_desc.GetBeta();
                    const auto gamma = activ_desc.GetGamma();

                    if(problem.GetDirection() == miopenProblemDirectionForward)
                    {
                        operator_args.params.emplace_back(
                            std::make_unique<miopen::fusion::ActivationOpInvokeParam>(
                                alpha, beta, gamma));
                    }
                    else
                    {
                        const auto x = buffers.at(miopenTensorActivationX);
                        const auto y = buffers.at(miopenTensorActivationY);

                        operator_args.params.emplace_back(
                            std::make_unique<miopen::fusion::ActivationBwdOpInvokeParam>(
                                y, x, alpha, beta, gamma));
                    }
                },
                [&](const BiasDescriptor&) {
                    const auto bias_ptr = buffers.at(miopenTensorBias);
                    operator_args.params.emplace_back(
                        std::make_unique<miopen::fusion::BiasOpInvokeParam>(bias_ptr));
                },

                [&](const MhaDescriptor&) {
                    // Not implemented
                    assert(false);
                    MIOPEN_THROW(miopenStatusNotImplemented,
                                 "Mha is not implemented for FusedProblem");
                },
                [&](const SoftmaxDescriptor&) {
                    // Not implemented
                    assert(false);
                    MIOPEN_THROW(miopenStatusNotImplemented,
                                 "Softmax is not implemented for FusedProblem");
                },
                [&](const BatchnormDescriptor& /*descriptor*/) {
                    /// \todo: fix this to pass actual values
                    switch(problem.GetDirection())
                    {
                    case miopenProblemDirectionForward:
                        operator_args.params.emplace_back(
                            std::make_unique<miopen::fusion::BatchNormFwdTrainingOpInvokeParam>(
                                buffers.at(miopenTensorBatchnormRunningMean),
                                buffers.at(miopenTensorBatchnormRunningVariance),
                                buffers.at(miopenTensorBatchnormSavedMean),
                                buffers.at(miopenTensorBatchnormSavedVariance),
                                buffers.at(miopenTensorBatchnormScale),
                                buffers.at(miopenTensorBatchnormBias),
                                get_scalar(miopenScalarBatchnormExpAvgFactor, double{}),
                                get_scalar(miopenScalarBatchnormEpsilon, double{})));
                        break;
                    case miopenProblemDirectionBackward:
                        operator_args.params.emplace_back(
                            std::make_unique<miopen::fusion::BatchNormBwdTrainingOpInvokeParam>(
                                buffers.at(miopenTensorBatchnormX),
                                buffers.at(miopenTensorBatchnormScale),
                                buffers.at(miopenTensorBatchnormBias),
                                buffers.at(miopenTensorBatchnormScaleDiff),
                                buffers.at(miopenTensorBatchnormBiasDiff),
                                buffers.at(miopenTensorBatchnormSavedMean),
                                buffers.at(miopenTensorBatchnormSavedVariance)));
                        break;
                    case miopenProblemDirectionInference: {
                        operator_args.params.emplace_back(
                            std::make_unique<miopen::fusion::BatchNormInferenceOpInvokeParam>(
                                buffers.at(miopenTensorBatchnormScale),
                                buffers.at(miopenTensorBatchnormBias),
                                buffers.at(miopenTensorBatchnormEstimatedMean),
                                buffers.at(miopenTensorBatchnormEstimatedVariance),
                                get_scalar(miopenScalarBatchnormEpsilon, double{})));
                        break;
                    }
                    default:
                        MIOPEN_THROW(
                            miopenStatusBadParm,
                            "Batchnorm only has forward, backward and inference directions");
                    }
                }),
            problem.operator_descriptor);
    }

    return {operator_args, in_desc, in, out_desc, out, gfx90aaltimpl};
}

FusionPlanDescriptor FusedProblem::AsFusionPlan() const
{
    FusionPlanDescriptor plan;
    plan.input_desc  = GetInput();
    plan.output_desc = GetOutput();
    for(const auto& problem : problems)
        AddProblemToPlan(plan, problem);
    return plan;
}

} // namespace miopen
