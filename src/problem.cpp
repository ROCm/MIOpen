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
#include <miopen/convolution.hpp>
#include <miopen/mha/problem_description.hpp>
#include <miopen/mha/solvers.hpp>
#include <miopen/conv_algo_name.hpp>
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

#include <boost/variant/apply_visitor.hpp>
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
struct VisitType<Visitor, boost::variant<VariantArgs...>>
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
                             miopenTensorArgumentId_t id,
                             const TensorDescriptor& descriptor)
{
    const auto preallocated = options.preallocated_tensors.find(id);

    if(preallocated != options.preallocated_tensors.end())
        return preallocated->second;

    const auto element_size = get_data_size(descriptor.GetType());
    auto buffer             = handle.Create(descriptor.GetElementSpace() * element_size);

    visit_float(descriptor.GetType(), [&](auto as_float) {
        const auto zero = as_float(0.f);
        SetTensor(handle, descriptor, buffer.get(), &zero);
    });

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
    auto buffers       = std::unordered_map<miopenTensorArgumentId_t, Data_t>{};

    const auto allocate = [&](auto id, auto&& descriptor) {
        auto buffer = AllocateTensor(handle, options, owned_buffers, id, descriptor);
        buffers.emplace(id, buffer);
        return buffer;
    };

    for(const auto& pair : tensor_descriptors)
        allocate(pair.first, pair.second);

    auto ret = boost::apply_visitor(
        boost::hof::match(
            [&](const ConvolutionDescriptor& op_desc) {
                return FindSolutionsImpl(handle, options, max_solutions, buffers, op_desc);
            },
            [&](const ActivationDescriptor& /*op_desc*/) -> std::vector<Solution> {
                MIOPEN_THROW(miopenStatusNotImplemented);
            },
            [&](const MHADescriptor& op_desc) {
                return FindSolutionsImpl(handle, options, max_solutions, buffers, op_desc);
            },
            [&](const BiasDescriptor& /*op_desc*/) -> std::vector<Solution> {
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

    std::swap(transposed.tensor_descriptors.at(GetInputId()),
              transposed.tensor_descriptors.at(GetOutputId()));

    return transposed;
}

conv::ProblemDescription Problem::AsConvolution() const
{
    const auto& conv_desc = boost::get<ConvolutionDescriptor>(operator_descriptor);

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
    const auto& activ_desc = boost::get<ActivationDescriptor>(operator_descriptor);

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

mha::ProblemDescription Problem::AsMHA() const
{
    const auto& mha_desc = boost::get<MHADescriptor>(operator_descriptor);

    if(GetDirection() == miopenProblemDirectionBackward)
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "MHA Backward is not currently implemented!");
    }

    mha::MHAInputDescsForward mhaInputDescsForward = {
        GetTensorDescriptorChecked(miopenTensorMHAK, "miopenTensorMHAK"),
        GetTensorDescriptorChecked(miopenTensorMHAQ, "miopenTensorMHAQ"),
        GetTensorDescriptorChecked(miopenTensorMHAV, "miopenTensorMHAV"),
        GetTensorDescriptorChecked(miopenTensorMHADescaleK, "miopenTensorMHADescaleK"),
        GetTensorDescriptorChecked(miopenTensorMHADescaleQ, "miopenTensorMHADescaleQ"),
        GetTensorDescriptorChecked(miopenTensorMHADescaleV, "miopenTensorMHADescaleV"),
        GetTensorDescriptorChecked(miopenTensorMHADescaleS, "miopenTensorMHADescaleS"),
        GetTensorDescriptorChecked(miopenTensorMHAScaleS, "miopenTensorMHAScaleS"),
        GetTensorDescriptorChecked(miopenTensorMHAScaleO, "miopenTensorMHAScaleO"),
        mha_desc.GetDropoutProbability(),
        mha_desc.GetDropoutSeed(),
        mha_desc.GetDropoutOffset(),
        GetTensorDescriptorChecked(miopenTensorMHAO, "miopenTensorMHAO"),
        GetTensorDescriptorChecked(miopenTensorMHAAmaxO, "miopenTensorMHAAmaxO"),
        GetTensorDescriptorChecked(miopenTensorMHAAmaxS, "miopenTensorMHAAmaxS"),
        GetTensorDescriptorChecked(miopenTensorMHAM, "miopenTensorMHAM"),
        GetTensorDescriptorChecked(miopenTensorMHAZInv, "miopenTensorMHAZInv"),
    };

    return mha::ProblemDescription(mhaInputDescsForward);
}

std::vector<Solution> Problem::FindSolutionsImpl(Handle& handle,
                                                 const FindOptions& options,
                                                 std::size_t max_solutions,
                                                 const Buffers& buffers,
                                                 const ConvolutionDescriptor& conv_desc) const
{
    auto ret = std::vector<Solution>{};

    if(tensor_descriptors.size() != 3)
    {
        MIOPEN_THROW(miopenStatusInvalidValue,
                     "Convolution problem should have exactly three tensor descriptors.");
    }

    // These are not swapped for now to preserve argument order in calls
    const auto& x_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    const auto& y_desc =
        GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    const auto& x = buffers.at(miopenTensorConvolutionX);
    const auto& w = buffers.at(miopenTensorConvolutionW);
    const auto& y = buffers.at(miopenTensorConvolutionY);

    const auto conv_problem =
        conv_desc.mode == miopenTranspose ? MakeTransposed().AsConvolution() : AsConvolution();

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

    auto find1_solutions = std::vector<miopenConvAlgoPerf_t>{};
    find1_solutions.resize(max_solutions);
    int found;

    switch(direction)
    {
    case miopenProblemDirectionForward: {
        const auto method = conv_desc.mode == miopenTranspose
                                ? &ConvolutionDescriptor::FindConvBwdDataAlgorithm
                                : &ConvolutionDescriptor::FindConvFwdAlgorithm;

        (conv_desc.*method)(handle,
                            x_desc,
                            x,
                            w_desc,
                            w,
                            y_desc,
                            y,
                            max_solutions,
                            &found,
                            find1_solutions.data(),
                            workspace,
                            workspace_size,
                            options.exhaustive_search);
        break;
    }
    case miopenProblemDirectionBackward: {
        const auto method = conv_desc.mode == miopenTranspose
                                ? &ConvolutionDescriptor::FindConvFwdAlgorithm
                                : &ConvolutionDescriptor::FindConvBwdDataAlgorithm;

        (conv_desc.*method)(handle,
                            y_desc,
                            y,
                            w_desc,
                            w,
                            x_desc,
                            x,
                            max_solutions,
                            &found,
                            find1_solutions.data(),
                            workspace,
                            workspace_size,
                            options.exhaustive_search);
        break;
    }
    case miopenProblemDirectionBackwardWeights: {
        decltype(auto) x_desc_ = conv_desc.mode == miopenTranspose ? y_desc : x_desc;
        decltype(auto) x_      = conv_desc.mode == miopenTranspose ? y : x;
        decltype(auto) y_desc_ = conv_desc.mode == miopenTranspose ? x_desc : y_desc;
        decltype(auto) y_      = conv_desc.mode == miopenTranspose ? x : y;

        conv_desc.FindConvBwdWeightsAlgorithm(handle,
                                              y_desc_,
                                              y_,
                                              x_desc_,
                                              x_,
                                              w_desc,
                                              w,
                                              max_solutions,
                                              &found,
                                              find1_solutions.data(),
                                              workspace,
                                              workspace_size,
                                              options.exhaustive_search);
        break;
    }
    }

    ret.reserve(found);

    const auto conv_dir = ([&]() {
        const auto dir = static_cast<conv::Direction>(direction);
        if(dir == conv::Direction::BackwardWeights || conv_desc.mode != miopenTranspose)
            return dir;
        return dir == conv::Direction::Forward ? conv::Direction::BackwardData
                                               : conv::Direction::Forward;
    })();

    const auto netcfg = conv_problem.MakeNetworkConfig();
    auto conv_ctx     = ExecutionContext{&handle};
    conv_problem.SetupFloats(conv_ctx);

    decltype(auto) db = GetDb(conv_ctx);

    for(auto i = 0; i < found; ++i)
    {
        const auto algo = ConvolutionAlgoToDirectionalString(
            static_cast<miopenConvAlgorithm_t>(find1_solutions[i].fwd_algo), conv_dir);

        auto solution = Solution{};
        solution.SetTime(find1_solutions[i].time);
        solution.SetWorkspaceSize(find1_solutions[i].memory);
        solution.SetSolver(handle.GetFound1_0SolverId(netcfg, AlgorithmName{algo}).value());
        solution.SetPerfConfig(
            solution.GetSolver().GetSolver().GetPerfCfgParams(conv_ctx, conv_problem, db));
        solution.SetProblem({*this});
        MIOPEN_LOG_I("Found solution: " << solution.GetSolver().ToString() << " , "
                                        << solution.GetWorkspaceSize() << ", "
                                        << solution.GetTime());

        ret.emplace_back(std::move(solution));
    }

    return ret;
}

std::vector<Solution> Problem::FindSolutionsImpl(Handle& handle,
                                                 const FindOptions& options,
                                                 std::size_t max_solutions,
                                                 const Buffers& buffers,
                                                 const MHADescriptor& mha_desc) const
{
    auto ret = std::vector<Solution>{};

    auto ctx = ExecutionContext{&handle};

    const mha::ProblemDescription problem_description = AsMHA();

    const auto algo = AlgorithmName{"MHA"};

    static solver::mha::MHA mhaSolver;

    std::vector<solver::mha::MHASolver*> solvers;

    solvers.push_back(&mhaSolver);

    for(auto solver : solvers)
    {
        if(!solver->IsApplicable(ctx, problem_description))
        {
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

void Problem::ValidateGroupCount(const TensorDescriptor& xDesc,
                                 const TensorDescriptor& wDesc,
                                 const ConvolutionDescriptor& conv)
{
    if(conv.group_count == 1)
    {
        if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
    if(conv.group_count > 1)
    {
        if(xDesc.GetLengths()[1] % conv.group_count != 0 ||
           wDesc.GetLengths()[0] % conv.group_count != 0 ||
           conv.group_count > xDesc.GetLengths()[1] || conv.group_count > wDesc.GetLengths()[0])
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if(xDesc.GetLengths()[1] / conv.group_count != wDesc.GetLengths()[1])
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
}

void Problem::LogDriverCommand() const
{
    const auto log_function =
        boost::hof::match([&](const ConvolutionDescriptor& op_desc) { LogDriverCommand(op_desc); },
                          [&](const ActivationDescriptor& op_desc) { LogDriverCommand(op_desc); },
                          [&](const BiasDescriptor&) {},
                          [&](const MHADescriptor&) {});

    boost::apply_visitor(log_function, operator_descriptor);
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

void to_json(nlohmann::json& json, const BiasDescriptor&) { json = nlohmann::json{}; }

void from_json(const nlohmann::json&, BiasDescriptor&) {}

void to_json(nlohmann::json& json, const Problem& problem)
{
    json = nlohmann::json{
        {"direction", problem.direction},
        {"tensors", problem.tensor_descriptors},
        {"primitive", problem.operator_descriptor.which()},
    };

    auto operator_serialization = [&](auto&& op) { json["operator"] = op; };
    boost::apply_visitor(operator_serialization, problem.operator_descriptor);
}

namespace detail {
template <class Descriptor>
struct OperatorDescriptorDeserializer
{
    const nlohmann::json* json;
    OperatorDescriptor* descriptor;

    void operator()() const { *descriptor = json->get<Descriptor>(); }
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
        {"problem_type", problem.item.which()},
    };

    auto operator_serialization = [&](auto&& op) { json["value"] = op; };
    boost::apply_visitor(operator_serialization, problem.item);
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

    boost::apply_visitor(
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
            [&](const MHADescriptor&) { RegisterTensorDescriptor(GetOutputId(), GetInput()); },
            [&](const BiasDescriptor&) { RegisterTensorDescriptor(GetOutputId(), GetInput()); }),
        operator_descriptor);
}

miopenTensorArgumentId_t Problem::GetInputId() const
{
    return boost::apply_visitor(
        boost::hof::match([](const ConvolutionDescriptor&) { return miopenTensorConvolutionX; },
                          [](const ActivationDescriptor&) { return miopenTensorActivationX; },
                          [](const BiasDescriptor&) { return miopenTensorBiasX; },
                          [](const MHADescriptor&) { return miopenTensorMHAK; }),
        operator_descriptor);
}

miopenTensorArgumentId_t Problem::GetOutputId() const
{
    return boost::apply_visitor(
        boost::hof::match([](const ConvolutionDescriptor&) { return miopenTensorConvolutionY; },
                          [](const ActivationDescriptor&) { return miopenTensorActivationY; },
                          [](const BiasDescriptor&) { return miopenTensorBiasY; },
                          [](const MHADescriptor&) { return miopenTensorMHAO; }),
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
    const auto find1_solutions = [&]() {
        OperatorArgs params;
        auto owned_buffers = std::vector<Allocator::ManageDataPtr>{};

        const auto make_invoke_params = [&]() {
            auto buffer_allocator = [&](auto id, auto&& desc) {
                return AllocateTensor(handle, options, owned_buffers, id, desc);
            };

            return MakeInvokeParams(buffer_allocator, params);
        };

        return AsFusionPlan().Find(handle, make_invoke_params, options);
    }();

    auto ret = std::vector<Solution>{};
    ret.reserve(find1_solutions.size());
    // decltype(auto) db = GetDb(ExecutionContext{&handle});

    for(const auto& find1_solution : find1_solutions)
    {
        auto solution = Solution{};
        solution.SetTime(find1_solution.time);
        solution.SetWorkspaceSize(find1_solution.workspace);
        solution.SetSolver(find1_solution.solver_id);
        solution.SetProblem({*this});
        // solution.SetPerfConfig(solution.GetSolver().GetSolver().GetPerfCfgParams(conv_ctx,
        // legacy_problem, db));
        MIOPEN_LOG_I("Found solution: " << solution.GetSolver().ToString() << " , "
                                        << solution.GetWorkspaceSize() << ", "
                                        << solution.GetTime());

        ret.emplace_back(std::move(solution));
    }

    SortFindResults(options, ret);
    ret.resize(std::min(ret.size(), max_solutions));
    return ret;
}

void FusedProblem::AddProblemToPlan(FusionPlanDescriptor& plan, const Problem& problem)
{
    boost::apply_visitor(
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
            [&](const MHADescriptor&) {
                // Not implemented
                assert(false);
                MIOPEN_THROW(miopenStatusNotImplemented, "MHA is not implemented for FusedProblem");
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
        auto buffer = buffer_getter(id, descriptor);
        buffers.emplace(id, buffer);
        return buffer;
    };

    bool gfx90aaltimpl = false;
    auto in            = get_buffer(GetInputId(), in_desc);
    auto out           = get_buffer(GetOutputId(), out_desc);

    for(const auto& problem : problems)
    {
        for(const auto& pair : problem.tensor_descriptors)
            if(pair.first != problem.GetInputId() && pair.first != problem.GetOutputId())
                get_buffer(pair.first, pair.second);

        boost::apply_visitor(
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
                [&](const MHADescriptor&) {
                    // Not implemented
                    assert(false);
                    MIOPEN_THROW(miopenStatusNotImplemented,
                                 "MHA is not implemented for FusedProblem");
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
