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

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solution.hpp>
#include <miopen/search_options.hpp>
#include <miopen/tensor_ops.hpp>

#include <nlohmann/json.hpp>

#include <boost/variant/apply_visitor.hpp>
#include <boost/hof/match.hpp>

namespace miopen::debug {
// Todo: This should be updated when a separate driver command is implemented
void LogCmdFindConvolution(const miopen::TensorDescriptor& x,
                           const miopen::TensorDescriptor& w,
                           const miopen::ConvolutionDescriptor& conv,
                           const miopen::TensorDescriptor& y,
                           miopenProblemDirection_t dir,
                           std::optional<uint64_t> solver_id);
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

std::vector<Solution>
Problem::FindSolutions(Handle& handle, const FindOptions& options, std::size_t max_solutions) const
{
    auto owned_buffers = std::vector<Allocator::ManageDataPtr>{};
    auto buffers       = std::unordered_map<miopenTensorArgumentId_t, Data_t>{};

    for(const auto& pair : tensor_descriptors)
    {
        const auto preallocated = options.preallocated_tensors.find(pair.first);

        if(preallocated != options.preallocated_tensors.end())
        {
            buffers.emplace(pair.first, preallocated->second);
            continue;
        }

        const auto& descriptor  = pair.second;
        const auto element_size = get_data_size(descriptor.GetType());
        auto buffer             = handle.Create(descriptor.GetElementSpace() * element_size);

        visit_float(descriptor.GetType(), [&](auto as_float) {
            const auto zero = as_float(0.f);
            SetTensor(handle, descriptor, buffer.get(), &zero);
        });

        buffers.emplace(pair.first, buffer.get());
        owned_buffers.emplace_back(std::move(buffer));
    }

    const auto find = boost::hof::match(
        [&](const ConvolutionDescriptor& op_desc) {
            return FindSolutionsImpl(handle, options, max_solutions, buffers, op_desc);
        },
        [&](const ActivationDescriptor& /*op_desc*/) -> std::vector<Solution> {
            MIOPEN_THROW(miopenStatusNotImplemented);
        });

    auto ret = boost::apply_visitor(find, operator_descriptor);

    const auto sorter = [&]() -> std::function<bool(const Solution&, const Solution&)> {
        switch(options.results_order)
        {
        case miopenFindResultsOrderByTime:
            return [](auto&& l, auto&& r) { return l.GetTime() < r.GetTime(); };
        case miopenFindResultsOrderByWorkspaceSize:
            return [](auto&& l, auto&& r) { return l.GetWorkspaceSize() < r.GetWorkspaceSize(); };
        }
        MIOPEN_THROW(miopenStatusNotImplemented);
    }();

    std::sort(ret.begin(), ret.end(), sorter);

    return ret;
}

const TensorDescriptor&
Problem::GetTensorDescriptorChecked(miopenTensorArgumentId_t name,
                                    [[maybe_unused]] const std::string& name_str) const
{
    const auto found = tensor_descriptors.find(name);
    if(found == tensor_descriptors.end())
        MIOPEN_THROW(miopenStatusInvalidValue,
                     "Problem is missing " + name_str + " tensor descriptor.");
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
        [&](const ActivationDescriptor& /*op_desc*/) { MIOPEN_THROW(miopenStatusNotImplemented); });

    boost::apply_visitor(transpose_tensors, operator_descriptor);

    return transposed;
}

void Problem::TransposeImpl(const ConvolutionDescriptor& /*conv_desc*/)
{
    std::swap(tensor_descriptors.at(miopenTensorConvolutionX),
              tensor_descriptors.at(miopenTensorConvolutionY));
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

std::vector<Solution> Problem::FindSolutionsImpl(Handle& handle,
                                                 const FindOptions& options,
                                                 std::size_t max_solutions,
                                                 const AllocatedBuffers& buffers,
                                                 const ConvolutionDescriptor& conv_desc) const
{
    auto ret = std::vector<Solution>{};

    if(tensor_descriptors.size() != 3)
        MIOPEN_THROW(miopenStatusInvalidValue,
                     "Convolution problem should have exactly three tensor descriptors.");

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

    const auto legacy_problem = ProblemDescription{conv_problem};
    const auto netcfg         = conv_problem.BuildConfKey();
    auto conv_ctx             = ExecutionContext{&handle};
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
            solution.GetSolver().GetSolver().GetPerfCfgParams(conv_ctx, legacy_problem, db));
        solution.SetProblem(*this);
        MIOPEN_LOG_I("Found solution: " << solution.GetSolver().ToString() << " , "
                                        << solution.GetWorkspaceSize() << ", "
                                        << solution.GetTime());

        ret.emplace_back(std::move(solution));
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
    const auto log_function = boost::hof::match(
        [&](const ConvolutionDescriptor& op_desc) { return LogDriverCommand(op_desc); },
        [&](const ActivationDescriptor& /*op_desc*/) { MIOPEN_THROW(miopenStatusNotImplemented); });

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

} // namespace miopen
