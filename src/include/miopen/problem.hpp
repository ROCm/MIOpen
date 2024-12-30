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

#pragma once

#include <miopen/miopen.h>

#include <miopen/activ.hpp>
#include <miopen/allocator.hpp>
#include <miopen/convolution.hpp>
#include <miopen/mha/mha_descriptor.hpp>
#include <miopen/mha/problem_description.hpp>
#include <miopen/softmax.hpp>
#include <miopen/object.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/tensor.hpp>

#include <nlohmann/json_fwd.hpp>

#include <variant>

#include <cstring>
#include <unordered_map>
#include "miopen/fusion/fusion_op_args.hpp"
#include "miopen/fusion/fusion_invoke_params.hpp"
#include "fusion_plan.hpp"

namespace miopen {

struct Handle;
struct Solution;
struct FindOptions;

namespace activ {
struct ProblemDescription;
} // namespace activ

namespace conv {
struct ProblemDescription;
} // namespace conv

namespace softmax {
struct ProblemDescription;
} // namespace softmax

struct BiasDescriptor
{
};

struct BatchnormDescriptor
{
    miopenBatchNormMode_t mode;
    bool runningMeanVariance;

    friend void to_json(nlohmann::json& j, const BatchnormDescriptor& descriptor);
    friend void from_json(const nlohmann::json& j, BatchnormDescriptor& descriptor);
};

// The order of types is important for deserialization and should be preserved between releases.
using OperatorDescriptor = std::variant<ConvolutionDescriptor,
                                        ActivationDescriptor,
                                        BiasDescriptor,
                                        SoftmaxDescriptor,
                                        MhaDescriptor,
                                        BatchnormDescriptor>;

struct Problem
{
    friend struct FusedProblem;

    Problem() = default;

    const TensorDescriptor& GetTensorDescriptor(miopenTensorArgumentId_t name) const
    {
        return tensor_descriptors.at(name);
    }

    miopenProblemDirection_t GetDirection() const { return direction; }

    bool RegisterTensorDescriptor(miopenTensorArgumentId_t name, TensorDescriptor descriptor)
    {
        return tensor_descriptors.emplace(std::make_pair(name, std::move(descriptor))).second;
    }

    void SetDirection(miopenProblemDirection_t value) { direction = value; }

    void SetOperatorDescriptor(OperatorDescriptor descriptor)
    {
        operator_descriptor = std::move(descriptor);
    }

    const OperatorDescriptor& GetOperatorDescriptor() const { return operator_descriptor; }

    std::vector<Solution>
    FindSolutions(Handle& handle, const FindOptions& options, std::size_t max_solutions) const;

    conv::ProblemDescription AsConvolution() const;
    activ::ProblemDescription AsActivation() const;
    mha::ProblemDescription AsMha() const;
    softmax::ProblemDescription AsSoftmax() const;

    [[nodiscard]] miopenTensorArgumentId_t GetInputId() const;
    [[nodiscard]] miopenTensorArgumentId_t GetOutputId() const;

    [[nodiscard]] const TensorDescriptor& GetInput() const
    {
        return tensor_descriptors.at(GetInputId());
    }

    [[nodiscard]] const TensorDescriptor& GetOutput() const
    {
        return tensor_descriptors.at(GetOutputId());
    }

    [[nodiscard]] bool HasInput() const
    {
        return tensor_descriptors.find(GetInputId()) != tensor_descriptors.end();
    }

    [[nodiscard]] bool HasOutput() const
    {
        return tensor_descriptors.find(GetOutputId()) != tensor_descriptors.end();
    }

    void CalculateOutput();

    const TensorDescriptor& GetTensorDescriptorChecked(miopenTensorArgumentId_t name,
                                                       const std::string& name_str) const;

    const TensorDescriptor& GetTensorDescriptor(miopenTensorArgumentId_t name,
                                                const TensorDescriptor& default_value) const;

    Problem MakeTransposed() const;

    void TransposeImpl(const ConvolutionDescriptor& conv_desc);

    AnyInvokeParams MakeConvInvokeParams(const TensorDescriptor& x_desc,
                                         Data_t x,
                                         const TensorDescriptor& w_desc,
                                         Data_t w,
                                         const TensorDescriptor& y_desc,
                                         Data_t y,
                                         Data_t workspace,
                                         size_t workspace_size) const;

    static void ValidateGroupCount(const TensorDescriptor& xDesc,
                                   const TensorDescriptor& wDesc,
                                   const ConvolutionDescriptor& conv);

    void LogDriverCommand() const;

    friend void to_json(nlohmann::json& j, const Problem& problem);
    friend void from_json(const nlohmann::json& j, Problem& problem);

private:
    using Buffers = std::unordered_map<miopenTensorArgumentId_t, Data_t>;

    miopenProblemDirection_t direction = miopenProblemDirectionForward;
    std::unordered_map<miopenTensorArgumentId_t, TensorDescriptor> tensor_descriptors;
    OperatorDescriptor operator_descriptor;

    std::vector<Solution> FindSolutionsImpl(Handle& handle,
                                            const FindOptions& options,
                                            std::size_t max_solutions,
                                            const Buffers& buffers,
                                            const ConvolutionDescriptor& conv_desc,
                                            const Problem& original) const;

    std::vector<Solution> FindSolutionsImpl(Handle& handle,
                                            const FindOptions& options,
                                            std::size_t max_solutions,
                                            const Buffers& buffers,
                                            const MhaDescriptor& mha_desc) const;

    std::vector<Solution> FindSolutionsImpl(Handle& handle,
                                            const FindOptions& options,
                                            std::size_t max_solutions,
                                            const Buffers& buffers,
                                            const SoftmaxDescriptor& softmax_desc) const;

    void LogDriverCommand(const ConvolutionDescriptor& conv_desc) const;
    void LogDriverCommand(const ActivationDescriptor& descriptor) const;
    void LogDriverCommand(const BiasDescriptor& descriptor) const;
    void LogDriverCommand(const MhaDescriptor& descriptor) const;
    void LogDriverCommand(const SoftmaxDescriptor& descriptor) const;
    void LogDriverCommand(const BatchnormDescriptor& descriptor) const;
};

struct MIOPEN_INTERNALS_EXPORT FusedProblem
{
    std::vector<Problem> problems;

    void LogDriverCommand() const
    {
        // Not implemented, but silently
    }

    [[nodiscard]] std::vector<Solution>
    FindSolutions(Handle& handle, const FindOptions& options, std::size_t max_solutions) const;

    void PropagateDescriptors();

    [[nodiscard]] miopenTensorArgumentId_t GetInputId() const
    {
        return problems.front().GetInputId();
    }

    [[nodiscard]] miopenTensorArgumentId_t GetOutputId() const
    {
        return problems.back().GetOutputId();
    }

    [[nodiscard]] const TensorDescriptor& GetInput() const { return problems.front().GetInput(); }
    [[nodiscard]] const TensorDescriptor& GetOutput() const { return problems.back().GetOutput(); }

    [[nodiscard]] FusionPlanDescriptor AsFusionPlan() const;

    friend void to_json(nlohmann::json& j, const FusedProblem& problem);
    friend void from_json(const nlohmann::json& j, FusedProblem& problem);

    [[nodiscard]] fusion::FusionInvokeParams
    MakeInvokeParams(const std::function<Data_t(miopenTensorArgumentId_t, const TensorDescriptor&)>&
                         buffer_getter,
                     OperatorArgs& operator_args) const;

private:
    static void AddProblemToPlan(struct FusionPlanDescriptor& plan, const Problem& problem);
};

struct ProblemContainer : miopenProblem
{
    // The order of types is important for deserialization and should be preserved between releases.
    using Item = std::variant<Problem, FusedProblem>;

    Item item;

    ProblemContainer() = default;
    ProblemContainer(Item item_) // NOLINT(*-explicit-constructor)
        : item(std::move(item_))
    {
    }

    friend void to_json(nlohmann::json& j, const ProblemContainer& problem);
    friend void from_json(const nlohmann::json& j, ProblemContainer& problem);
};

} // namespace miopen

inline std::ostream& operator<<(std::ostream& stream, const miopen::Problem& problem)
{
    // Todo: sane printing
    stream << &problem;
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const miopen::FusedProblem& problem)
{
    // Todo: sane printing
    stream << &problem;
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const miopen::ProblemContainer& problem)
{
    // Todo: sane printing
    stream << &problem;
    return stream;
}

MIOPEN_DEFINE_OBJECT(miopenProblem, miopen::ProblemContainer);
