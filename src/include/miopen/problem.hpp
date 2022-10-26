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

#include <miopen/allocator.hpp>
#include <miopen/convolution.hpp>
#include <miopen/object.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/tensor.hpp>

#include <nlohmann/json_fwd.hpp>

#include <boost/variant.hpp>

#include <cstring>
#include <unordered_map>

namespace miopen {

struct Handle;
struct Solution;
struct FindOptions;

namespace conv {
struct ProblemDescription;
} // namespace conv

using OperatorDescriptor = boost::variant<ConvolutionDescriptor>;

struct Problem : miopenProblem
{
    using Buffers = std::unordered_map<miopenTensorArgumentId_t, Data_t>;

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

    std::vector<Solution> FindSolutions(Handle& handle,
                                        const FindOptions& options,
                                        const Buffers& buffers,
                                        const Data_t workspace,
                                        const std::size_t workspace_size,
                                        std::size_t max_solutions) const;

    conv::ProblemDescription AsConvolution() const;

    const TensorDescriptor& GetTensorDescriptorChecked(miopenTensorArgumentId_t name,
                                                       const std::string& name_str) const;

    Problem MakeTransposed() const;

    static void ValidateGroupCount(const TensorDescriptor& xDesc,
                                   const TensorDescriptor& wDesc,
                                   const ConvolutionDescriptor& conv);

    friend void to_json(nlohmann::json& j, const Problem& problem);
    friend void from_json(const nlohmann::json& j, Problem& problem);

private:
    miopenProblemDirection_t direction = miopenProblemDirectionForward;
    std::unordered_map<miopenTensorArgumentId_t, TensorDescriptor> tensor_descriptors;
    OperatorDescriptor operator_descriptor;

    std::vector<Solution> FindSolutionsImpl(Handle& handle,
                                            const FindOptions& options,
                                            std::size_t max_solutions,
                                            const Buffers& buffers,
                                            const Data_t workspace,
                                            const std::size_t workspace_size,
                                            const ConvolutionDescriptor& conv_desc) const;

    void TransposeImpl(const ConvolutionDescriptor& conv_desc);
};

} // namespace miopen

inline std::ostream& operator<<(std::ostream& stream, const miopen::Problem& problem)
{
    // Todo: sane printing
    stream << &problem;
    return stream;
}

MIOPEN_DEFINE_OBJECT(miopenProblem, miopen::Problem);
