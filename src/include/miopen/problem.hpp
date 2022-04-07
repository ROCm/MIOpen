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

#include <miopen/convolution.hpp>
#include <miopen/binary_serialization.hpp>
#include <miopen/object.hpp>
#include <miopen/operator_descriptor.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/tensor.hpp>

#include <cstring>
#include <unordered_map>

namespace miopen {

struct Handle;
struct Solution;
struct SearchOptions;

namespace conv {
struct ProblemDescription;
} // namespace conv

struct Problem : miopenProblem
{
    const TensorDescriptor& GetTensorDescriptor(miopenTensorName_t name) const
    {
        return tensor_descriptors.at(name);
    }

    miopenProblemDirection_t GetDirection() const { return direction; }

    bool RegisterTensorDescriptor(miopenTensorName_t name, TensorDescriptor descriptor)
    {
        return tensor_descriptors.emplace(std::make_pair(name, std::move(descriptor))).second;
    }

    void SetDirection(miopenProblemDirection_t value) { direction = value; }

    void SetOperatorDescriptor(const OperatorDescriptor* descriptor)
    {
        operator_descriptor = std::shared_ptr<OperatorDescriptor>(descriptor->Clone());
    }
    const OperatorDescriptor& GetOperatorDescriptor() const { return *operator_descriptor; }

    std::vector<Solution>
    FindSolutions(Handle& handle, const SearchOptions& options, std::size_t max_solutions) const;

    conv::ProblemDescription AsConvolution() const;

    const TensorDescriptor& GetTensorDescriptorChecked(miopenTensorName_t name,
                                                       const std::string& name_str) const;

    template <class Stream, std::enable_if_t<IsBinarySerializationRelated<Stream>{}, bool> = true>
    friend Stream& operator<<(Stream& stream, Problem& problem)
    {
        stream << problem.direction;
        stream << problem.tensor_descriptors;

        auto primitive = stream.IsDeserializing() ? solver::Primitive{}
                                                  : problem.operator_descriptor->GetPrimitive();

        stream << primitive;

        switch(primitive)
        {
        case solver::Primitive::Convolution:
            if(stream.IsDeserializing())
                problem.operator_descriptor = std::make_shared<ConvolutionDescriptor>();
            stream << *static_cast<ConvolutionDescriptor*>(problem.operator_descriptor.get());
            break;
        case solver::Primitive::Activation:
        case solver::Primitive::Batchnorm:
        case solver::Primitive::Pooling:
        case solver::Primitive::Invalid:
        default: MIOPEN_THROW(miopenStatusNotImplemented);
        }

        return stream;
    }

private:
    miopenProblemDirection_t direction;
    std::unordered_map<miopenTensorName_t, TensorDescriptor> tensor_descriptors;
    std::shared_ptr<OperatorDescriptor> operator_descriptor;

    std::vector<Solution> FindConvSolutions(Handle& handle,
                                            const SearchOptions& options,
                                            std::size_t max_solutions) const;
};

} // namespace miopen

inline std::ostream& operator<<(std::ostream& stream, const miopen::Problem& problem)
{
    // Todo: sane printing
    stream << &problem;
    return stream;
}

MIOPEN_DEFINE_OBJECT(miopenProblem, miopen::Problem);
