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

#include <miopen/object.hpp>
#include <miopen/solver.hpp>
#include <miopen/tensor.hpp>

#include <unordered_map>

namespace miopen {

struct OperatorDescriptor
{
    virtual ~OperatorDescriptor()                  = default;
    virtual solver::Primitive GetPrimitive() const = 0;
    virtual OperatorDescriptor* Clone() const      = 0;
};

template <class Derived, solver::Primitive primitive>
struct OperatorDescriptorImpl : OperatorDescriptor
{
    solver::Primitive GetPrimitive() const final { return primitive; }

    OperatorDescriptor* Clone() const final
    {
        return new Derived{reinterpret_cast<const Derived&>(*this)};
    }
};

struct Problem : miopenProblem
{
    const TensorDescriptor& GetTensorDescriptor(miopenProblemTensorName_t name) const
    {
        return tensor_descriptors.at(name);
    }

    miopenProblemDirection_t GetDirection() const { return direction; }

    bool RegisterTensorDescriptor(miopenProblemTensorName_t name, TensorDescriptor descriptor)
    {
        return tensor_descriptors.emplace(std::make_pair(name, std::move(descriptor))).second;
    }

    void SetDirection(miopenProblemDirection_t value) { direction = value; }
    void SetOperatorDescriptor(const OperatorDescriptor* descriptor)
    {
        operator_descriptor = std::shared_ptr<OperatorDescriptor>(descriptor->Clone());
    }

private:
    miopenProblemDirection_t direction;
    std::unordered_map<miopenProblemTensorName_t, TensorDescriptor> tensor_descriptors;
    std::shared_ptr<OperatorDescriptor> operator_descriptor;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenProblem, miopen::Problem);
