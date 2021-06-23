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

#pragma once

#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace activ {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescription
{
    ProblemDescription(Direction direction_,
                       const ActivationDescriptor& activ,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_)
        : direction(direction_), activDesc(activ), xDesc(xDesc_), yDesc(yDesc_)
    {
    }

    Direction GetDirection() const { return direction; }
    const ActivationDescriptor& GetActivDesc() const { return activDesc; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    NetworkConfig MakeNetworkConfig() const;

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    private:
    Direction direction;
    ActivationDescriptor activDesc;
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
};

} // namespace activ

} // namespace miopen
