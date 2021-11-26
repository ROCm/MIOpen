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

#include <miopen/tensor.hpp>
#include <miopen/pooling.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace pooling {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescription
{
    // Forward
    ProblemDescription(const PoolingDescriptor& pooling_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       bool save_index_)
        : direction(Direction::Forward),
          pooling(pooling_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          save_index(save_index_)
    {
    }

    // Backward
    ProblemDescription(const PoolingDescriptor& pooling_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dyDesc_)
        : direction(Direction::Backward),
          pooling(pooling_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          dxDesc(dxDesc_),
          dyDesc(dyDesc_)
    {
    }

    Direction GetDirection() const { return direction; }
    const PoolingDescriptor& GetPooling() const { return pooling; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    const TensorDescriptor& GetDXDesc() const
    {
        assert(direction == Direction::Backward);
        return dxDesc;
    }

    const TensorDescriptor& GetDYDesc() const
    {
        assert(direction == Direction::Backward);
        return dyDesc;
    }

    bool SaveIndex() const
    {
        assert(direction == Direction::Forward);
        return save_index;
    }

    NetworkConfig MakeNetworkConfig() const;

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    private:
    Direction direction;
    PoolingDescriptor pooling;
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    TensorDescriptor dxDesc;
    TensorDescriptor dyDesc;
    bool save_index = false;
};

} // namespace pooling

} // namespace miopen
