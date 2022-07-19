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

#include <miopen/quanti.hpp>
#include <miopen/tensor.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace quanti {

struct ProblemDescription
{
    // constructor
    ProblemDescription(const QuantizationDescriptor& quanti,
                       const TensorDescriptor& in,
                       const TensorDescriptor& out)
        : quantiDesc(quanti), in(in_), out(out_)
    {
    }

    const QuantizationDescriptor& GetQuantiDesc() const { return quantiDesc; }
    const TensorDescriptor& GetInDesc() const { return inDesc; }
    const TensorDescriptor& GetOutDesc() const { return outDesc; }

    NetworkConfig MakeNetworkConfig() const;

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

private:
    QuantizationDescriptor quantiDesc;
    TensorDescriptor in;
    TensorDescriptor out;

} // namespace quanti

} // namespace miopen
