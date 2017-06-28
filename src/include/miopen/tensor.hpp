/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TENSOR_HPP_
#define GUARD_MIOPEN_TENSOR_HPP_

#include <cassert>
#include <iostream>
#include <miopen/common.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <vector>
// TODO(paul): remove this include later
#include <cstdio>

namespace miopen {

template <class T>
auto tie4(T&& x) -> decltype(std::tie(x[0], x[1], x[2], x[3]))
{
    assert(x.size() == 4);
    return std::tie(x[0], x[1], x[2], x[3]);
}

template <class T>
auto tie2(T&& x) -> decltype(std::tie(x[0], x[1]))
{
    assert(x.size() == 2);
    return std::tie(x[0], x[1]);
}

struct TensorDescriptor : miopenTensorDescriptor
{
    TensorDescriptor();
    TensorDescriptor(miopenDataType_t t, std::initializer_list<int> plens);
    TensorDescriptor(miopenDataType_t t,
                     std::initializer_list<int> plens,
                     std::initializer_list<int> pstrides);
    TensorDescriptor(miopenDataType_t t, const int* plens, int size);
    TensorDescriptor(miopenDataType_t t, const int* plens, const int* pstrides, int size);

    void CalculateStrides();

    const std::vector<int>& GetLengths() const;
    const std::vector<int>& GetStrides() const;
    int GetSize() const;

    miopenDataType_t GetType() const;

    int GetElementSize() const;

    int GetIndex(std::initializer_list<int> l) const;

    template <class... Ts>
    int GetIndex(Ts... is) const
    {
        return this->GetIndex({is...});
    }

    bool operator==(const TensorDescriptor& rhs) const;
    bool operator!=(const TensorDescriptor& rhs) const;

    std::string ToString() const;

    friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    private:
    std::vector<int> lens;
    std::vector<int> strides;

    miopenDataType_t type;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenTensorDescriptor, miopen::TensorDescriptor)

#endif // GUARD_MIOPEN_TENSOR_HPP_
