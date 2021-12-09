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
#include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <utility>
#include <cstdlib>
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "tensor_util.hpp"

template <typename T>
struct set_data_t
{
    const T alpha;

    void operator()(T& r_data) const { r_data = alpha; }
};

template <class T>
struct verify_tensor_set
{
    miopen::TensorDescriptor subDesc;
    tensor<T> super;
    int offset;
    T alpha;

    verify_tensor_set(const tensor<T>& rSuper,
                      const miopen::TensorDescriptor& rSubDesc,
                      const int offsetIn,
                      const T alphaIn)
    {
        subDesc = rSubDesc;
        super   = rSuper;
        offset  = offsetIn;
        alpha   = alphaIn;
    }

    tensor<T> cpu() const
    {
        tensor<T> superCpu = super;

        const set_data_t<T> data_operator = {alpha};

        operate_over_subtensor(data_operator, superCpu, subDesc, offset);

        return superCpu;
    }

    tensor<T> gpu() const
    {
        tensor<T> superGpu = super;

        auto&& handle  = get_handle();
        auto super_dev = handle.Write(superGpu.data);

        miopen::SetTensor(handle, subDesc, super_dev.get(), &alpha, offset);

        superGpu.data = handle.Read<T>(super_dev, superGpu.data.size());

        return superGpu;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Set: " << std::endl;
        std::cout << "super-tensor: " << super.desc.ToString() << std::endl;
        std::cout << "sub-tensor: " << subDesc.ToString() << std::endl;
    }
};

template <class T>
struct tensor_set_driver : test_driver
{
    tensor<T> super;
    std::vector<int> superLens;
    miopen::TensorDescriptor subDesc;
    std::vector<int> subLens;
    int offset = 0;

    tensor_set_driver()
    {
        disabled_cache        = true;
        std::vector<int> lens = {32, 32, 16, 16, 16};

        add(superLens, "superLens", generate_data({lens}, lens));
        add(subLens, "subLens", generate_data(get_sub_tensor(), {32, 8, 10}));
        add(offset, "offset", generate_data(get_tensor_offset(), 7));
    }

    void run()
    {
        unsigned long max_value =
            miopen_type<T>{} == miopenHalf ? 5 : miopen_type<T>{} == miopenInt8 ? 127 : 17;

        super = tensor<T>{superLens}.generate(tensor_elem_gen_integer{max_value});

        std::vector<size_t> superStrides = super.desc.GetStrides();
        std::vector<int> subStrides(superStrides.begin() + (super.desc.GetSize() - subLens.size()),
                                    superStrides.end());

        subDesc =
            miopen::TensorDescriptor(this->type, subLens.data(), subStrides.data(), subLens.size());

        verify_equals(verify_tensor_set<T>{super, subDesc, offset, T(1.111)});
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_set_driver>(argc, argv); }
