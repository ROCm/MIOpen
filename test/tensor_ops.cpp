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

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#define MIO_OPS_DEBUG 0

template <class T>
struct tensor_ops_base
{
    tensor<T> a;
    tensor<T> b;
    tensor<T> c;

    void fail(float = 0)
    {
        std::cout << "A tensor: " << a.desc.ToString() << std::endl;
        std::cout << "B tensor: " << b.desc.ToString() << std::endl;
        std::cout << "C tensor: " << a.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_tensor_ops : tensor_ops_base<T>
{
    using tensor_ops_base<T>::a;
    using tensor_ops_base<T>::b;
    using tensor_ops_base<T>::c;

    verify_tensor_ops(const tensor<T>& pa, const tensor<T>& pb)
    {
        a = pa;
        b = pb;
    }

    verify_tensor_ops(const tensor<T>& pa, const tensor<T>& pb, const std::vector<T>& dims)
    {
        a = pa(dims);
        b = pb(dims);
    }

    T add_elem(T aelem, T belem) { return aelem + belem; }

    void tensor_for_loop(const tensor<T>& aten,
                         const tensor<T>& bten,
                         tensor<T>& cten,
                         const std::vector<size_t>& a_dims,
                         const std::vector<size_t>& b_dims,
                         int coffset,
                         int boffset,
                         int dim)
    {

        int cstride = cten.desc.GetStrides()[dim];
        int bstride = bten.desc.GetStrides()[dim];

        for(int idx = 0; idx < a_dims[dim]; idx++)
        {
            size_t acindex = coffset + cstride * idx;
            size_t bindex  = (b_dims[dim] == a_dims[dim]) ? boffset + bstride * idx : boffset;

            if(bindex < bten.desc.GetElementSize())
                cten[acindex] = add_elem(aten[acindex], bten[bindex]);
            if(dim < (a_dims.size() - 1))
            {

                tensor_for_loop(aten, bten, cten, a_dims, b_dims, acindex, bindex, dim + 1);
            }
        }
        return;
    }

    tensor<T> cpu()
    {
        c = a;
        std::fill(c.begin(), c.end(), 0);
        std::fill(c.begin(), c.end(), 0);
        auto clens    = c.desc.GetLengths();
        auto blens    = b.desc.GetLengths();
        auto bstrides = b.desc.GetStrides();
        auto cstrides = c.desc.GetStrides();

        tensor_for_loop(a, b, c, clens, blens, 0, 0, 0);

#if(MIO_OPS_DEBUG)
        for(int i = 0; i < c.desc.GetElementSize(); i++)
        {
            std::cout << "C_CPU[" << i << "]: " << c[i] << std::endl;
        }
#endif
        return c;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();

        c = a;
        // return c;
        std::fill(c.begin(), c.end(), 0);

        auto c_dev = handle.Write(c.data);
        auto a_dev = handle.Write(a.data);
        auto b_dev = handle.Write(b.data);

        int alpha1 = 1, alpha2 = 1, beta = 0;

        miopen::OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha1,
                         a.desc,
                         a_dev.get(),
                         &alpha2,
                         b.desc,
                         b_dev.get(),
                         &beta,
                         c.desc,
                         c_dev.get());

        c.data = handle.Read<T>(c_dev, c.data.size());

#if(MIO_OPS_DEBUG)
        handle.Finish();
        for(int i = 0; i < c.desc.GetElementSize(); i++)
        {
            std::cout << "C_GPU[" << i << "]: " << c[i] << std::endl;
        }
#endif
        return c;
    }

    void fail(float = 0)
    {
        std::cout << "TensorOp: " << std::endl;
        this->tensor_ops_base<T>::fail();
    }
};

template <class T>
struct tensor_ops_driver : test_driver
{
    tensor<T> a;
    tensor<T> b;

    tensor_ops_driver()
    {
        add(a, "a", generate_tensor(get_tensor_a(), {11, 7, 13, 13}));
        add(b, "b", generate_tensor(get_tensor_b(), {1, 7, 1, 1}));
    }

    std::set<std::vector<int>> get_tensor_a()
    {
        std::vector<std::vector<int>> a_dims{
            {32, 8, 16, 16, 8}, {32, 8, 16, 16}, {32, 8, 16}, {32, 8}, {8},
        };
        return (std::set<std::vector<int>>(a_dims.begin(), a_dims.end()));
    }

    std::set<std::vector<int>> get_tensor_b()
    {
        std::vector<std::vector<int>> b_dims{
            {1, 8, 1, 1, 8},
            {1, 1, 1, 16, 8},
            {1, 1, 16, 1, 1},
            {1, 1, 16, 16, 8},
            {1, 8, 1, 16, 1},
            {1, 8, 16, 1, 8},
            {1, 8, 16, 16, 1},
            {32, 8, 1, 1, 8},
            {32, 8, 1, 16, 1},
            {32, 8, 16, 1, 8},
            {32, 8, 16, 16, 1},
            {32, 8, 16, 16, 8},
            {1, 8, 1, 1},
            {1, 1, 1, 16},
            {1, 1, 16, 1},
            {1, 1, 16, 16},
            {1, 8, 1, 16},
            {1, 8, 16, 1},
            {1, 8, 16, 16},
            {32, 8, 1, 1},
            {32, 8, 1, 16},
            {32, 8, 16, 1},
            {32, 8, 16, 16},
            {1, 8, 1},
            {1, 1, 16},
            {32, 1, 1},
            {1, 8, 16},
            {32, 8, 1},
            {32, 1, 16},
            {32, 8, 16},
            {1, 8},
            {32, 1},
            {32, 8},
            {8},
        };
        return (std::set<std::vector<int>>(b_dims.begin(), b_dims.end()));
    }

    // void run() { verify(verify_tensor_ops<T, 2>{a, b}); }
    // void run() { verify(verify_tensor_ops<T, 4>{a, b}); }
    void run()
    {
        if(a.desc.GetSize() == b.desc.GetSize())
            verify(verify_tensor_ops<T>{a, b});
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_ops_driver<float>>(argc, argv); }
