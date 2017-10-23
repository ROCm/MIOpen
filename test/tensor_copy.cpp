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

#define MIO_OPS_DEBUG 1

template <class T>
struct tensor_copy_base
{
    tensor<T> a;
    tensor<T> c;
    int aoffset;
    int coffset;

    void fail(float = 0)
    {
        std::cout << "A tensor: " << a.desc.ToString() << std::endl;
        std::cout << "C tensor: " << c.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_tensor_copy : tensor_copy_base<T>
{
    using tensor_copy_base<T>::a;
    using tensor_copy_base<T>::c;
    using tensor_copy_base<T>::aoffset;
    using tensor_copy_base<T>::coffset;
    

    verify_tensor_copy(const tensor<T>& pa, const tensor<T>& pc)
    {
        a = pa;
        c = pc;
    }

    verify_tensor_copy(const tensor<T>& pa, const tensor<T>& pc, const std::vector<T>& dims)
    {
        a = pa(dims);
        c = pc(dims);
    }

    void tensor_for_loop(const tensor<T>& aten,
                         tensor<T>& cten,
                         const std::vector<size_t>& a_dims,
                         const std::vector<size_t>& c_dims,
                         int aoffsetIndex,
                         int coffsetIndex,
                         int dim)
    {

        int astride = aten.desc.GetStrides()[dim];
        int cstride = cten.desc.GetStrides()[dim];

        for(int idx = 0; idx < a_dims[dim]; idx++)
        {
            size_t aindex = aoffsetIndex + astride * idx;
            size_t cindex = coffsetIndex + cstride * idx;
            
            if(cindex < cten.desc.GetElementSize() && aindex < aten.desc.GetElementSize())
                cten[aindex] = aten[aindex];
            
            if(dim < (a_dims.size() - 1))
            {
                tensor_for_loop(aten, cten, a_dims, c_dims, aindex, cindex, dim + 1);
            }
        }
        return;
    }

    tensor<T> cpu()
    {
        
        c = a;
        std::fill(c.begin(), c.end(), 0);
        for(int i = 0; i < a.desc.GetElementSize(); i++)
        {
            a[i] = i;
        }
        
        auto alens    = a.desc.GetLengths();
        auto clens    = c.desc.GetLengths();
        
        auto astrides = a.desc.GetStrides();
        auto cstrides = c.desc.GetStrides();

        tensor_for_loop(a, c, alens, clens, 0, 0, 0);

#if(MIO_OPS_DEBUG)
//        for(int i = 0; i < c.desc.GetElementSize(); i++)
//        {
//            std::cout << "C_CPU[" << i << "]: " << c[i] << std::endl;
//        }
#endif
        return c;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();

        c = a;
        // return c;
        std::fill(c.begin(), c.end(), 0);
        for(int i = 0; i < a.desc.GetElementSize(); i++)
        {
            a[i] = i;
        }

        auto c_dev = handle.Write(c.data);
        auto a_dev = handle.Write(a.data);
        
        
        miopen::CopyTensor(handle,
                        a.desc,
                        a_dev.get(),
                        c.desc,
                        c_dev.get(),
                        aoffset,
                        coffset);

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
        std::cout << "Tensor Copy: " << std::endl;
        this->tensor_copy_base<T>::fail();
    }
};

template <class T>
struct tensor_copy_driver : test_driver
{
    tensor<T> a;
    tensor<T> c;

    tensor_copy_driver()
    {
        add(a, "a", generate_tensor(get_tensor_a(), {4, 3, 2, 2, 3}));
        add(c, "c", generate_tensor(get_tensor_c(), {4, 3, 2, 2, 3}));
//        add(a, "a", generate_tensor(get_tensor_a(), {11, 7, 13, 13}));
//        add(c, "c", generate_tensor(get_tensor_c(), {11, 7, 13, 13}));
    }
    
    

    std::set<std::vector<int>> get_tensor_a()
    {
        std::vector<std::vector<int>> a_dims{
            {32, 8, 16, 16, 8}, {32, 8, 16, 16}, {32, 8, 16}, {32, 8}, {8},
        };
        return (std::set<std::vector<int>>(a_dims.begin(), a_dims.end()));
    }

    std::set<std::vector<int>> get_tensor_c()
    {
        std::vector<std::vector<int>> c_dims{     //TODO: enumerate this list with descriptors
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
        return (std::set<std::vector<int>>(c_dims.begin(), c_dims.end()));
    }

    // void run() { verify(verify_tensor_ops<T, 2>{a, b}); }
    // void run() { verify(verify_tensor_ops<T, 4>{a, b}); }
    void run()
    {
        if(a.desc.GetSize() == c.desc.GetSize())
            verify(verify_tensor_copy<T>{a, c});
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_copy_driver<float>>(argc, argv); }
