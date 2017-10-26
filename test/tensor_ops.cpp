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

    int Aoffset;
    int Boffset;
    int Coffset;

    verify_tensor_ops(const tensor<T>& pa,
                      const tensor<T>& pb,
                      const tensor<T>& pc,
                      int pAoffset,
                      int pBoffset,
                      int pCoffset)
    {
        a = pa;
        b = pb;
        c = pc;

        Aoffset = pAoffset;
        Boffset = pBoffset;
        Coffset = pCoffset;
    }

    // verify_tensor_ops(const tensor<T>& pa, const tensor<T>& pb, const tensor<T>& pc, const
    // std::vector<T>& dims)
    //{
    // a = pa(dims);
    // b = pb(dims);
    // c = pc(dims);
    //}

    T add_elem(T aelem, T belem) { return aelem + belem; }
    T mul_elem(T aelem, T belem) { return aelem * belem; }

    void tensor_for_loop(const tensor<T>& aten,
                         const tensor<T>& bten,
                         tensor<T>& cten,
                         const std::vector<size_t>& a_dims,
                         const std::vector<size_t>& b_dims,
                         float alpha,
                         float beta,
                         int recurr_coffset,
                         int recurr_boffset,
                         int dim,
                         int AtenOffset,
                         int BtenOffset,
                         int CtenOffset)
    {

        int cstride = cten.desc.GetStrides()[dim];
        int bstride = bten.desc.GetStrides()[dim];

        for(int idx = 0; idx < a_dims[dim]; idx++)
        {
            size_t acindex = recurr_coffset + cstride * idx;
            size_t bindex =
                (b_dims[dim] == a_dims[dim]) ? recurr_boffset + bstride * idx : recurr_boffset;

            if((bindex < bten.desc.GetElementSize()) && (dim == a_dims.size() - 1))
            {
                cten[acindex + CtenOffset] =
                    mul_elem(aten[acindex + AtenOffset], bten[bindex + BtenOffset]) * alpha +
                    beta * cten[acindex + Coffset];
            }
            if(dim < (a_dims.size() - 1))
            {

                tensor_for_loop(aten,
                                bten,
                                cten,
                                a_dims,
                                b_dims,
                                alpha,
                                beta,
                                acindex,
                                bindex,
                                dim + 1,
                                AtenOffset,
                                BtenOffset,
                                CtenOffset);
            }
        }
        return;
    }

    tensor<T> cpu()
    {
        std::fill(c.begin(), c.end(), 1);
        auto clens    = c.desc.GetLengths();
        auto blens    = b.desc.GetLengths();
        auto bstrides = b.desc.GetStrides();
        auto cstrides = c.desc.GetStrides();

        float alpha = 1, beta = 1;
        tensor_for_loop(a, b, c, clens, blens, alpha, beta, 0, 0, 0, Aoffset, Boffset, Coffset);

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

        // return c;
        std::fill(c.begin(), c.end(), 1);

        auto c_dev = handle.Write(c.data);
        auto a_dev = handle.Write(a.data);
        auto b_dev = handle.Write(b.data);

        float alpha1 = 1, alpha2 = 1, beta = 1;

        miopen::OpTensor(handle,
                         // miopenTensorOpAdd,
                         miopenTensorOpMul,
                         &alpha1,
                         a.desc,
                         a_dev.get(),
                         &alpha2,
                         b.desc,
                         b_dev.get(),
                         &beta,
                         c.desc,
                         c_dev.get(),
                         Aoffset,
                         Boffset,
                         Coffset);

        c.data = handle.Read<T>(c_dev, c.data.size());

#if(MIO_OPS_DEBUG)
        handle.Finish();
        for(int i = 0; i < a.desc.GetElementSize(); i++)
        {
            std::cout << "A[" << i << "]: " << a[i] << std::endl;
        }
        for(int i = 0; i < b.desc.GetElementSize(); i++)
        {
            std::cout << "B[" << i << "]: " << b[i] << std::endl;
        }
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

    tensor<T> super_a;
    tensor<T> super_b;
    tensor<T> super_c;

    tensor<T> a;
    tensor<T> b;
    tensor<T> c;

    tensor_ops_driver()
    {
        add(super_a, "super_a", generate_tensor(get_super_tensor(), {32, 16, 8}));
        add(super_b, "super_b", generate_tensor(get_super_tensor(), {32, 16, 8}));
        add(super_c, "super_c", generate_tensor(get_super_tensor(), {32, 16, 8}));

        // add(super_a, "super_a", generate_tensor(get_super_tensor(), {8, 4}));
        // add(super_b, "super_b", generate_tensor(get_super_tensor(), {8, 4}));
        // add(super_c, "super_c", generate_tensor(get_super_tensor(), {8, 4}));
    }

    std::set<std::vector<int>> get_super_tensor()
    {
        std::vector<std::vector<int>> a_dims{
            {32, 16, 8},
        };

        return (std::set<std::vector<int>>(a_dims.begin(), a_dims.end()));
    }

    void run()
    {

        a = make_tensor<T, int>(super_a, {16, 8, 4}, {32, 4, 1});
        b = make_tensor<T, int>(super_b, {16, 8, 4}, {32, 4, 1});
        c = make_tensor<T, int>(super_b, {16, 8, 4}, {32, 4, 1});
        // a = make_tensor<T, int>(super_a, {8, 4}, {4, 1});
        // b = make_tensor<T, int>(super_b, {8, 4}, {4, 1});
        // c = make_tensor<T, int>(super_b, {8, 4}, {4, 1});
        if(a.desc.GetSize() == b.desc.GetSize())
            verify(verify_tensor_ops<T>{a, b, c, 4 * 32, 4 * 32, 4 * 32});
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_ops_driver<float>>(argc, argv); }
