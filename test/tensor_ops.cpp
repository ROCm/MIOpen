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

    tensor<T> cpu()
    {
        c = a;
        std::fill(c.begin(), c.end(), 0);
        const std::vector<size_t>& a_dims = a.desc.GetLengths();
        std::fill(c.begin(), c.end(), 0);
        int c_n, c_c, c_d, c_h, c_w;
        int b_n, b_c, b_d, b_h, b_w;

        auto dims = a_dims.size();
        switch(dims)
        {
        case 5:
            std::tie(c_n, c_c, c_d, c_h, c_w) = miopen::tien<5>(c.desc.GetLengths());
            std::tie(b_n, b_c, b_d, b_h, b_w) = miopen::tien<5>(b.desc.GetLengths());
            for(int n = 0; n < c_n; n++)
            {
                c(n, 0, 0, 0, 0) = (b_n == c_n) ? a(n, 0, 0, 0, 0) + b(n, 0, 0, 0, 0)
                                                : a(n, 0, 0, 0, 0) + b(0, 0, 0, 0, 0);
                for(int x = 0; x < c_c; x++)
                {
                    c(n, x, 0, 0, 0) = (b_c == c_c)
                                           ? a(n, x, 0, 0, 0) + b((b_n == c_n ? n : 0), x, 0, 0, 0)
                                           : a(n, x, 0, 0, 0) + b((b_n == c_n ? n : 0), 0, 0, 0, 0);

                    for(int d = 0; d < c_d; d++)
                    {
                        c(n, x, d, 0, 0) =
                            (b_d == c_d)
                                ? a(n, x, d, 0, 0) +
                                      b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), d, 0, 0)
                                : a(n, x, d, 0, 0) +
                                      b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), 0, 0, 0);
                        for(int h = 0; h < c_h; h++)
                        {
                            c(n, x, d, h, 0) = (b_h == c_h)
                                                   ? a(n, x, d, h, 0) + b((b_n == c_n ? n : 0),
                                                                          (b_c == c_c ? x : 0),
                                                                          (b_d == c_d ? d : 0),
                                                                          h,
                                                                          0)
                                                   : a(n, x, d, h, 0) + b((b_n == c_n ? n : 0),
                                                                          (b_c == c_c ? x : 0),
                                                                          (b_d == c_d ? d : 0),
                                                                          0,
                                                                          0);

                            for(int w = 0; w < c_w; w++)
                            {
                                c(n, x, d, h, w) = (b_w == c_w)
                                                       ? a(n, x, d, h, w) + b((b_n == c_n ? n : 0),
                                                                              (b_c == c_c ? x : 0),
                                                                              (b_d == c_d ? d : 0),
                                                                              (b_h == c_h ? h : 0),
                                                                              w)
                                                       : a(n, x, d, h, w) + b((b_n == c_n ? n : 0),
                                                                              (b_c == c_c ? x : 0),
                                                                              (b_d == c_d ? d : 0),
                                                                              (b_h == c_h ? h : 0),
                                                                              0);
                            }
                        }
                    }
                }
            }
            break;

        case 4:

            std::tie(c_n, c_c, c_h, c_w) = miopen::tien<4>(c.desc.GetLengths());
            std::tie(b_n, b_c, b_h, b_w) = miopen::tien<4>(b.desc.GetLengths());
            for(int n = 0; n < c_n; n++)
            {
                c(n, 0, 0, 0) =
                    (b_n == c_n) ? a(n, 0, 0, 0) + b(n, 0, 0, 0) : a(n, 0, 0, 0) + b(0, 0, 0, 0);
                for(int x = 0; x < c_c; x++)
                {
                    c(n, x, 0, 0) = (b_c == c_c) ? a(n, x, 0, 0) + b((b_n == c_n ? n : 0), x, 0, 0)
                                                 : a(n, x, 0, 0) + b((b_n == c_n ? n : 0), 0, 0, 0);

                    for(int h = 0; h < c_h; h++)
                    {
                        c(n, x, h, 0) =
                            (b_h == c_h)
                                ? a(n, x, h, 0) +
                                      b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), h, 0)
                                : a(n, x, h, 0) +
                                      b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), 0, 0);

                        for(int w = 0; w < c_w; w++)
                        {
                            c(n, x, h, w) = (b_w == c_w)
                                                ? a(n, x, h, w) + b((b_n == c_n ? n : 0),
                                                                    (b_c == c_c ? x : 0),
                                                                    (b_h == c_h ? h : 0),
                                                                    w)
                                                : a(n, x, h, w) + b((b_n == c_n ? n : 0),
                                                                    (b_c == c_c ? x : 0),
                                                                    (b_h == c_h ? h : 0),
                                                                    0);
                        }
                    }
                }
            }
            break;

        case 3:

            std::tie(c_n, c_c, c_h) = miopen::tien<3>(c.desc.GetLengths());
            std::tie(b_n, b_c, b_h) = miopen::tien<3>(b.desc.GetLengths());
            for(int n = 0; n < c_n; n++)
            {
                c(n, 0, 0) = (b_n == c_n) ? a(n, 0, 0) + b(n, 0, 0) : a(n, 0, 0) + b(0, 0, 0);
                for(int x = 0; x < c_c; x++)
                {
                    c(n, x, 0) = (b_c == c_c) ? a(n, x, 0) + b((b_n == c_n ? n : 0), x, 0)
                                              : a(n, x, 0) + b((b_n == c_n ? n : 0), 0, 0);
                    for(int h = 0; h < c_h; h++)
                    {
                        c(n, x, h) =
                            (b_h == c_h)
                                ? a(n, x, h) + b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), h)
                                : a(n, x, h) + b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), 0);
                    }
                }
            }
            break;

        case 2:
            std::tie(c_n, c_c) = miopen::tien<2>(c.desc.GetLengths());
            std::tie(b_n, b_c) = miopen::tien<2>(b.desc.GetLengths());
            for(int n = 0; n < c_n; n++)
            {
                c(n, 0) = (b_n == c_n) ? a(n, 0) + b(n, 0) : a(n, 0) + b(0, 0);
                for(int x = 0; x < c_c; x++)
                {
                    c(n, x) = (b_c == c_c) ? a(n, x) + b((b_n == c_n ? n : 0), x)
                                           : a(n, x) + b((b_n == c_n ? n : 0), 0);
                }
            }
            break;

        case 1:
            std::tie(c_n) = miopen::tien<1>(c.desc.GetLengths());
            std::tie(b_n) = miopen::tien<1>(b.desc.GetLengths());
            for(int n = 0; n < c_n; n++)
            {
                c(n) = (b_n == c_n) ? a(n) + b(n) : a(n) + b(0);
            }
            break;

        default:; // TODO:  some exception here
        }

// tensor_for_loop(a, b, c, a_dims, b_dims, 0);
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
