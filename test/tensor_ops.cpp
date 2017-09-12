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
    
    T add_elem(T a, T b){
        return a + b;
    }
    
    //template<typename F>
    void tensor_for_loop(const tensor<T>& a, const tensor<T>& b,  tensor<T>& c, 
            const std::vector<size_t>& a_dims, const std::vector<size_t>& b_dims, 
            size_t acoffset, size_t acscale, 
            size_t boffset, size_t bscale, int dim){
        
//        static int cnt = 0;
//        printf("count: %d\n", cnt++);
//        printf("DIM: %d adim-1: %d\n", dim, int(a_dims.size()) - 1);
                
        for(int idx = 0; idx < a_dims[dim]; idx++){
            size_t acindex = acoffset + idx;
            size_t bidx  = (b_dims[dim] == a_dims[dim]) ? acindex : 0;
            size_t bindex = boffset + bidx;
//            printf("acscale :%d, bscale: %d\n", acscale, bscale);
//            printf("acoffset :%d, boffset: %d\n", acoffset, boffset);
//            printf("acindex: %d, bindex: %d, acsize: %d, bsize: %d\n", 
//                    acindex, bindex, a.desc.GetElementSize(), b.desc.GetElementSize());
            if (bindex < b.desc.GetElementSize()) c[acindex] = add_elem(a[acindex], b[bindex]);
            
            if(dim < (a_dims.size() - 1)){
                
                int newdim = dim+1;
                
                size_t newacoffset = acscale*idx + acoffset;
                size_t newacscale  = acscale / a_dims[dim];
                
                size_t newboffset = bscale*bidx + boffset;
                size_t newbscale  = bscale / b_dims[dim];
                
                tensor_for_loop(a, b, c, 
                                a_dims, b_dims, 
                                newacoffset, newacscale, 
                                newboffset, newbscale, 
                                newdim);
            }
        }
//        cnt--;
        return;
    }
    
    
    
    tensor<T> cpu()
    {//TODO: (dlowell) make this variable length
        c = a;
        std::fill(c.begin(), c.end(), 0);
        const std::vector<size_t>& a_dims = a.desc.GetLengths();
        const std::vector<size_t>& b_dims = b.desc.GetLengths();
        
        //const std::vector<size_t>& c_dims = c.desc.GetLengths();
        
        /*for(int n = 0; n < c_n; n++)
        {
        
        for(int n = 0; n < c_n; n++)
        {
            c(n, 0, 0, 0) = (b_n == c_n) ? a(n, 0, 0, 0) + b(n, 0, 0, 0)
                                         : a(n, 0, 0, 0) + b(0, 0, 0, 0);
            for(int x = 0; x < c_c; x++)
            {
                c(n, x, 0, 0) = (b_c == c_c) ? a(n, x, 0, 0) + b((b_n == c_n ? n : 0), x, 0, 0)
                                             : a(n, x, 0, 0) + b((b_n == c_n ? n : 0), 0, 0, 0);

                for(int h = 0; h < c_h; h++)
                {
                    c(n, x, h, 0) =
                        (b_h == c_h)
                            ? a(n, x, h, 0) + b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), h, 0)
                            : a(n, x, h, 0) + b((b_n == c_n ? n : 0), (b_c == c_c ? x : 0), 0, 0);

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
        }*/
        
        size_t acscale = a.desc.GetElementSize() / a_dims[0];
        size_t bscale = b.desc.GetElementSize() / b_dims[0];
        
        tensor_for_loop(a, b, c, a_dims, b_dims, 0, acscale, 0, bscale, 0);
        return c;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();

        c = a;
        //return c;
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
        add(a, "a", generate_tensor(get_tensor_a(), {11, 7, 13, 13, 7}));
        add(b, "b", generate_tensor(get_tensor_b(), {1, 7, 1, 13, 7}));
//        add(a, "a", generate_tensor(get_tensor_a(), {11, 7, 13, 13}));
//        add(b, "b", generate_tensor(get_tensor_b(), {1, 7, 1, 1}));
//        add(a, "a", generate_tensor(get_tensor_a(), {11, 7, 13}));
//        add(b, "b", generate_tensor(get_tensor_b(), {1, 7, 1}));
//         add(a, "a", generate_tensor(get_tensor_a(), {11, 7}));
//         add(b, "b", generate_tensor(get_tensor_b(), {1, 7}));
    }

    std::set<std::vector<int>> get_tensor_a()
    {
        std::vector<std::vector<int>> a_dims{
            {32, 8, 16, 16},
        };
        return (std::set<std::vector<int>>(a_dims.begin(), a_dims.end()));
    }

    std::set<std::vector<int>> get_tensor_b()
    {
        std::vector<std::vector<int>> b_dims{
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
        };
        return (std::set<std::vector<int>>(b_dims.begin(), b_dims.end()));
    }

    //void run() { verify(verify_tensor_ops<T, 2>{a, b}); }
    //void run() { verify(verify_tensor_ops<T, 4>{a, b}); }
    void run() { verify(verify_tensor_ops<T>{a, b}); }
};

int main(int argc, const char* argv[]) { test_drive<tensor_ops_driver<float>>(argc, argv); }
