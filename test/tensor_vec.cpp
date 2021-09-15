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
#include <sys/time.h>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/util.hpp>
#include <utility>
#include <cstdlib>
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

template <class T>
void tensor_vec_forward(
    const tensor<T>& src, tensor<T>& dst, const bool trans, int vec_size, float alpha, float beta)
{
    int n_dst, c_dst, h, w;
    std::tie(n_dst, c_dst, h, w) = miopen::tien<4>(dst.desc.GetLengths());
    int n_src, c_src;
    std::tie(n_src, c_src, std::ignore, std::ignore) = miopen::tien<4>(src.desc.GetLengths());
    int in_hw                                        = h * w;
    int in_chw                                       = c_src * in_hw;

    int out_w  = w * vec_size;
    int out_hw = h * out_w;

    auto hi_shft = static_cast<int>(log2(vec_size));
    auto lo_mask = vec_size - 1;

    for(int n_i = 0; n_i < n_dst; n_i++)
    {
        for(int c_i = 0; c_i < c_dst; c_i++)
        {
            for(int h_i = 0; h_i < h; h_i++)
            {
                for(int w_i = 0; w_i < w; w_i++)
                {
                    int in_offset = n_i * in_chw + c_i * in_hw + h_i * w + w_i;

                    if(trans)
                    {
                        int out_nhw = n_dst * in_hw;
                        int n_hi_i  = n_i >> hi_shft;
                        int n_lo_i  = n_i & lo_mask;
                        int out_offset =
                            c_i * out_nhw + n_hi_i * out_hw + h_i * out_w + w_i * vec_size + n_lo_i;
                        dst.data[out_offset] = (n_i < n_src)
                                                   ? T(alpha * float(src.data[in_offset]) +
                                                       beta * float(dst.data[out_offset]))
                                                   : 0;
                    }
                    else
                    {
                        int out_chw = c_dst * in_hw;
                        int c_hi_i  = c_i >> hi_shft;
                        int c_lo_i  = c_i & lo_mask;
                        int out_offset =
                            n_i * out_chw + c_hi_i * out_hw + h_i * out_w + w_i * vec_size + c_lo_i;
                        dst.data[out_offset] = (c_i < c_src)
                                                   ? T(alpha * float(src.data[in_offset]) +
                                                       beta * float(dst.data[out_offset]))
                                                   : 0;
                    }
                }
            }
        }
    }
}

template <class T>
void tensor_vec_backward(
    const tensor<T>& src, tensor<T>& dst, const bool trans, int vec_size, float alpha, float beta)
{
    int n_dst, c_dst, h, w;
    std::tie(n_dst, c_dst, h, w) = miopen::tien<4>(dst.desc.GetLengths());
    int n_src, c_src;
    std::tie(n_src, c_src, std::ignore, std::ignore) = miopen::tien<4>(src.desc.GetLengths());
    int out_hw                                       = h * w;
    int out_chw                                      = c_dst * out_hw;

    int in_w  = w * vec_size;
    int in_hw = h * in_w;

    auto hi_shft = static_cast<int>(log2(vec_size));
    auto lo_mask = vec_size - 1;

    for(int n_i = 0; n_i < n_src; n_i++)
    {
        for(int c_i = 0; c_i < c_src; c_i++)
        {
            for(int h_i = 0; h_i < h; h_i++)
            {
                for(int w_i = 0; w_i < w; w_i++)
                {
                    int out_offset = n_i * out_chw + c_i * out_hw + h_i * w + w_i;

                    if(trans)
                    {
                        int in_nhw = n_src * out_hw;
                        int n_hi_i = n_i >> hi_shft;
                        int n_lo_i = n_i & lo_mask;
                        int in_offset =
                            c_i * in_nhw + n_hi_i * in_hw + h_i * in_w + w_i * vec_size + n_lo_i;
                        if(n_i < n_dst)
                            dst.data[out_offset] = T(alpha * float(src.data[in_offset]) +
                                                     beta * float(dst.data[out_offset]));
                    }
                    else
                    {
                        int in_chw = c_src * out_hw;
                        int c_hi_i = c_i >> hi_shft;
                        int c_lo_i = c_i & lo_mask;
                        int in_offset =
                            n_i * in_chw + c_hi_i * in_hw + h_i * in_w + w_i * vec_size + c_lo_i;
                        if(c_i < c_dst)
                            dst.data[out_offset] = T(alpha * float(src.data[in_offset]) +
                                                     beta * float(dst.data[out_offset]));
                    }
                }
            }
        }
    }
}

template <class T>
struct verify_tensor_vec_forward
{
    tensor<T> src;
    tensor<T> dst;
    bool trans;
    float alpha;
    float beta;

    verify_tensor_vec_forward(const tensor<T>& p_src,
                              const tensor<T>& p_dst,
                              const bool p_trans,
                              const float palpha,
                              const float pbeta)
    {
        trans = p_trans;
        src   = p_src;
        dst   = p_dst;
        alpha = palpha;
        beta  = pbeta;
    }

    tensor<T> cpu() const
    {
        auto r       = dst;
        int vec_size = 4 / sizeof(T);
        tensor_vec_forward(src, r, trans, vec_size, alpha, beta);
        return r;
    }

    tensor<T> gpu() const
    {
        auto r        = dst;
        auto&& handle = get_handle();
        auto src_dev  = handle.Write(src.data);
        auto dst_dev  = handle.Write(r.data);
        int vec_size  = 4 / sizeof(T);
        miopen::transpose_NCHW2Vec(handle,
                                   src.desc.GetLengths(),
                                   src_dev.get(),
                                   dst_dev.get(),
                                   vec_size,
                                   trans,
                                   true,
                                   &alpha,
                                   &beta);
        r.data = handle.Read<T>(dst_dev, dst.data.size());
        return r;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Vectorization Forward: " << std::endl;
        std::cout << "src tensor: " << src.desc.ToString() << std::endl;
        std::cout << "dst tensor: " << dst.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_tensor_vec_backward
{
    tensor<T> src;
    tensor<T> dst;
    bool trans;
    float alpha;
    float beta;

    verify_tensor_vec_backward(const tensor<T>& p_src,
                               const tensor<T>& p_dst,
                               const bool p_trans,
                               const float palpha,
                               const float pbeta)
    {
        trans = p_trans;
        src   = p_src;
        dst   = p_dst;
        alpha = palpha;
        beta  = pbeta;
    }

    tensor<T> cpu() const
    {
        auto r       = dst;
        int vec_size = 4 / sizeof(T);
        tensor_vec_backward(src, r, trans, vec_size, alpha, beta);
        return r;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto r        = dst;
        auto src_dev  = handle.Write(src.data);
        auto dst_dev  = handle.Write(r.data);
        int vec_size  = 4 / sizeof(T);
        miopen::transpose_NCHW2Vec(handle,
                                   dst.desc.GetLengths(),
                                   src_dev.get(),
                                   dst_dev.get(),
                                   vec_size,
                                   trans,
                                   false,
                                   &alpha,
                                   &beta);
        r.data = handle.Read<T>(dst_dev, r.data.size());

        return r;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Vectorization BackWard: " << std::endl;
        std::cout << "src tensor: " << src.desc.ToString() << std::endl;
        std::cout << "dst tensor: " << dst.desc.ToString() << std::endl;
    }
};

template <class T>
struct tensor_vec_driver : test_driver
{
    tensor<T> src;
    tensor<T> dst;

    bool trans = false;
    bool forw  = true;

    std::vector<int> src_lens;
    std::vector<float> scales;

    std::vector<std::vector<int>> get_tensor_src()
    {
        return {{64, 64, 56, 56},   {64, 64, 56, 56},  {64, 256, 56, 56},  {64, 64, 55, 55},
                {64, 64, 55, 55},   {64, 256, 55, 55}, {64, 128, 28, 28},  {64, 512, 28, 28},
                {64, 256, 28, 28},  {64, 128, 28, 28}, {64, 256, 28, 28},  {64, 512, 28, 28},
                {64, 640, 28, 28},  {64, 256, 28, 28}, {64, 1024, 14, 14}, {64, 256, 14, 14},
                {64, 256, 14, 14},  {64, 512, 14, 14}, {64, 512, 14, 14},  {64, 1024, 14, 14},
                {64, 1280, 14, 14}, {64, 512, 14, 14}, {64, 512, 7, 7},    {64, 512, 7, 7},
                {64, 2048, 7, 7},   {64, 2560, 7, 7},  {64, 1024, 7, 7},   {64, 1024, 7, 7},
                {64, 1024, 7, 7},   {64, 2048, 7, 7},  {128, 127, 28, 28}, {256, 255, 14, 14},
                {512, 511, 7, 7},   {63, 63, 56, 56},  {127, 127, 28, 28}, {255, 255, 14, 14},
                {511, 511, 7, 7},   {64, 63, 56, 28},  {128, 127, 28, 14}, {256, 255, 14, 7},
                {512, 511, 7, 1},   {1, 511, 7, 1},    {511, 1, 7, 1}};
    }

    tensor_vec_driver()
    {
        add(src_lens, "srcLens", generate_data(get_tensor_src()));
        add(trans, "trans", generate_data({false, true}));
        add(forw, "forw", generate_data({true, false}));
        add(scales, "scales", generate_data({{1.f, 0.f}, {float(0.5), float(0.5)}}));

        auto&& handle = get_handle();
        handle.EnableProfiling();
    }

    void run()
    {
        float alpha = scales[0];
        float beta  = scales[1];

        if(std::is_same<T, float>::value)
        {
            std::cout << "VEC2 transpose does not support float type" << std::endl;
            return;
        }

        if(std::is_same<T, double>::value)
        {
            std::cout << "VEC2 transpose does not support double type" << std::endl;
            return;
        }

        if(!(miopen::float_equal(static_cast<const float>(alpha), 1.0) &&
             miopen::float_equal(static_cast<const float>(beta), 0.0)))
            return;

        auto dst_lens = src_lens;

        auto type_size = sizeof(T);
        auto vec_size  = 4 / type_size;

        if(trans)
            dst_lens[0] = (dst_lens[0] % vec_size != 0)
                              ? dst_lens[0] + (vec_size - dst_lens[0] % vec_size)
                              : dst_lens[0];
        else
            dst_lens[1] = (dst_lens[1] % vec_size != 0)
                              ? dst_lens[1] + (vec_size - dst_lens[1] % vec_size)
                              : dst_lens[1];

        unsigned long max_value =
            miopen_type<T>{} == miopenHalf ? 5 : miopen_type<T>{} == miopenInt8 ? 127 : 17;
        src = tensor<T>{src_lens}.generate(tensor_elem_gen_integer{max_value});
        dst = tensor<T>{dst_lens}.generate(tensor_elem_gen_integer{max_value});

        if(forw)
            verify_equals(verify_tensor_vec_forward<T>{src, dst, trans, alpha, beta});
        else
            verify_equals(verify_tensor_vec_backward<T>{dst, src, trans, alpha, beta});
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_vec_driver>(argc, argv); }
