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
void tensor_trans(const tensor<T>& src,
                  tensor<T>& dst,
                  const int stride_h,
                  const int stride_w,
                  const bool forward)
{
    int n, c, h_in, w_in, h_out, w_out;
    std::tie(n, c, h_in, w_in)                       = miopen::tien<4>(src.desc.GetLengths());
    std::tie(std::ignore, std::ignore, h_out, w_out) = miopen::tien<4>(dst.desc.GetLengths());

    int hw_in  = h_in * w_in;
    int chw_in = c * hw_in;
    int nhw_in = n * hw_in;

    int hw_out  = h_out * w_out;
    int chw_out = c * hw_out;
    int nhw_out = n * hw_out;

    for(int n_i = 0; n_i < n; n_i++)
        for(int c_i = 0; c_i < c; c_i++)
            for(int h_i = 0; h_i < (forward ? h_out : h_in); h_i++)
                for(int w_i = 0; w_i < (forward ? w_out : w_in); w_i++)
                {
                    int in_offset =
                        forward
                            ? (n_i * chw_in + c_i * hw_in + h_i * w_in * stride_h + w_i * stride_w)
                            : (c_i * nhw_in + n_i * hw_in + h_i * w_in + w_i);
                    int out_offset = forward ? (c_i * nhw_out + n_i * hw_out + h_i * w_out + w_i)
                                             : (n_i * chw_out + c_i * hw_out +
                                                h_i * w_out * stride_h + w_i * stride_w);

                    dst.data[out_offset] = src.data[in_offset];
                }
}

template <class T>
struct verify_tensor_trans
{
    tensor<T> src;
    tensor<T> dst;
    bool forward;
    int stride_h;
    int stride_w;

    verify_tensor_trans(const tensor<T>& p_src,
                        const tensor<T>& p_dst,
                        const int p_stride_h,
                        const int p_stride_w,
                        const bool p_forward)
    {
        forward  = p_forward;
        src      = p_src;
        dst      = p_dst;
        stride_h = p_stride_h;
        stride_w = p_stride_w;
    }

    tensor<T> cpu() const
    {
        auto r = dst;

        tensor_trans(src, r, stride_h, stride_w, forward);

        return r;
    }

    tensor<T> gpu() const
    {
        auto r        = dst;
        auto&& handle = get_handle();
        auto src_dev  = handle.Write(src.data);
        auto dst_dev  = handle.Write(r.data);
        int n, c, h, w, h_out, w_out;
        std::tie(n, c, h, w)                             = miopen::tien<4>(src.desc.GetLengths());
        std::tie(std::ignore, std::ignore, h_out, w_out) = miopen::tien<4>(dst.desc.GetLengths());

        miopenDataType_t type = miopenFloat;
        if(std::is_same<T, int8_t>::value)
            type = miopenInt8;
        else if(std::is_same<T, half_float::half>::value)
            type = miopenHalf;

        if(forward)
            miopen::transpose_NCHW2CNHW(handle,
                                        n,
                                        c,
                                        h,
                                        w,
                                        h_out,
                                        w_out,
                                        src_dev.get(),
                                        dst_dev.get(),
                                        0,
                                        0,
                                        stride_h,
                                        stride_w,
                                        type);
        else
            miopen::transpose_CNHW2NCHW(handle,
                                        n,
                                        c,
                                        h,
                                        w,
                                        h_out,
                                        w_out,
                                        src_dev.get(),
                                        dst_dev.get(),
                                        0,
                                        0,
                                        stride_h,
                                        stride_w,
                                        type);
        r.data = handle.Read<T>(dst_dev, dst.data.size());

        return r;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Transpose: " << std::endl;
        std::cout << "src tensor: " << src.desc.ToString() << std::endl;
        std::cout << "dst tensor: " << dst.desc.ToString() << std::endl;
    }
};

template <class T>
struct tensor_vec_driver : test_driver
{
    tensor<T> src;
    tensor<T> dst;

    int stride_h = 1, stride_w = 1;

    bool forw = true;

    std::vector<int> src_lens;

    std::vector<std::vector<int>> get_tensor_src()
    {
        return {
            {64, 64, 56, 56},   {64, 64, 56, 56},  {64, 256, 56, 56},  {64, 64, 55, 55},
            {64, 64, 55, 55},   {64, 256, 55, 55}, {64, 128, 28, 28},  {64, 512, 28, 28},
            {64, 256, 28, 28},  {64, 128, 28, 28}, {64, 256, 28, 28},  {64, 512, 28, 28},
            {64, 640, 28, 28},  {64, 256, 28, 28}, {64, 1024, 14, 14}, {64, 256, 14, 14},
            {64, 256, 14, 14},  {64, 512, 14, 14}, {64, 512, 14, 14},  {64, 1024, 14, 14},
            {64, 1280, 14, 14}, {64, 512, 14, 14}, {64, 512, 7, 7},    {64, 512, 7, 7},
            {64, 2048, 7, 7},   {64, 2560, 7, 7},  {64, 1024, 7, 7},   {64, 1024, 7, 7},
            {64, 1024, 7, 7},   {64, 2048, 7, 7},  {128, 127, 28, 28}, {256, 255, 14, 14},
            {512, 511, 7, 7},   {63, 63, 56, 56},  {127, 127, 28, 28}, {255, 255, 14, 14},
            {511, 511, 7, 7},   {64, 63, 56, 28},  {128, 127, 28, 14}, {256, 255, 14, 7},
        };
    }

    tensor_vec_driver()
    {
        disabled_cache = true;
        add(src_lens, "srcLens", generate_data(get_tensor_src()));
        add(forw, "forw", generate_data({true, false}));
        add(stride_h, "stride", generate_data({1, 2}));

        auto&& handle = get_handle();
        handle.EnableProfiling();
    }

    void run()
    {
        auto dst_lens = src_lens;

        stride_w = stride_h;

        if(forw)
        {
            dst_lens[2] = dst_lens[2] / stride_h;
            dst_lens[3] = dst_lens[3] / stride_w;
        }
        else
        {
            src_lens[2] = src_lens[2] / stride_h;
            src_lens[3] = src_lens[3] / stride_w;
        }

        unsigned long max_value =
            miopen_type<T>{} == miopenHalf ? 5 : miopen_type<T>{} == miopenInt8 ? 127 : 17;
        src = tensor<T>{src_lens}.generate(tensor_elem_gen_integer{max_value});
        dst = tensor<T>{dst_lens}.generate(tensor_elem_gen_integer{max_value});

        verify_equals(verify_tensor_trans<T>{src, dst, stride_h, stride_w, forw});
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_vec_driver>(argc, argv); }
