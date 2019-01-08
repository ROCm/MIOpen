/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#include <miopen/tensor_ops.hpp>
#include <utility>
#include <cstdlib>
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

template <class T>
struct verify_tensor_transform
{
    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    tensor<T> srcSuper;
    tensor<T> dstSuper;
    T alpha;
    T beta;

    verify_tensor_transform(const tensor<T>& psrc_super,
                            const tensor<T>& pdst_super,
                            const miopen::TensorDescriptor& psd,
                            const miopen::TensorDescriptor& pdd,
                            const T palpha,
                            const T pbeta)
    {
        srcDesc  = psd;
        dstDesc  = pdd;
        srcSuper = psrc_super;
        dstSuper = pdst_super;
        alpha    = palpha;
        beta     = pbeta;
    }

    tensor<T> cpu() const
    {
        tensor<T> dstSuperCpu = dstSuper;

        if(dstDesc.GetType() == miopenInt8 && srcDesc.GetType() == miopenInt8)
        {
            int x_c;
            std::tie(std::ignore, x_c, std::ignore, std::ignore) =
                miopen::tien<4>(srcDesc.GetLengths());

            int y_n, y_c, y_h, y_w;
            std::tie(y_n, y_c, y_h, y_w) = miopen::tien<4>(dstDesc.GetLengths());

            par_ford(y_n, y_c, y_h, y_w)([&](int l, int k, int i, int j) {
                if(k < x_c)
                {
                    dstSuperCpu(l, k, i, j) = srcSuper(l, k, i, j);
                }
                else
                {
                    dstSuperCpu(l, k, i, j) = T(0);
                }
            });
        }

        return dstSuperCpu;
    }

    tensor<T> gpu() const
    {
        tensor<T> dstSuperGpu = dstSuper;

        auto&& handle     = get_handle();
        auto dstSuper_dev = handle.Write(dstSuperGpu.data);
        auto srcSuper_dev = handle.Write(srcSuper.data);

        miopen::TransformTensor(
            handle, &alpha, srcDesc, srcSuper_dev.get(), &beta, dstDesc, dstSuper_dev.get());

        dstSuperGpu.data = handle.Read<T>(dstSuper_dev, dstSuperGpu.data.size());

        return dstSuperGpu;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Copy: " << std::endl;
        std::cout << "src super-tensor: " << srcSuper.desc.ToString() << std::endl;
        std::cout << "dst super-tensor: " << dstSuper.desc.ToString() << std::endl;
        std::cout << "src sub-tensor: " << srcDesc.ToString() << std::endl;
        std::cout << "dst sub-tensor: " << dstDesc.ToString() << std::endl;
    }
};

template <class T>
struct tensor_transform_driver : test_driver
{
    tensor<T> srcSuper_pad;
    tensor<T> dstSuper_pad;
    tensor<T> srcSuper_depad;
    tensor<T> dstSuper_depad;
    std::vector<int> srcLens;

    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;

    tensor_transform_driver()
    {
        disabled_cache = true;
        add(srcLens,
            "srcLens",
            generate_data({{32, 11, 32, 16},
                           {16, 30, 16, 16},
                           {15, 1, 14, 14},
                           {10, 16, 7, 7},
                           {1, 1, 1, 1}}));
    }

    void run()
    {
        unsigned long max_value =
            miopen_type<T>{} == miopenHalf ? 5 : miopen_type<T>{} == miopenInt8 ? 127 : 17;

        srcSuper_pad   = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
        dstSuper_depad = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
        srcDesc        = miopen::TensorDescriptor(this->type, srcLens.data(), srcLens.size());

        srcLens[1]     = (srcLens[1] % 4 == 0) ? srcLens[1] : ((srcLens[1] + 3) / 4) * 4;
        dstSuper_pad   = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
        srcSuper_depad = tensor<T>{srcLens}.generate(tensor_elem_gen_integer{max_value});
        dstDesc        = miopen::TensorDescriptor(this->type, srcLens.data(), srcLens.size());

        if(srcDesc.GetLengths().size() == dstDesc.GetLengths().size())
        {
            verify_equals(verify_tensor_transform<T>{
                srcSuper_pad, dstSuper_pad, srcDesc, dstDesc, T(0), T(0)});

            verify_equals(verify_tensor_transform<T>{
                srcSuper_depad, dstSuper_depad, dstDesc, srcDesc, T(0), T(0)});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_transform_driver>(argc, argv); }
