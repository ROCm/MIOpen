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
struct verify_tensor_cast
{
    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    tensor<int> srcSuper;
    tensor<T> dstSuper;
    int srcOffset;
    int dstOffset;
    float alpha;
    float max_val;

    verify_tensor_cast(const tensor<int>& psrc_super,
                       const tensor<T>& pdst_super,
                       const miopen::TensorDescriptor& psd,
                       const miopen::TensorDescriptor& pdd,
                       std::vector<int> offsets,
                       const float palpha,
                       const float pmax_val)
    {
        srcDesc   = psd;
        dstDesc   = pdd;
        srcSuper  = psrc_super;
        dstSuper  = pdst_super;
        srcOffset = offsets[0];
        dstOffset = offsets[1];
        alpha     = palpha;
        max_val   = pmax_val;
    }

    void tensor_cast_for_loop(tensor<T>& dstSuperCpu,
                              int src_offset_index,
                              int dst_offset_index,
                              int dim) const
    {
        auto src_stride = srcDesc.GetStrides()[dim];
        auto dst_stride = dstDesc.GetStrides()[dim];

        for(int idx = 0; idx < srcDesc.GetLengths()[dim]; idx++)
        {
            std::size_t src_super_index =
                ((dim == 0) ? srcOffset : 0) + src_offset_index + src_stride * idx;
            std::size_t dst_super_index =
                ((dim == 0) ? dstOffset : 0) + dst_offset_index + dst_stride * idx;

            if(dim < (srcDesc.GetLengths().size() - 1))
            {
                tensor_cast_for_loop(dstSuperCpu, src_super_index, dst_super_index, dim + 1);
            }
            if(dst_super_index < dstSuperCpu.desc.GetElementSpace() &&
               src_super_index < srcSuper.desc.GetElementSpace())
            {
                float temp_val               = float(srcSuper[src_super_index]) * alpha;
                dstSuperCpu[dst_super_index] = T(temp_val >= max_val ? max_val : temp_val);
            }
        }
    }

    tensor<T> cpu() const
    {
        tensor<T> dstSuperCpu = dstSuper;

        tensor_cast_for_loop(dstSuperCpu, 0, 0, 0);

        return dstSuperCpu;
    }

    tensor<T> gpu() const
    {
        tensor<T> dstSuperGpu = dstSuper;

        auto&& handle     = get_handle();
        auto dstSuper_dev = handle.Write(dstSuperGpu.data);
        auto srcSuper_dev = handle.Write(srcSuper.data);

        miopen::CastTensor(handle,
                           &alpha,
                           srcDesc,
                           srcSuper_dev.get(),
                           dstDesc,
                           dstSuper_dev.get(),
                           srcOffset,
                           dstOffset);

        dstSuperGpu.data = handle.Read<T>(dstSuper_dev, dstSuperGpu.data.size());

        return dstSuperGpu;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Cast: " << std::endl;
        std::cout << "src super-tensor: " << srcSuper.desc.ToString() << std::endl;
        std::cout << "dst super-tensor: " << dstSuper.desc.ToString() << std::endl;
        std::cout << "src sub-tensor: " << srcDesc.ToString() << std::endl;
        std::cout << "dst sub-tensor: " << dstDesc.ToString() << std::endl;
    }
};

template <class T>
struct tensor_cast_driver : test_driver
{
    tensor<int> srcSuper;
    tensor<T> dstSuper;
    std::vector<int> srcSuperLens;
    std::vector<int> dstSuperLens;
    float alpha   = 1.0;
    float max_val = 0.;

    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    std::vector<int> castLens;
    std::vector<int> offsets;

    tensor_cast_driver()
    {
        std::vector<int> src_lens = {32, 16, 32, 16, 16};
        std::vector<int> dst_lens = {32, 32, 16, 16, 16};

        add(srcSuperLens, "srcSuperLens", generate_data({src_lens}, src_lens));
        add(dstSuperLens, "dstSuperLens", generate_data({dst_lens}, dst_lens));
        add(castLens, "castLens", generate_data(get_sub_tensor(), {32, 8, 10}));
        add(offsets, "offsets", generate_data(get_tensor_offsets(), {7, 11}));
        add(alpha, "alpha", generate_data({1.0 / 127 / 127, 1.0 / 127, 127.0, 1.0}));
    }

    void run()
    {
        unsigned long max_value = miopen_type<T>{} == miopenHalf ? 5 : 32767;
        max_val = miopen_type<T>{} == miopenHalf ? 65504.0 : miopen_type<T>{} == miopenInt8
                                                                 ? 127.0
                                                                 : miopen_type<T>{} == miopenInt32
                                                                       ? 2147483647.0
                                                                       : 3.402823466e+38F;

        srcSuper = tensor<int>{srcSuperLens}.generate(tensor_elem_gen_integer{max_value});
        dstSuper = tensor<T>{dstSuperLens}.generate(tensor_elem_gen_integer{max_value});

        std::vector<size_t> srcSuperStrides = srcSuper.desc.GetStrides();
        std::vector<size_t> dstSuperStrides = dstSuper.desc.GetStrides();
        std::vector<int> src_super_strides(srcSuperStrides.begin() +
                                               (srcSuper.desc.GetSize() - castLens.size()),
                                           srcSuperStrides.end());
        std::vector<int> dst_super_strides(dstSuperStrides.begin() +
                                               (dstSuper.desc.GetSize() - castLens.size()),
                                           dstSuperStrides.end());

        srcDesc = miopen::TensorDescriptor(
            miopenInt32, castLens.data(), src_super_strides.data(), castLens.size());
        dstDesc = miopen::TensorDescriptor(
            miopen_type<T>{}, castLens.data(), dst_super_strides.data(), castLens.size());

        if(srcDesc.GetLengths().size() == dstDesc.GetLengths().size())
        {
            verify_equals(verify_tensor_cast<T>{
                srcSuper, dstSuper, srcDesc, dstDesc, offsets, alpha, max_val});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_cast_driver>(argc, argv); }
