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
#include <miopen/tensor_ops.hpp>
#include <utility>
#include <cstdlib>
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

template <class T>
struct verify_tensor_copy
{
    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    tensor<T> srcSuper;
    tensor<T> dstSuper;
    int srcOffset;
    int dstOffset;

    verify_tensor_copy(const tensor<T>& psrc_super,
                       const tensor<T>& pdst_super,
                       const miopen::TensorDescriptor& psd,
                       const miopen::TensorDescriptor& pdd,
                       std::vector<int> offsets)
    {
        srcDesc   = psd;
        dstDesc   = pdd;
        srcSuper  = psrc_super;
        dstSuper  = pdst_super;
        srcOffset = offsets[0];
        dstOffset = offsets[1];
    }

    void tensor_copy_for_loop(tensor<T>& dstSuperCpu,
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
                tensor_copy_for_loop(dstSuperCpu, src_super_index, dst_super_index, dim + 1);
            }
            if(dst_super_index < dstSuperCpu.desc.GetElementSpace() &&
               src_super_index < srcSuper.desc.GetElementSpace())
            {
                dstSuperCpu[dst_super_index] = srcSuper[src_super_index];
            }
        }
    }

    tensor<T> cpu() const
    {
        tensor<T> dstSuperCpu = dstSuper;

        tensor_copy_for_loop(dstSuperCpu, 0, 0, 0);

        return dstSuperCpu;
    }

    tensor<T> gpu() const
    {
        tensor<T> dstSuperGpu = dstSuper;

        auto&& handle     = get_handle();
        auto dstSuper_dev = handle.Write(dstSuperGpu.data);
        auto srcSuper_dev = handle.Write(srcSuper.data);

        miopen::CopyTensor(
            handle, srcDesc, srcSuper_dev.get(), dstDesc, dstSuper_dev.get(), srcOffset, dstOffset);

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
struct tensor_copy_driver : test_driver
{
    tensor<T> srcSuper;
    tensor<T> dstSuper;
    std::vector<int> srcSuperLens;
    std::vector<int> dstSuperLens;

    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    std::vector<int> copyLens;
    std::vector<int> offsets;

    tensor_copy_driver()
    {
        disabled_cache            = true;
        std::vector<int> src_lens = {32, 16, 32, 16, 16};
        std::vector<int> dst_lens = {32, 32, 16, 16, 16};

        add(srcSuperLens, "srcSuperLens", generate_data({src_lens}, src_lens));
        add(dstSuperLens, "dstSuperLens", generate_data({dst_lens}, dst_lens));
        add(copyLens, "copyLens", generate_data(get_sub_tensor(), {32, 8, 10}));
        add(offsets, "offsets", generate_data(get_tensor_offsets(), {7, 11}));
    }

    void run()
    {
        unsigned long max_value =
            miopen_type<T>{} == miopenHalf ? 5 : miopen_type<T>{} == miopenInt8 ? 127 : 17;

        srcSuper = tensor<T>{srcSuperLens}.generate(tensor_elem_gen_integer{max_value});
        dstSuper = tensor<T>{dstSuperLens}.generate(tensor_elem_gen_integer{max_value});

        std::vector<size_t> srcSuperStrides = srcSuper.desc.GetStrides();
        std::vector<size_t> dstSuperStrides = dstSuper.desc.GetStrides();
        std::vector<int> src_super_strides(srcSuperStrides.begin() +
                                               (srcSuper.desc.GetSize() - copyLens.size()),
                                           srcSuperStrides.end());
        std::vector<int> dst_super_strides(dstSuperStrides.begin() +
                                               (dstSuper.desc.GetSize() - copyLens.size()),
                                           dstSuperStrides.end());

        srcDesc = miopen::TensorDescriptor(
            this->type, copyLens.data(), src_super_strides.data(), copyLens.size());
        dstDesc = miopen::TensorDescriptor(
            this->type, copyLens.data(), dst_super_strides.data(), copyLens.size());

        if(srcDesc.GetLengths().size() == dstDesc.GetLengths().size())
        {
            verify_equals(verify_tensor_copy<T>{srcSuper, dstSuper, srcDesc, dstDesc, offsets});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_copy_driver>(argc, argv); }
