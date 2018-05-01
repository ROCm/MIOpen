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

#define MIO_TENSORCOPY_DEBUG 0

template <class T>
struct verify_tensor_copy
{
    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    tensor<T> asuper;
    tensor<T> csuper;
    int srcOffset;
    int dstOffset;

    verify_tensor_copy(const tensor<T>& pasuper,
                       const tensor<T>& pcsuper,
                       const miopen::TensorDescriptor& psd,
                       const miopen::TensorDescriptor& pdd,
                       std::vector<int> offsets)
    {
        srcDesc   = psd;
        dstDesc   = pdd;
        asuper    = pasuper;
        csuper    = pcsuper;
        srcOffset = offsets[0];
        dstOffset = offsets[1];
    }

    void
    tensor_copy_for_loop(tensor<T>& csuperCpu, int aoffsetIndex, int coffsetIndex, int dim) const
    {
        auto astride = srcDesc.GetStrides()[dim];
        auto cstride = dstDesc.GetStrides()[dim];

        for(int idx = 0; idx < srcDesc.GetLengths()[dim]; idx++)
        {
            size_t aindex = ((dim == 0) ? srcOffset : 0) + aoffsetIndex + astride * idx;
            size_t cindex = ((dim == 0) ? dstOffset : 0) + coffsetIndex + cstride * idx;

            if(dim < (srcDesc.GetLengths().size() - 1))
            {
                tensor_copy_for_loop(csuperCpu, aindex, cindex, dim + 1);
            }
            if(cindex < csuperCpu.desc.GetElementSpace() && aindex < asuper.desc.GetElementSpace())
            {
                csuperCpu[cindex] = asuper[aindex];
            }
        }
    }

    tensor<T> cpu() const
    {

#if(MIO_TENSORCOPY_DEBUG == 1)
        printf("CPU test start...");
        fflush(nullptr);
#endif
        tensor<T> csuperCpu = csuper;

        tensor_copy_for_loop(csuperCpu, 0, 0, 0);

#if(MIO_TENSORCOPY_DEBUG == 1)
        printf("done\n");
        fflush(nullptr);
#endif
        return csuperCpu;
    }

    tensor<T> gpu() const
    {

#if(MIO_TENSORCOPY_DEBUG == 1)
        printf("GPU test start...");
        fflush(nullptr);
#endif

        tensor<T> csuperGpu = csuper;

        auto&& handle   = get_handle();
        auto csuper_dev = handle.Write(csuperGpu.data);
        auto asuper_dev = handle.Write(asuper.data);

        struct timeval start, end;
        gettimeofday(&start, NULL);
        miopen::CopyTensor(
            handle, srcDesc, asuper_dev.get(), dstDesc, csuper_dev.get(), srcOffset, dstOffset);
        gettimeofday(&end, NULL);

        long w_time =
            ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

        std::size_t nbyte = sizeof(T{}) * std::accumulate(srcDesc.GetLengths().begin(),
                                                          srcDesc.GetLengths().end(),
                                                          std::size_t{1},
                                                          std::multiplies<std::size_t>());

        std::cout << "wall time: " << w_time / 1000.0 << "ms" << std::endl;
        std::cout << "kernel time: " << handle.GetKernelTime() << "ms" << std::endl;
        std::cout << "bandwidth: "
                  << 2.0 * (float)nbyte / ((std::size_t(1) << 30) * handle.GetKernelTime() / 1000)
                  << "GB/s" << std::endl;

        csuperGpu.data = handle.Read<T>(csuper_dev, csuperGpu.data.size());

#if(MIO_TENSORCOPY_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
#endif
        return csuperGpu;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Copy: " << std::endl;
        std::cout << "a input super-tensor:  " << asuper.desc.ToString() << std::endl;
        std::cout << "c output super-tensor: " << csuper.desc.ToString() << std::endl;
        std::cout << "src sub-tensor: " << srcDesc.ToString() << std::endl;
        std::cout << "dst sub-tensor: " << dstDesc.ToString() << std::endl;
    }
};

template <class T>
struct tensor_copy_driver : test_driver
{
    tensor<T> a;
    tensor<T> c;
    tensor<T> aSuper;
    tensor<T> cSuper;
    miopen::TensorDescriptor srcDesc;
    miopen::TensorDescriptor dstDesc;
    std::vector<int> copylens;
    std::vector<int> offsets;

#if 1
    tensor_copy_driver()
    {

#if(MIO_TENSORCOPY_DEBUG == 1)
        printf("Generating super tensors...");
        fflush(nullptr);
#endif
        std::vector<int> alens = {{32, 16, 32, 16, 16}};
        std::vector<int> clens = {{32, 32, 16, 16, 16}};
        aSuper                 = tensor<T>{alens}.generate(rand_gen{});
        cSuper                 = tensor<T>{clens}.generate(rand_gen{});

#if(MIO_TENSORCOPY_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
        printf("Generating sub-tensors lengths...");
        fflush(nullptr);
#endif

        add(copylens, "copy-lens", generate_data(get_sub_tensor(), {32, 8, 10}));
        add(offsets, "offsets", generate_data(get_tensor_offsets(), {7, 11}));

#if(MIO_TENSORCOPY_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
#endif
    }
#endif

#if 0
    tensor_copy_driver()
    {
        std::vector<int> alens = {{1,  64, 128, 192, 192}};
        std::vector<int> clens = {{1,  64, 128, 192, 192}};
      //std::vector<int> clens = {{1, 128,  64, 192, 192}};
        aSuper                 = tensor<T>{alens}.generate(rand_gen{});
        cSuper                 = tensor<T>{clens}.generate(rand_gen{});

        add(copylens, "copy-lens", generate_data(get_sub_tensor(), {1, 64,  64, 192, 192}));
        add(offsets, "offsets", generate_data(get_tensor_offsets(), {192 * 192}));

    }
#endif
    void run()
    {
        std::vector<size_t> aSuperStrides = aSuper.desc.GetStrides();
        std::vector<size_t> cSuperStrides = cSuper.desc.GetStrides();
        std::vector<int> astrides(aSuperStrides.begin() + (5 - copylens.size()),
                                  aSuperStrides.end());
        std::vector<int> cstrides(cSuperStrides.begin() + (5 - copylens.size()),
                                  cSuperStrides.end());

        srcDesc =
            miopen::TensorDescriptor(this->type, copylens.data(), astrides.data(), copylens.size());
        dstDesc =
            miopen::TensorDescriptor(this->type, copylens.data(), cstrides.data(), copylens.size());

        if(srcDesc.GetLengths().size() == dstDesc.GetLengths().size())
        {
#if(MIO_TENSORCOPY_DEBUG == 1)
            printf("offsets {src, dst}: %d, %d\n", offsets[0], offsets[1]);
#endif
            verify_equals(verify_tensor_copy<T>{aSuper, cSuper, srcDesc, dstDesc, offsets});
        }
    }
};

int main(int argc, const char* argv[])
{

#if(MIO_TENSORCOPY_DEBUG == 1)
    printf("Starting.\n");
#endif
    test_drive<tensor_copy_driver>(argc, argv);
}
