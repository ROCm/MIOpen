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
#include "tensor_util.hpp"

#define MIO_TENSORSET_DEBUG 0

template <typename T>
struct set_data_t
{
    const T alpha;

    void operator()(T& r_data) const { r_data = alpha; }
};

template <class T>
struct verify_tensor_set
{
    miopen::TensorDescriptor subDesc;
    tensor<T> super;
    int offset;
    T alpha;

    verify_tensor_set(const tensor<T>& rSuper,
                      const miopen::TensorDescriptor& rSubDesc,
                      const int offsetIn,
                      const T alphaIn)
    {
        subDesc = rSubDesc;
        super   = rSuper;
        offset  = offsetIn;
        alpha   = alphaIn;
    }

    tensor<T> cpu() const
    {

#if(MIO_TENSORSET_DEBUG == 1)
        printf("CPU test start...");
        fflush(nullptr);
#endif
        tensor<T> superCpu = super;

        const set_data_t<T> data_operator = {alpha};

        operate_over_subtensor(data_operator, superCpu, subDesc, offset);

#if(MIO_TENSORSET_DEBUG == 1)
        printf("done\n");
        fflush(nullptr);
#endif
        return superCpu;
    }

    tensor<T> gpu() const
    {

#if(MIO_TENSORSET_DEBUG == 1)
        printf("GPU test start...");
        fflush(nullptr);
#endif

        tensor<T> superGpu = super;

        auto&& handle  = get_handle();
        auto super_dev = handle.Write(superGpu.data);

        struct timeval start, end;
        gettimeofday(&start, NULL);
        miopen::SetTensor(handle, subDesc, super_dev.get(), &alpha, offset);
        gettimeofday(&end, NULL);

        long w_time = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)); 

        std::size_t nbyte = sizeof(T{}) * std::accumulate(
            subDesc.GetLengths().begin(), subDesc.GetLengths().end(), std::size_t{1}, std::multiplies<std::size_t>());

        std::cout << "wall time: " << w_time/1000.0 << "ms" << std::endl;
        std::cout << "bandwidth: " << nbyte/((std::size_t(1)<<30)*handle.GetKernelTime()/1000) << "GB/s" << std::endl;

        superGpu.data = handle.Read<T>(super_dev, superGpu.data.size());

#if(MIO_TENSORSET_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
#endif
        return superGpu;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Set: " << std::endl;
        std::cout << "super-tensor: " << super.desc.ToString() << std::endl;
        std::cout << "sub-tensor: " << subDesc.ToString() << std::endl;
    }
};

template <class T>
struct tensor_set_driver : test_driver
{
    tensor<T> super;
    miopen::TensorDescriptor subDesc;
    std::vector<int> subLens;
    int offset = 0;

#if 0
    tensor_set_driver()
    {

#if(MIO_TENSORSET_DEBUG == 1)
        printf("Generating super tensors...");
        fflush(nullptr);
#endif
        std::vector<int> lens = {{32, 32, 16, 16, 16}};
        super                 = tensor<T>{lens}.generate(rand_gen{});

#if(MIO_TENSORSET_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
        printf("Generating sub-tensors lengths...");
        fflush(nullptr);
#endif

        add(subLens, "subLens", generate_data(get_sub_tensor(), {32, 8, 10}));
        add(offset, "offset", generate_data(get_tensor_offset(), 7));

#if(MIO_TENSORSET_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
#endif
    }
#endif

    tensor_set_driver()
    {

#if(MIO_TENSORSET_DEBUG == 1)
        printf("Generating super tensors...");
        fflush(nullptr);
#endif
        std::vector<int> lens = {{1,  64, 128, 192, 192}};
//      std::vector<int> lens = {{1, 128, 256, 64, 64}};
        super                 = tensor<T>{lens}.generate(rand_gen{});

#if(MIO_TENSORSET_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
        printf("Generating sub-tensors lengths...");
        fflush(nullptr);
#endif

//      add(subLens, "subLens", generate_data(get_sub_tensor(), {1,32,128,192,192}));
        add(subLens, "subLens", generate_data(get_sub_tensor(), {1*32*128*192*192}));
        add(offset, "offset", generate_data(get_tensor_offset(), 128*192*192));

//      add(subLens, "subLens", generate_data(get_sub_tensor(), {1,64,256,64,64}));
//      add(subLens, "subLens", generate_data(get_sub_tensor(), {1*64*256*64*64}));
//      add(offset, "offset", generate_data(get_tensor_offset(), 256*64*64));

#if(MIO_TENSORSET_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
#endif
    }


    void run()
    {
        std::vector<size_t> superStrides = super.desc.GetStrides();
        std::vector<int> subStrides(superStrides.begin() + (5 - subLens.size()),
                                    superStrides.end());

        subDesc =
            miopen::TensorDescriptor(this->type, subLens.data(), subStrides.data(), subLens.size());

#if(MIO_TENSORSET_DEBUG == 1)
        printf("offset: %d\n", offset);
#endif
        verify_equals(verify_tensor_set<T>{super, subDesc, offset, T(1.111)});
    }
};

int main(int argc, const char* argv[])
{

#if(MIO_TENSORSET_DEBUG == 1)
    printf("Starting.\n");
#endif
    test_drive<tensor_set_driver>(argc, argv);
}
