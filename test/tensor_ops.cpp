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

#define MIO_TENSOROP_DEBUG 0

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
    tensor<T> aSuper;
    tensor<T> bSuper;
    tensor<T> cSuper;

    miopen::TensorDescriptor aDesc;
    miopen::TensorDescriptor bDesc;
    miopen::TensorDescriptor cDesc;

    float alpha0;
    float alpha1;
    float beta;

    int aOffset;
    int bOffset;
    int cOffset;

    verify_tensor_ops(const tensor<T>& paSuper,
                      const tensor<T>& pbSuper,
                      const tensor<T>& pcSuper,
                      const miopen::TensorDescriptor& paDesc,
                      const miopen::TensorDescriptor& pbDesc,
                      const miopen::TensorDescriptor& pcDesc,
                      std::vector<int>& offsets,
                      float palpha0 = 1,
                      float palpha1 = 1,
                      float pbeta   = 0)
    {
        aSuper = paSuper;
        bSuper = pbSuper;
        cSuper = pcSuper;

        aDesc = paDesc;
        bDesc = pbDesc;
        cDesc = pcDesc;

        aOffset = offsets[0];
        bOffset = offsets[1];
        cOffset = offsets[2];

        alpha0 = palpha0;
        alpha1 = palpha1;
        beta   = pbeta;
    }

    T add_elem(T aelem, T belem) { return aelem + belem; }
    T mul_elem(T aelem, T belem) { return aelem * belem; }
    T max_elem(T aelem, T belem) { return ((aelem > belem) ? aelem : belem); }
    T min_elem(T aelem, T belem) { return ((aelem < belem) ? aelem : belem); }

    void tensor_for_loop(int recurr_aoffset, int recurr_boffset, int recurr_coffset, int dim)
    {

        int astride                = aDesc.GetStrides()[dim];
        int bstride                = bDesc.GetStrides()[dim];
        int cstride                = cDesc.GetStrides()[dim];
        std::vector<size_t> a_dims = cDesc.GetLengths();
        std::vector<size_t> b_dims = bDesc.GetLengths();

        // printf("cstride: %d\n", cstride);

        for(int idx = 0; idx < a_dims[dim]; idx++)
        {
            size_t aindex = recurr_aoffset + astride * idx;
            size_t cindex = recurr_coffset + cstride * idx;
            size_t bindex =
                (b_dims[dim] == a_dims[dim]) ? recurr_boffset + bstride * idx : recurr_boffset;

            // if((bindex < bten.desc.GetElementSize()) && (dim == a_dims.size() - 1))
            if(dim == (a_dims.size() - 1))
            {
                cSuper[cindex + cOffset] =
                    add_elem(aSuper[aindex + aOffset] * alpha0, bSuper[bindex + bOffset] * alpha1) +
                    beta * cSuper[cindex + cOffset];
            }
            if(dim < (a_dims.size() - 1))
            {

                tensor_for_loop(aindex, bindex, cindex, dim + 1);
            }
        }
        return;
    }

    tensor<T> cpu()
    {
        std::fill(cSuper.begin(), cSuper.end(), 1);
        tensor_for_loop(0, 0, 0, 0);

        return cSuper;
    }

    tensor<T> gpu()
    {

        std::fill(cSuper.begin(), cSuper.end(), 1);
        auto&& handle = get_handle();

        auto aSuper_dev = handle.Write(aSuper.data);
        auto bSuper_dev = handle.Write(bSuper.data);
        auto cSuper_dev = handle.Write(cSuper.data);

        miopen::OpTensor(handle,
                         miopenTensorOpAdd,
                         // miopenTensorOpMax,
                         // miopenTensorOpMin,
                         // miopenTensorOpMul,
                         &alpha0,
                         aDesc,
                         aSuper_dev.get(),
                         &alpha1,
                         bDesc,
                         bSuper_dev.get(),
                         &beta,
                         cDesc,
                         cSuper_dev.get(),
                         aOffset,
                         bOffset,
                         cOffset);

        cSuper.data = handle.Read<T>(cSuper_dev, cSuper.data.size());

        return cSuper;
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
    tensor<T> c;
    tensor<T> aSuper;
    tensor<T> bSuper;
    tensor<T> cSuper;
    miopen::TensorDescriptor aDesc;
    miopen::TensorDescriptor bDesc;
    miopen::TensorDescriptor cDesc;
    std::vector<int> tensorlens;
    std::vector<int> offsets;
    std::vector<float> alphabeta;

    tensor_ops_driver()
    {

#if(MIO_TENSOROP_DEBUG == 1)
        printf("Generating super tensors...");
        fflush(nullptr);
#endif

        std::vector<int> alens = {{32, 16, 16, 16, 16}};
        std::vector<int> blens = {{32, 16, 16, 16, 16}};
        std::vector<int> clens = {{32, 16, 16, 16, 16}};

        aSuper = tensor<T>{alens}.generate(rand_gen{});
        bSuper = tensor<T>{blens}.generate(rand_gen{});
        cSuper = tensor<T>{clens}.generate(rand_gen{});

#if(MIO_TENSOROP_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
        printf("Generating sub-tensors lengths...");
        fflush(nullptr);
#endif

        std::vector<std::vector<float>> get_scalars = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        std::vector<std::vector<int>> get_offsets   = {{1, 11, 1}, {2, 12, 1}, {3, 13, 1}};

        add(tensorlens, "tensorlens", generate_data(get_sub_tensor()));
        add(offsets, "offsets", generate_data(get_offsets));
        add(alphabeta, "alphabeta", generate_data(get_scalars));

#if(MIO_TENSOROP_DEBUG == 1)
        printf("done.\n");
        fflush(nullptr);
#endif
    }

    void run()
    {
        std::vector<size_t> aSuperStrides = aSuper.desc.GetStrides();
        std::vector<size_t> bSuperStrides = bSuper.desc.GetStrides();
        std::vector<size_t> cSuperStrides = cSuper.desc.GetStrides();
        std::vector<int> astrides(aSuperStrides.begin() + (5 - tensorlens.size()),
                                  aSuperStrides.end());
        std::vector<int> bstrides(bSuperStrides.begin() + (5 - tensorlens.size()),
                                  bSuperStrides.end());
        std::vector<int> cstrides(cSuperStrides.begin() + (5 - tensorlens.size()),
                                  cSuperStrides.end());

        aDesc = miopen::TensorDescriptor(
            miopenFloat, tensorlens.data(), astrides.data(), tensorlens.size());
        bDesc = miopen::TensorDescriptor(
            miopenFloat, tensorlens.data(), bstrides.data(), tensorlens.size());
        cDesc = miopen::TensorDescriptor(
            miopenFloat, tensorlens.data(), cstrides.data(), tensorlens.size());

        float alpha0 = alphabeta[0];
        float alpha1 = alphabeta[1];
        float beta   = alphabeta[2];

        if(aDesc.GetLengths().size() == cDesc.GetLengths().size())
        {
            printf("offsets {src, dst}: %d, %d\n", offsets[0], offsets[1]);
            verify(verify_tensor_ops<T>{
                aSuper, bSuper, cSuper, aDesc, bDesc, cDesc, offsets, alpha0, alpha1, beta});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_ops_driver<float>>(argc, argv); }
