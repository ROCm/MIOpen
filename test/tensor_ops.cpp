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

    int Aoffset;
    int Boffset;
    int Coffset;

    float alpha0;
    float alpha1;
    float beta;

    verify_tensor_ops(const tensor<T>& pa,
                      const tensor<T>& pb,
                      const tensor<T>& pc,
                      std::vector<size_t>& offsets,
                      float palpha0 = 1,
                      float palpha1 = 1,
                      float pbeta   = 0)
    {
        a = pa;
        b = pb;
        c = pc;

        Aoffset = offsets[0];
        Boffset = offsets[1];
        Coffset = offsets[2];

        alpha0 = palpha0;
        alpha1 = palpha1;
        beta   = pbeta;
    }

    // verify_tensor_ops(const tensor<T>& pa, const tensor<T>& pb, const tensor<T>& pc, const
    // std::vector<T>& dims)
    //{
    // a = pa(dims);
    // b = pb(dims);
    // c = pc(dims);
    //}

    T add_elem(T aelem, T belem) { return aelem + belem; }
    T mul_elem(T aelem, T belem) { return aelem * belem; }
    T max_elem(T aelem, T belem) { return ((aelem > belem) ? aelem : belem); }
    T min_elem(T aelem, T belem) { return ((aelem < belem) ? aelem : belem); }

    void tensor_for_loop(const tensor<T>& aten,
                         const tensor<T>& bten,
                         tensor<T>& cten,
                         const std::vector<size_t>& a_dims,
                         const std::vector<size_t>& b_dims,
                         float palpha0,
                         float palpha1,
                         float pbeta,
                         int recurr_aoffset,
                         int recurr_boffset,
                         int recurr_coffset,
                         int dim,
                         int AtenOffset,
                         int BtenOffset,
                         int CtenOffset)
    {

        int astride = aten.desc.GetStrides()[dim];
        int bstride = bten.desc.GetStrides()[dim];
        int cstride = cten.desc.GetStrides()[dim];

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
#if(MIO_OPS_DEBUG)
                printf("c[%lu](%f) = a[%lu](%f) + b[%lu](%f)\n",
                       cindex + CtenOffset,
                       cten[cindex + CtenOffset],
                       aindex + AtenOffset,
                       aten[aindex + AtenOffset],
                       bindex + Boffset,
                       bten[bindex + Boffset]);
#endif
                cten[cindex + CtenOffset] =
                    // add_elem(aten[aindex + AtenOffset] * palpha0, bten[bindex + BtenOffset] *
                    // palpha1) +
                    // max_elem(aten[aindex + AtenOffset] * palpha0, bten[bindex + BtenOffset] *
                    // palpha1) +
                    // min_elem(aten[aindex + AtenOffset] * palpha0, bten[bindex + BtenOffset] *
                    // palpha1) +
                    mul_elem(aten[aindex + AtenOffset] * palpha0,
                             bten[bindex + BtenOffset] * palpha1) +
                    pbeta * cten[cindex + Coffset];
            }
            if(dim < (a_dims.size() - 1))
            {

                tensor_for_loop(aten,
                                bten,
                                cten,
                                a_dims,
                                b_dims,
                                palpha0,
                                palpha1,
                                pbeta,
                                aindex,
                                bindex,
                                cindex,
                                dim + 1,
                                AtenOffset,
                                BtenOffset,
                                CtenOffset);
            }
        }
        return;
    }

    tensor<T> cpu()
    {
        std::fill(c.begin(), c.end(), 1);
        auto clens    = c.desc.GetLengths();
        auto blens    = b.desc.GetLengths();
        auto bstrides = b.desc.GetStrides();
        auto cstrides = c.desc.GetStrides();

        tensor_for_loop(
            a, b, c, clens, blens, alpha0, alpha1, beta, 0, 0, 0, 0, Aoffset, Boffset, Coffset);

#if(MIO_OPS_DEBUG)
        for(int i = 0; i < c.desc.GetElementSize(); i++)
            printf("CPU_C[%d]: %f\n", i, c.data[i]);
#endif
        return c;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();

        // return c;
        std::fill(c.begin(), c.end(), 1);

        auto c_dev = handle.Write(c.data);
        auto a_dev = handle.Write(a.data);
        auto b_dev = handle.Write(b.data);

        miopen::OpTensor(handle,
                         // miopenTensorOpAdd,
                         // miopenTensorOpMax,
                         // miopenTensorOpMin,
                         miopenTensorOpMul,
                         &alpha0,
                         a.desc,
                         a_dev.get(),
                         &alpha1,
                         b.desc,
                         b_dev.get(),
                         &beta,
                         c.desc,
                         c_dev.get(),
                         Aoffset,
                         Boffset,
                         Coffset);

        c.data = handle.Read<T>(c_dev, c.data.size());

#if(MIO_OPS_DEBUG)
        handle.Finish();
        auto clens    = c.desc.GetLengths();
        auto cstrides = c.desc.GetStrides();
        for(int i = 0; i < c.desc.GetElementSize(); i++)
            printf("GPU_C[%d]: %f\n", i, c.data[i]);
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

    tensor<T> super_a;
    tensor<T> super_b;
    tensor<T> super_c;

    // tensor<T> a;
    // tensor<T> b;
    // tensor<T> c;

    tensor_ops_driver()
    {

        add(super_a, "super_a", generate_tensor(get_super_tensor(), {40, 10, 8, 20, 4}));
        add(super_b, "super_b", generate_tensor(get_super_tensor(), {40, 10, 8, 20, 4}));
        add(super_c, "super_c", generate_tensor(get_super_tensor(), {40, 10, 8, 20, 4}));
    }

    std::set<std::vector<int>> get_super_tensor()
    {
        std::vector<std::vector<int>> a_dims{
            {40, 10, 8, 20, 4},
        };

        return (std::set<std::vector<int>>(a_dims.begin(), a_dims.end()));
    }

    std::vector<tensor<T>> get_subtensors()
    {
        std::vector<tensor<T>> tensorList;

        unsigned int num_tensor_per_dims_size = 8;

        std::vector<std::vector<int>> lens{
            {32, 8, 8, 16, 4},
            {32, 4, 4, 8, 2},
            {16, 8, 4, 16, 2},
            {16, 2, 8, 4, 4},
            {8, 2, 8, 4, 4},
            {8, 8, 8, 4, 4},
            {4, 2, 4, 8, 2},
            {1, 8, 4, 8, 2}, // 5d
            {8, 1, 16, 4},
            {4, 2, 8, 2},
            {8, 4, 16, 2},
            {2, 8, 4, 4},
            {2, 2, 8, 4},
            {8, 8, 4, 4},
            {1, 4, 8, 2},
            {1, 1, 8, 2}, // 4d
            {8, 16, 4},
            {4, 8, 2},
            {4, 16, 2},
            {8, 1, 4},
            {8, 2, 4},
            {8, 4, 4},
            {4, 8, 2},
            {1, 8, 2}, // 3d
            {16, 4},
            {8, 2},
            {16, 2},
            {4, 4},
            {2, 4},
            {4, 1},
            {8, 2},
            {1, 4}, // 2d
            {32},
            {10},
            {16},
            {8},
            {6},
            {4},
            {3},
            {1}, // 1d
        };

        std::vector<std::vector<int>> strides{
            {6400, 640, 80, 4, 1}, {640, 80, 4, 1}, {80, 4, 1}, {4, 1}, {1}};

        for(int i = 0; i < lens.size(); i++)
        {
            tensorList.push_back(
                make_tensor<T, int>(super_a, lens[i], strides[i / num_tensor_per_dims_size]));
        }

        return tensorList;
    }

    void run()
    {
        std::vector<tensor<T>> aTensorList = get_subtensors();
        std::vector<tensor<T>> bTensorList = get_subtensors();
        std::vector<tensor<T>> cTensorList = get_subtensors();

        std::vector<std::vector<size_t>> offsetList   = {{32, 16, 32}, {16, 32, 16}};
        std::vector<std::vector<float>> alphaBetaList = {
            {1, 1, 1}, {-1, 1, -1}, {0, 0, 0}, {-1.5, 0.5, 2}};

        for(int i = 0; i < aTensorList.size(); i++)
        {
            if(aTensorList[i].desc.GetSize() == bTensorList[i].desc.GetSize())
            {
                for(int j = 0; j < offsetList.size(); j++)
                {
                    for(int k = 0; k < alphaBetaList.size(); k++)
                    {
                        printf("%d %d %d\n", i, j, k);
                        verify(verify_tensor_ops<T>{aTensorList[i],
                                                    bTensorList[i],
                                                    cTensorList[i],
                                                    offsetList[j],
                                                    alphaBetaList[k][0],
                                                    alphaBetaList[k][1],
                                                    alphaBetaList[k][2]});
                    }
                }
            }
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_ops_driver<float>>(argc, argv); }
