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
struct tensor_copy_base
{
    tensor<T> a;
    tensor<T> c;
    int aoffset;
    int coffset;

    void fail(float = 0)
    {
        std::cout << "A tensor: " << a.desc.ToString() << std::endl;
        std::cout << "C tensor: " << c.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_tensor_copy : tensor_copy_base<T>
{
    using tensor_copy_base<T>::a;
    using tensor_copy_base<T>::c;
    int aoffset;
    int coffset;

    verify_tensor_copy(const tensor<T>& pa, const tensor<T>& pc)
    {
        a = pa;
        c = pc;
    }

    verify_tensor_copy(const tensor<T>& pa, const tensor<T>& pc, const std::vector<T>& dims)
    {
        a = pa(dims);
        c = pc(dims);
    }

    void tensor_for_loop(const tensor<T>& aten,
                         tensor<T>& cten,
                         const std::vector<size_t>& a_dims,
                         const std::vector<size_t>& c_dims,
                         int aoffsetIndex,
                         int coffsetIndex,
                         int dim)
    {
        aoffset = coffset = 0;
        int astride       = aten.desc.GetStrides()[dim];
        int cstride       = cten.desc.GetStrides()[dim];
        for(int idx = 0; idx < a_dims[dim]; idx++)
        {
            size_t aindex = aoffsetIndex + astride * idx;
            size_t cindex = coffsetIndex + cstride * idx;

            if(dim < (a_dims.size() - 1))
            {
                tensor_for_loop(aten, cten, a_dims, c_dims, aindex, cindex, dim + 1);
            }

            if(cindex < cten.desc.GetElementSpace() && aindex < aten.desc.GetElementSpace())
            {
                cten[cindex + coffset] = aten[aindex + aoffset];
            }
        }
        return;
    }

    tensor<T> cpu()
    {

        // printf("In CopyTensor CPU test.\n");
        std::fill(c.begin(), c.end(), 0);

        auto alens = a.desc.GetLengths();
        auto clens = c.desc.GetLengths();

        auto astrides = a.desc.GetStrides();
        auto cstrides = c.desc.GetStrides();

        // printf("Running CopyTensor CPU.\n");
        tensor_for_loop(a, c, alens, clens, 0, 0, 0);

#if(MIO_OPS_DEBUG)
        realid = 0;
        for(int i = 0; i < c.desc.GetLengths()[0]; i++)
        {
            for(int j = 0; j < c.desc.GetLengths()[1]; j++)
            {
                id = i * c.desc.GetStrides()[0] + j;
                if(realid < elemSize)
                    std::cout << "C_CPU[" << realid++ << "]: " << c[id] << std::endl;
            }
        }
#endif
        return c;
    }

    tensor<T> gpu()
    {
        // printf("In CopyTensor GPU test.\n");
        auto&& handle = get_handle();

        std::fill(c.begin(), c.end(), 0);

        auto c_dev = handle.Write(c.data);
        auto a_dev = handle.Write(a.data);
        aoffset = coffset = 0;

        // printf("Running CopyTensor GPU.\n");
        miopen::CopyTensor(handle, a.desc, a_dev.get(), c.desc, c_dev.get(), aoffset, coffset);

        // handle.Finish();
        c.data = handle.Read<T>(c_dev, c.data.size());

#if(MIO_OPS_DEBUG)
        handle.Finish();
        realid = 0;
        for(int i = 0; i < c.desc.GetLengths()[0]; i++)
        {
            for(int j = 0; j < c.desc.GetLengths()[1]; j++)
            {
                id = i * c.desc.GetStrides()[0] + j;
                if(realid < elemSize)
                    std::cout << "C_GPU[" << realid++ << "]: " << c[id] << std::endl;
            }
        }
#endif
        return c;
    }

    void fail(float = 0)
    {
        std::cout << "Tensor Copy: " << std::endl;
        this->tensor_copy_base<T>::fail();
    }
};

template <class T>
struct tensor_copy_driver : test_driver
{
    tensor<T> a;
    tensor<T> c;

    tensor_copy_driver()
    {
        std::array<int, 5> alens = {{4, 3, 2, 2, 3}};
        std::array<int, 5> clens = {{4, 3, 2, 2, 3}};

        add(a,
            "a",
            generate_tensor(get_descs_a(),
                            miopen::TensorDescriptor(miopenFloat, alens.data(), alens.size())));
        add(c,
            "c",
            generate_tensor(get_descs_c(),
                            miopen::TensorDescriptor(miopenFloat, clens.data(), clens.size())));
    }

    std::set<miopen::TensorDescriptor> get_descs_a()
    {
        std::vector<miopen::TensorDescriptor> aDescList;

        std::vector<std::vector<int>> lens{
            {32, 8, 16, 16, 8}, {32, 8, 16, 16}, {32, 8, 16}, {32, 8}, {8},
        };

        std::vector<std::vector<int>> dimOffsets{
            {0, 0, 0, 0, 0},
            //{0, 0, 0, 0, 8},
            //{32, 8, 0, 0, 8},
            {32, 8, 0, 16, 0},
            //{32, 8, 16, 0, 8},
            {32, 8, 16, 16, 0},
            //{0, 0, 0, 16, 8},
            {0, 0, 16, 0, 0},
            //{0, 0, 16, 16, 8},
            {0, 8, 0, 16, 0},
            //{0, 8, 16, 0, 8},
            {0, 8, 16, 16, 0},
            {32, 8, 16, 16, 8},
            {0, 0, 0, 0},
            //{0, 8, 0, 0},
            //{0, 0, 0, 16},
            //{0, 8, 16, 16},
            //{32, 8, 0, 0},
            {32, 8, 0, 16},
            {0, 0, 16, 0},
            {0, 0, 16, 16},
            {0, 8, 0, 16},
            {0, 8, 16, 0},
            {32, 8, 16, 0},
            {32, 8, 16, 16},
            {0, 0, 0},
            //{0, 8, 0},
            //{32, 0, 16},
            //{0, 0, 16},
            {32, 0, 0},
            {0, 8, 16},
            {32, 8, 0},
            {32, 8, 16},
            {0, 0},
            {0, 8},
            {32, 0},
            {32, 8},
            {0},
        };

        for(int i = 0; i < lens.size(); i++)
        {
            int tensorLen = lens[i].size();
            std::vector<int> adjLens(tensorLen, 0);
            std::vector<int> strides(tensorLen, 0);
            for(int j = 0; j < dimOffsets.size(); j++)
            {
                if(dimOffsets[j].size() != lens[i].size())
                    continue;
                std::fill(adjLens.begin(), adjLens.end(), 0);
                std::fill(strides.begin(), strides.end(), 0);
                strides.back() = 1;
                bool zeros     = std::all_of(
                    dimOffsets[j].begin(), dimOffsets[j].end(), [](int z) { return z == 0; });
                if(tensorLen > 1 && !zeros)
                {
                    std::transform(lens[i].begin(),
                                   lens[i].end(),
                                   dimOffsets[j].begin(),
                                   adjLens.begin(),
                                   std::plus<size_t>());
                    std::partial_sum(adjLens.rbegin(),
                                     adjLens.rend() - 1,
                                     strides.rbegin() + 1,
                                     std::multiplies<std::size_t>());
                    aDescList.push_back(miopen::TensorDescriptor(
                        miopenFloat, lens[i].data(), strides.data(), tensorLen));
                }
                else
                {
                    aDescList.push_back(
                        miopen::TensorDescriptor(miopenFloat, lens[i].data(), tensorLen));
                }
            }
        }

        return (std::set<miopen::TensorDescriptor>(aDescList.begin(), aDescList.end()));
    }

    std::set<miopen::TensorDescriptor> get_descs_c()
    {
        std::vector<miopen::TensorDescriptor> cDescList;

        std::vector<std::vector<int>> lens{
            {32, 8, 16, 16, 8}, {32, 8, 16, 16}, {32, 8, 16}, {32, 8}, {8},
        };

        std::vector<std::vector<int>> dimOffsets{
            {0, 0, 0, 0, 0},
            //{0, 0, 0, 0, 8},
            {0, 0, 0, 16, 8},
            //{0, 0, 16, 0, 0},
            //{0, 0, 16, 16, 8},
            //{0, 8, 0, 16, 0},
            {0, 8, 16, 0, 8},
            //{0, 8, 16, 16, 0},
            {32, 8, 0, 0, 8},
            {32, 8, 0, 16, 0},
            {32, 8, 16, 0, 8},
            //{32, 8, 16, 16, 0},
            {32, 8, 16, 16, 8},
            {0, 0, 0, 0},
            {0, 8, 0, 0},
            {0, 0, 0, 16},
            //{0, 0, 16, 0},
            //{0, 0, 16, 16},
            {0, 8, 0, 16},
            {0, 8, 16, 0},
            //{0, 8, 16, 16},
            {32, 8, 0, 0},
            {32, 8, 0, 16},
            {32, 8, 16, 0},
            //{32, 8, 16, 16},
            {0, 0, 0},
            {0, 8, 0},
            //{0, 0, 16},
            {32, 0, 0},
            {0, 8, 16},
            //{32, 8, 0},
            //{32, 0, 16},
            {32, 8, 16},
            {0, 0},
            {32, 0},
            {0, 8},
            {32, 8},
            {0},
        };

        for(int i = 0; i < lens.size(); i++)
        {
            int tensorLen = lens[i].size();
            std::vector<int> adjLens(tensorLen, 0);
            std::vector<int> strides(tensorLen, 0);
            for(int j = 0; j < dimOffsets.size(); j++)
            {

                if(dimOffsets[j].size() != lens[i].size())
                    continue;
                std::fill(adjLens.begin(), adjLens.end(), 0);
                std::fill(strides.begin(), strides.end(), 0);
                strides.back() = 1;
                bool zeros     = std::all_of(
                    dimOffsets[j].begin(), dimOffsets[j].end(), [](int z) { return z == 0; });
                if(tensorLen > 1 && !zeros)
                {
                    std::transform(lens[i].begin(),
                                   lens[i].end(),
                                   dimOffsets[j].begin(),
                                   adjLens.begin(),
                                   std::plus<size_t>());
                    std::partial_sum(adjLens.rbegin(),
                                     adjLens.rend() - 1,
                                     strides.rbegin() + 1,
                                     std::multiplies<size_t>());
                    cDescList.push_back(miopen::TensorDescriptor(
                        miopenFloat, lens[i].data(), strides.data(), tensorLen));
                }
                else
                {
                    cDescList.push_back(
                        miopen::TensorDescriptor(miopenFloat, lens[i].data(), tensorLen));
                }
            }
        }
        return (std::set<miopen::TensorDescriptor>(cDescList.begin(), cDescList.end()));
    }

    void run()
    {
        if(a.desc.GetSize() == c.desc.GetSize() && a.desc.GetLengths() == c.desc.GetLengths())
            verify(verify_tensor_copy<T>{a, c});
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_copy_driver<float>>(argc, argv); }
