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
struct verify_tensor_ops
{
    tensor<T> a;
    tensor<T> b;
    tensor<T> c;

    int Aoffset;
    int Boffset;
    int Coffset;

    float alpha0;
    float alpha1;
    float beta;

    verify_tensor_ops(const tensor<T>& pa,
                      const tensor<T>& pb,
                      const tensor<T>& pc,
                      const std::vector<int>& offsets,
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

    static T add_elem(T aelem, T belem) { return aelem + belem; }
    static T mul_elem(T aelem, T belem) { return aelem * belem; }
    static T max_elem(T aelem, T belem) { return ((aelem > belem) ? aelem : belem); }
    static T min_elem(T aelem, T belem) { return ((aelem < belem) ? aelem : belem); }

    static void tensor_for_loop(const tensor<T>& aten,
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
                    mul_elem(T(aten[aindex + AtenOffset] * palpha0),
                             T(bten[bindex + BtenOffset] * palpha1)) +
                    pbeta * cten[cindex + CtenOffset];
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
    }

    tensor<T> cpu() const
    {
        auto r = c;
        std::fill(r.begin(), r.end(), 1);
        auto clens = r.desc.GetLengths();
        auto blens = b.desc.GetLengths();

        tensor_for_loop(
            a, b, r, clens, blens, alpha0, alpha1, beta, 0, 0, 0, 0, Aoffset, Boffset, Coffset);

#if(MIO_OPS_DEBUG)
        for(int i = 0; i < r.desc.GetElementSize(); i++)
            printf("CPU_C[%d]: %f\n", i, r.data[i + Coffset]);
#endif
        return r;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto r        = c;

        // return c;
        std::fill(r.begin(), r.end(), 1);

        auto c_dev = handle.Write(r.data);
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
                         r.desc,
                         c_dev.get(),
                         Aoffset,
                         Boffset,
                         Coffset);

        r.data = handle.Read<T>(c_dev, r.data.size());

#if(MIO_OPS_DEBUG)
        handle.Finish();
        auto clens    = r.desc.GetLengths();
        auto cstrides = r.desc.GetStrides();
        for(int i = 0; i < r.desc.GetElementSize(); i++)
            printf("GPU_C[%d]: %f\n", i, c.data[i + Coffset]);
#endif
        return r;
    }

    void fail(int = 0) const
    {
        std::cout << "TensorOp: " << std::endl;
        std::cout << "A tensor: " << a.desc.ToString() << std::endl;
        std::cout << "B tensor: " << b.desc.ToString() << std::endl;
        std::cout << "C tensor: " << a.desc.ToString() << std::endl;
        std::cout << "Offsets: " << Aoffset << "," << Boffset << "," << Coffset << std::endl;
    }
};

template <class T>
struct tensor_ops_driver : test_driver
{

    tensor<T> super_a;
    tensor<T> super_b;
    tensor<T> super_c;

    std::vector<int> tensorlens_ac;
    std::vector<int> tensorlens_b;
    std::vector<int> offsets;
    std::vector<float> alphabeta;
    bool packed = false;

    std::vector<std::vector<int>> get_sub_tensor_a()
    {
        return {{32, 16, 8, 4, 4}, {16, 20, 16, 8}, {20, 16, 8}, {1, 16, 8}, {16, 8}, {8}};
    }

    std::vector<std::vector<int>> get_sub_tensor_b()
    {
        return {{32, 16, 8, 4, 4},
                {32, 16, 1, 1, 1},
                {1, 16, 8, 1, 1},
                {1, 1, 8, 4, 1},
                {16, 20, 16, 8},
                {16, 20, 16, 1},
                {16, 20, 1, 1},
                {16, 1, 1, 1},
                {1, 20, 16, 8},
                {1, 20, 16, 1},
                {1, 20, 1, 1},
                {1, 1, 16, 8},
                {1, 1, 1, 8},
                {20, 16, 8},
                {20, 16, 1},
                {1, 16, 8},
                {1, 16, 1},
                {20, 1, 1},
                {16, 8},
                {16, 1},
                {1, 8},
                {8},
                {1}};
    }

    tensor_ops_driver()
    {
        disabled_cache         = true;
        std::vector<int> alens = {{32, 16, 20, 16, 8}};
        std::vector<int> blens = {{32, 16, 20, 16, 8}};
        std::vector<int> clens = {{32, 16, 20, 16, 8}};

        unsigned long max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        super_a = tensor<T>{alens}.generate(tensor_elem_gen_integer{max_value});
        super_b = tensor<T>{blens}.generate(tensor_elem_gen_integer{max_value});
        super_c = tensor<T>{clens}.generate(tensor_elem_gen_integer{max_value});

        std::vector<std::vector<int>> get_offsets     = {{64, 32, 16}, {32, 16, 32}, {32, 16, 32}};
        std::vector<std::vector<float>> get_alphabeta = {{1, 1, 0}, {-1, 1, 1}, {1.0, 0.5, 0.3}};

        add(tensorlens_ac, "a", generate_data(get_sub_tensor_a()));
        add(tensorlens_b, "b", generate_data(get_sub_tensor_b()));
        add(offsets, "offsets", generate_data(get_offsets));
        add(alphabeta, "alpha-beta", generate_data(get_alphabeta));
        add(packed, "packed", generate_data({false, true}));
    }

    tensor<T> get_subtensors(tensor<T>& super_tensor, std::vector<int>& lens, bool isPacked)
    {
        if(!isPacked)
        {
            std::vector<size_t> superStrides = super_tensor.desc.GetStrides();
            std::vector<int> strides(superStrides.begin() + (5 - lens.size()), superStrides.end());
            tensor<T> t = tensor<T>{lens, strides};
            t.data      = super_tensor.data;
            return t;
        }
        else
        {
            tensor<T> t = tensor<T>{lens};
            t.data      = super_tensor.data;
            return t;
        }
    }

    void run()
    {
        if(tensorlens_ac.size() == tensorlens_b.size() && tensorlens_ac >= tensorlens_b)
        {
            tensor<T> aTensor = get_subtensors(super_a, tensorlens_ac, packed);
            tensor<T> bTensor = get_subtensors(super_b, tensorlens_b, packed);
            tensor<T> cTensor = get_subtensors(super_c, tensorlens_ac, packed);

            if(packed)
                offsets = {0, 0, 0, 0, 0};

            verify(verify_tensor_ops<T>{
                aTensor, bTensor, cTensor, offsets, alphabeta[0], alphabeta[1], alphabeta[2]});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_ops_driver>(argc, argv); }
