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

    bool no_validate;

    verify_tensor_ops(tensor<T>&& pa,
                      tensor<T>&& pb,
                      tensor<T>&& pc,
                      const std::vector<int>& offsets,
                      const std::vector<float>& alphabeta,
                      bool no_validate_param)
        : a{std::move(pa)},
          b{std::move(pb)},
          c{std::move(pc)},
          Aoffset{offsets[0]},
          Boffset{offsets[1]},
          Coffset{offsets[2]},
          alpha0{alphabeta[0]},
          alpha1{alphabeta[1]},
          beta{alphabeta[2]},
          no_validate{no_validate_param}
    {
    }

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
                       bindex + BtenOffset,
                       bten[bindex + BtenOffset]);
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
        auto r     = c;
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

        if(not no_validate)
        {
            auto r = c;
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

        return c;
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
    std::vector<int> tensorlens_ac;
    std::vector<int> tensorlens_b;
    std::vector<int> offsets;
    std::vector<int> stride_a;
    std::vector<int> stride_b;
    std::vector<int> stride_c;
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
        disabled_cache = true;

        std::vector<std::vector<int>> get_offsets = {
            {0, 0, 0}, {64, 32, 16}, {32, 16, 32}, {32, 16, 32}};
        std::vector<std::vector<float>> get_alphabeta = {{1, 1, 0}, {-1, 1, 1}, {1.0, 0.5, 0.3}};
        std::vector<std::vector<int>> get_strides = {{8 * 16 * 20 * 16, 8 * 16 * 20, 8 * 16, 8, 1}};

        add(tensorlens_ac, "a", generate_data(get_sub_tensor_a()));
        add(tensorlens_b, "b", generate_data(get_sub_tensor_b()));
        add(stride_a, "sa", generate_data(get_strides));
        add(stride_b, "sb", generate_data(get_strides));
        add(stride_c, "sc", generate_data(get_strides));
        add(offsets, "offsets", generate_data(get_offsets));
        add(alphabeta, "alpha-beta", generate_data(get_alphabeta));
        add(packed, "packed", generate_data({false, true}));
    }

    tensor<T> get_subtensors(const std::vector<int>& lens,
                             const std::vector<int>& strides,
                             int offset,
                             bool isPacked)
    {
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        if(!isPacked)
        {
            std::vector<int> real_strides(strides.begin() + (strides.size() - lens.size()),
                                          strides.end());
            auto r = tensor<T>{lens, real_strides}.generate(tensor_elem_gen_integer{max_value});
            r.data.resize(r.data.size() + offset);
            return r;
        }
        else
        {
            return tensor<T>{lens}.generate(tensor_elem_gen_integer{max_value});
        }
    }

    void run()
    {
        if(tensorlens_ac.size() == tensorlens_b.size())
        {
            for(size_t idx = 0; idx < tensorlens_b.size(); ++idx)
            {
                if((tensorlens_b[idx] != 1) && (tensorlens_ac[idx] != tensorlens_b[idx]))
                    return;
            }

            std::vector<int> final_offsets{0, 0, 0};
            if(!packed)
            {
                if(std::any_of(offsets.begin(), offsets.end(), [](int o) { return o < 0; }))
                    return;

                final_offsets = offsets;
            }

            auto checkStride = [p = packed](const std::vector<int>& lens,
                                            const std::vector<int>& strides) {
                if(p)
                    return true;

                if(lens.size() > strides.size())
                    return false;

                // only sparsed case allowed, since all the kernels do not support the last
                // dimension strides
                if(strides.back() == 1)
                {
                    auto packedStrides =
                        miopen::TensorDescriptor(miopen_type<T>{}, lens).GetStrides();
                    return std::equal(packedStrides.rbegin(),
                                      packedStrides.rend(),
                                      strides.rbegin(),
                                      [](int ps, int s) { return s >= ps; });
                }

                // currently tensor operations do not support non-one stride in the last dimention.
                return false;
            };

            if(!checkStride(tensorlens_ac, stride_a))
                return;
            if(!checkStride(tensorlens_b, stride_b))
                return;
            if(!checkStride(tensorlens_ac, stride_c))
                return;

            verify(verify_tensor_ops<T>{get_subtensors(tensorlens_ac, stride_a, offsets[0], packed),
                                        get_subtensors(tensorlens_b, stride_b, offsets[1], packed),
                                        get_subtensors(tensorlens_ac, stride_c, offsets[2], packed),
                                        final_offsets,
                                        alphabeta,
                                        no_validate});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<tensor_ops_driver>(argc, argv); }
