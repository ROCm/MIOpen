/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "../driver/tensor_driver.hpp"
#include "../src/kernels/MIOpenReduceExtreme.hpp"
#include "get_handle.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <miopen/reduceextreme.hpp>
#include <miopen/miopen.h>
#include <vector>

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <typename T, ReduceExtremeOp_t op>
void cpu_extreme_forward(tensor<T> input,
                         tensor<T>& ref_output,
                         tensor<int32_t>& ref_indice,
                         int32_t dim,
                         miopenReduceExtremeOp_t reduceExtremeOp)
{
    auto input_dims = input.desc.GetLengths();
    std::vector<std::size_t> output_dims;

    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX) ||
       reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
        output_dims = ref_output.desc.GetLengths();
    else
        output_dims = ref_indice.desc.GetLengths();

    auto reduce_size = input_dims[dim];
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    auto inner_size = std::accumulate(
        input_dims.begin() + dim + 1, input_dims.end(), 1ULL, std::multiplies<uint64_t>());

    par_ford(output_numel)([&](size_t o) {
        size_t input_idx = (o / inner_size) * inner_size * reduce_size + o % inner_size;

        int32_t extreme_idx = 0;
        T extreme           = input[input_idx];

        ford(reduce_size)([&](size_t i) {
            T val = input[input_idx];
            reduce_func<T, int32_t, op>{}.calculate(extreme, val, extreme_idx, i);
            input_idx += inner_size;
        });
        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX) ||
           reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
            ref_output[o] = extreme;
        ref_indice[o] = extreme_idx;
    });
}

template <typename T>
void cpu_aminmax_backward(tensor<T> input,
                          tensor<T>& input_grad,
                          tensor<T> output,
                          tensor<T> output_grad,
                          tensor<int32_t> count,
                          tensor<int32_t> dims)
{
    auto input_tv       = miopen::get_inner_expanded_tv<5>(input.desc);
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(input_grad.desc);
    auto output_tv      = miopen::get_inner_expanded_tv<5>(output.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto count_tv       = miopen::get_inner_expanded_tv<5>(count.desc);

    auto N = input.desc.GetElementSize();
    par_ford(N)([&](size_t gid) {
        uint64_t oN, oC, oD, oH, oW;
        tensor_layout_t<5> tensor_layout(input_tv, gid);

        oN = dims[0] ? 0 : tensor_layout.layout[0];
        oC = dims[1] ? 0 : tensor_layout.layout[1];
        oD = dims[2] ? 0 : tensor_layout.layout[2];
        oH = dims[3] ? 0 : tensor_layout.layout[3];
        oW = dims[4] ? 0 : tensor_layout.layout[4];

        int32_t minmax_count = count[count_tv.get_tensor_view_idx({oN, oC, oD, oH, oW})];

        double temp =
            (static_cast<double>(input[input_tv.get_tensor_view_idx(tensor_layout)]) ==
             static_cast<double>(output[output_tv.get_tensor_view_idx({oN, oC, oD, oH, oW})]))
                ? static_cast<double>(
                      output_grad[output_grad_tv.get_tensor_view_idx({oN, oC, oD, oH, oW})]) /
                      minmax_count
                : 0;
        input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(temp);
    });
}

template <class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct ReduceExtremeTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t dim;
    miopenReduceExtremeOp_t reduceExtremeOp;
    friend std::ostream& operator<<(std::ostream& os, const ReduceExtremeTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " dim:" << tc.dim
                  << " reduceExtremeOp:" << tc.reduceExtremeOp;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

inline std::vector<ReduceExtremeTestCase>
ReduceExtremeTestConfigs(miopenReduceExtremeOp_t reduceExtremeOp)
{ // n c d h w dim
    // clang-format off
    if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
    {
        return {
            {  2,  0, 0, 0, 242991, 0 , MIOPEN_REDUCE_EXTREME_MIN},   //maskrcnn
            {  4,  0, 0, 0,   2004, 0 , MIOPEN_REDUCE_EXTREME_MIN},
            { 34,  0, 0, 0,   3234, 0 , MIOPEN_REDUCE_EXTREME_MIN},   //ssdlite
            { 57,  0, 0, 0,   3234, 0 , MIOPEN_REDUCE_EXTREME_MIN}
        };
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX)
    {
        return {
            {  2,  0, 0, 0, 242991, 0 , MIOPEN_REDUCE_EXTREME_MAX},   //maskrcnn
            {  4,  0, 0, 0,   2004, 0 , MIOPEN_REDUCE_EXTREME_MAX},
            { 34,  0, 0, 0,   3234, 0 , MIOPEN_REDUCE_EXTREME_MAX},   //ssdlite
            { 57,  0, 0, 0,   3234, 0 , MIOPEN_REDUCE_EXTREME_MAX}
        };
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMIN)
    {
        return {
            { 16, 21,   0, 513, 513, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},   //deeplabv3m
            { 24, 21,   0, 513, 513, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},   //deeplabv3r
            { 64, 21,   0, 230, 333, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},   //fcn_resnet_lraspp
            { 64, 21,   0, 215, 288, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},
            { 1,  21,   0, 333, 500, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},   //stdc
            { 1,  21,   0, 375, 500, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},
            { 15, 21,   0, 256, 256, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},   //unet
            { 22, 21,   0, 256, 256, 1 , MIOPEN_REDUCE_EXTREME_ARGMIN},
            { 21, 412,  0,   0, 500, 0 , MIOPEN_REDUCE_EXTREME_ARGMIN},
            { 21, 333,  0,   0, 500, 0 , MIOPEN_REDUCE_EXTREME_ARGMIN}
        };
    }
    else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMAX)
    {
        return {
            { 16, 21,   0, 513, 513, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},   //deeplabv3m
            { 24, 21,   0, 513, 513, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},   //deeplabv3r
            { 64, 21,   0, 230, 333, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},   //fcn_resnet_lraspp
            { 64, 21,   0, 215, 288, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},
            { 1,  21,   0, 333, 500, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},   //stdc
            { 1,  21,   0, 375, 500, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},
            { 15, 21,   0, 256, 256, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},   //unet
            { 22, 21,   0, 256, 256, 1 , MIOPEN_REDUCE_EXTREME_ARGMAX},
            { 21, 412,  0,   0, 500, 0 , MIOPEN_REDUCE_EXTREME_ARGMAX},
            { 21, 333,  0,   0, 500, 0 , MIOPEN_REDUCE_EXTREME_ARGMAX}
        };
    }
    return {};
    // clang-format on
}

struct ReduceExtremeTestCaseBwd
{
    std::vector<size_t> input_dim;
    std::vector<int32_t> dim;
    bool keep_dim;
    miopenReduceExtremeOp_t reduceExtremeOp;
    friend std::ostream& operator<<(std::ostream& os, const ReduceExtremeTestCaseBwd& tc)
    {
        return os << " input_dim:" << tc.input_dim << " dim:" << tc.dim
                  << " keep_dim:" << tc.keep_dim << " reduceExtremeOp:" << tc.reduceExtremeOp;
    }
};

inline std::vector<ReduceExtremeTestCaseBwd> ReduceExtremeTestConfigsBwd()
{
    // clang-format off
    return {
        { {16, 21,513, 513, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMIN},   //deeplabv3m
        { {24, 21,513, 513, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMIN},   //deeplabv3r
        { {64, 21,230, 333, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMIN},   //fcn_resnet_lraspp
        { {64, 21,215, 288, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMIN},
        { {1,  21,333, 500, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMIN},   //stdc
        { {1,  21,375, 500, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMIN},
        { {15, 21,256, 256, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMIN},   //unet
        { {22, 21,256, 256, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMIN},
        { {21, 412,500}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMIN},
        { {21, 333,500}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMIN},
        { {16, 21,513, 513, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMAX},   //deeplabv3m
        { {24, 21,513, 513, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMAX},   //deeplabv3r
        { {64, 21,230, 333, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMAX},   //fcn_resnet_lraspp
        { {64, 21,215, 288, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMAX},
        { {1,  21,333, 500, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMAX},   //stdc
        { {1,  21,375, 500, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMAX},
        { {15, 21,256, 256, 1}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMAX},   //unet
        { {22, 21,256, 256, 1}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMAX},
        { {21, 412,500}, {0,1}, true, MIOPEN_REDUCE_EXTREME_AMAX},
        { {21, 333,500}, {0,1}, false, MIOPEN_REDUCE_EXTREME_AMAX}
    };
    // clang-format on
}

template <typename T = float>
struct ReduceExtremeTest : public ::testing::TestWithParam<ReduceExtremeTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        reduceextreme_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim             = reduceextreme_config.dim;
        reduceExtremeOp = reduceextreme_config.reduceExtremeOp;

        auto in_dims = reduceextreme_config.GetInput();

        input = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;

        for(int32_t i = 0; i < in_dims.size(); ++i)
        {
            if(i != dim)
            {
                out_dims.push_back(in_dims[i]);
            }
        }

        indice = tensor<int32_t>{out_dims};
        std::fill(indice.begin(), indice.end(), std::numeric_limits<int32_t>::quiet_NaN());

        ref_indice = tensor<int32_t>{out_dims};
        std::fill(ref_indice.begin(), ref_indice.end(), std::numeric_limits<int32_t>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        indice_dev = handle.Write(indice.data);

        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
           (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
        {
            output = tensor<T>{out_dims};
            std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

            ref_output = tensor<T>{out_dims};
            std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

            output_dev = handle.Write(output.data);
        }
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMIN)
        {
            cpu_extreme_forward<T, ReduceExtremeOp_t::Argmin>(
                input, ref_output, ref_indice, dim, reduceExtremeOp);
        }
        else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMAX)
        {
            cpu_extreme_forward<T, ReduceExtremeOp_t::Argmax>(
                input, ref_output, ref_indice, dim, reduceExtremeOp);
        }
        else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN)
        {
            cpu_extreme_forward<T, ReduceExtremeOp_t::Min>(
                input, ref_output, ref_indice, dim, reduceExtremeOp);
        }
        else if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX)
        {
            cpu_extreme_forward<T, ReduceExtremeOp_t::Max>(
                input, ref_output, ref_indice, dim, reduceExtremeOp);
        }

        miopenStatus_t status;
        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
           (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
        {
            status = miopen::ReduceExtremeForward(handle,
                                                  input.desc,
                                                  input_dev.get(),
                                                  output.desc,
                                                  output_dev.get(),
                                                  indice.desc,
                                                  indice_dev.get(),
                                                  dim,
                                                  reduceExtremeOp);
        }
        else
        {
            status = miopen::ReduceExtremeForward(handle,
                                                  input.desc,
                                                  input_dev.get(),
                                                  indice.desc,
                                                  indice_dev.get(),
                                                  dim,
                                                  reduceExtremeOp);
        }

        EXPECT_EQ(status, miopenStatusSuccess);

        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
           (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
        {
            output.data = handle.Read<T>(output_dev, output.data.size());
        }
        indice.data = handle.Read<int32_t>(indice_dev, indice.data.size());
    }

    void Verify()
    {
        if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
           (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX))
        {
            auto error = miopen::rms_range(ref_output, output);

            EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
            EXPECT_TRUE(std::abs(static_cast<float>(error)) == 0.0f)
                << "Error output beyond tolerance Error:" << error;
        }

        auto error_idx = miopen::mismatch_idx(ref_indice, indice, compare_equal<int32_t>);

        EXPECT_TRUE(miopen::range_distance(ref_indice) == miopen::range_distance(indice));
        EXPECT_TRUE(error_idx >= miopen::range_distance(ref_indice))
            << "Error Indice does not equal at " << error_idx << std::endl;
    }
    ReduceExtremeTestCase reduceextreme_config;

    tensor<T> input;
    tensor<T> output;
    tensor<int32_t> indice;

    tensor<T> ref_output;
    tensor<int32_t> ref_indice;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr indice_dev;

    int32_t dim;
    miopenReduceExtremeOp_t reduceExtremeOp;
};

template <typename T = float>
struct ReduceExtremeTestBwd : public ::testing::TestWithParam<ReduceExtremeTestCaseBwd>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        reduceextreme_config = GetParam();
        auto in_dims         = reduceextreme_config.input_dim;
        auto gen_value_input = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 100);
        };

        auto dim_sort = reduceextreme_config.dim;
        std::sort(dim_sort.begin(), dim_sort.end());
        // one-hot dim
        std::vector<int32_t> dim_onehot = std::vector<int32_t>(in_dims.size(), 0);
        for(auto&& d : dim_sort)
        {
            dim_onehot[d] = 1;
        }
        dim      = tensor<int32_t>{dim_onehot.size()};
        dim.data = dim_onehot;

        reduceExtremeOp = reduceextreme_config.reduceExtremeOp;
        keep_dim        = reduceextreme_config.keep_dim;

        input = tensor<T>{in_dims}.generate(gen_value_input);

        std::vector<size_t> out_dims;

        if(keep_dim)
        {
            for(int32_t i = 0; i < in_dims.size(); ++i)
            {
                if(dim[i] == 0)
                {
                    out_dims.push_back(in_dims[i]);
                }
                else if(dim[i] == 1)
                {
                    out_dims.push_back(1);
                }
            }
        }
        else
        {
            for(int32_t i = 0; i < in_dims.size(); ++i)
            {
                if(dim[i] == 0)
                {
                    out_dims.push_back(in_dims[i]);
                }
            }
        }
        if(out_dims.empty())
        {
            out_dims.push_back(1);
        }

        size_t num_reduce = 1;
        for(int32_t i = 0; i < in_dims.size(); ++i)
        {
            if(dim[i] == 1)
            {
                num_reduce *= in_dims[i];
            }
        }
        auto gen_value_count = [num_reduce](auto...) {
            return 1 + prng::gen_descreet_unsigned<int32_t>(1, num_reduce - 1);
        };
        count = tensor<int32_t>{out_dims}.generate(gen_value_count);

        auto gen_value_output = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 10);
        };
        output                     = tensor<T>{out_dims}.generate(gen_value_output);
        auto gen_value_output_grad = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 5);
        };
        output_grad = tensor<T>{out_dims}.generate(gen_value_output_grad);

        input_grad = tensor<T>{in_dims};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{in_dims};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev       = handle.Write(input.data);
        count_dev       = handle.Write(count.data);
        output_dev      = handle.Write(output.data);
        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
        dim_dev         = handle.Write(dim.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMIN ||
           reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMAX)
        {
            cpu_aminmax_backward<T>(input, ref_input_grad, output, output_grad, count, dim);
        }
        // leave the rest of the cases for future backward implementation

        if(reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMIN ||
           reduceExtremeOp == MIOPEN_REDUCE_EXTREME_AMAX)
        {
            status = miopen::ReduceExtremeBackward(handle,
                                                   input.desc,
                                                   input_dev.get(),
                                                   input_grad.desc,
                                                   input_grad_dev.get(),
                                                   output.desc,
                                                   output_dev.get(),
                                                   output.desc,
                                                   output_grad_dev.get(),
                                                   count.desc,
                                                   count_dev.get(),
                                                   dim.desc,
                                                   dim_dev.get(),
                                                   reduceExtremeOp);
        }
        // leave the rest of the cases for future backward implementation

        ASSERT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);

        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold * 10) << "Error backward Output beyond 10xthreshold : " << error
                                         << " Tolerance: " << threshold * 10;
    }
    ReduceExtremeTestCaseBwd reduceextreme_config;

    tensor<T> input;
    tensor<T> input_grad;
    tensor<T> output;
    tensor<T> output_grad;
    tensor<int32_t> count;
    tensor<int32_t> dim;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr count_dev;
    miopen::Allocator::ManageDataPtr dim_dev;

    miopenReduceExtremeOp_t reduceExtremeOp;
    bool keep_dim;
};
