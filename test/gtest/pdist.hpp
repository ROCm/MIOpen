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
#include <algorithm>
#include <cmath>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"

#include <vector>
#include <verify.hpp>
// #include <algorithm>
// #include <cstddef>
// #include <cstdint>
#include <ostream>
#include <math.h>

#include <miopen/miopen.h>
#include <miopen/pdist.hpp>
#include <gtest/gtest.h>
#include <cpu_pdist.hpp>

struct PdistTestCase
{
    std::vector<size_t> dims;
    double p;

    friend std::ostream& operator<<(std::ostream& os, const PdistTestCase& tc)
    {
        os << "dims: ( ";
        for(auto d : tc.dims)
        {
            os << d << " ";
        }
        os << ") ";
        os << "p: " << tc.p;

        return os;
    }

    std::vector<size_t> GetDims() const { return dims; }
    double GetP() const { return p; }

    PdistTestCase() {}

    PdistTestCase(std::vector<size_t> dims_, double p_) : dims(dims_), p(p_) {}
};

inline std::vector<PdistTestCase> PdistTestConfigs()
{
    return {
        // {{3, 4}, 2.0},
        {{3, 4}, HUGE_VALF},
        // {{3, 4}, 1},
        // TODO: How to test when expected output=tensor([])?
        // PdistTestCase({1, 1}, 0.0),
        // PdistTestCase({1, 1}, 1.0),
        // PdistTestCase({1, 1}, 2.0),
        // PdistTestCase({1, 1}, 5.0),
        // PdistTestCase({1, 1}, HUGE_VAL), // huge val

        PdistTestCase({5, 1}, 0.0),
        PdistTestCase({10, 1}, 1.0),
        PdistTestCase({2, 5}, 2.0),
        PdistTestCase({2, 10}, 5.0),

        PdistTestCase({5, 5}, 0.0),
        PdistTestCase({10, 10}, 0.0),
        PdistTestCase({100, 100}, 0.0),
        PdistTestCase({100, 100}, 1.0),
        PdistTestCase({100, 100}, 2.0),
        PdistTestCase({100, 100}, 5.0),

        // TODO: Fix wrong result for p=inf
        // PdistTestCase({100, 100}, HUGE_VAL),
    };
}

template <typename T>
struct PdistTestBackward : public ::testing::TestWithParam<PdistTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        pdist_config    = GetParam();
        auto input_dims = pdist_config.GetDims();
        // auto p          = config.GetP();
        // p = config.p;
        p      = pdist_config.GetP();
        auto N = input_dims[0];
        // auto M          = input_dims[1];]
        auto output_dim_size = N * (N - 1) / 2;

        if(output_dim_size == 0)
        {
            GTEST_SKIP();
        }
        std::vector<size_t> output_dims({N * (N - 1) / 2});

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto output_gen_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 200);
        };
        // auto doutput

        input = tensor<T>{input_dims}.generate(gen_value);
        // output  = tensor<T>{output_dims}.generate(output_gen_value);
        // tensor([2.0366, 0.4323, 2.4283])
        output = tensor<T>{output_dims};

        // p = 2
        // output[0] = 2.0366;
        // output[1] = 0.4323;
        // output[2] = 2.4283;

        // p = inf
        // 1.5400, 0.3800, 1.9200
        // output[0] = 1.5400;
        // output[1] = 0.3800;
        // output[2] = 1.9200;

        // p = 3.2
        // tensor([1.7241, 0.3903, 2.0729], grad_fn=<PdistBackward0>)
        // output[0] = 1.7241;
        // output[1] = 0.3903;
        // output[2] = 2.0729;

        // p = 1
        // tensor([3.6200, 0.7100, 4.3300], grad_fn=<PdistBackward0>)
        // output[0] = 3.6200;
        // output[1] = 0.7100;
        // output[2] = 4.3300;

        doutput = tensor<T>{output_dims};
        std::fill(doutput.begin(), doutput.end(), static_cast<T>(1));

        // // print input, output, doutput
        // std::cout << "input: ";
        // for(auto i : input)
        // {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "output: ";
        // for(auto i : output)
        // {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "doutput: ";
        // for(auto i : doutput)
        // {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        dinput = tensor<T>{input_dims};
        std::fill(dinput.begin(), dinput.end(), std::numeric_limits<T>::quiet_NaN());
        // std::fill(dinput.begin(), dinput.end(), 0);

        ref_dinput = tensor<T>{input_dims};
        std::fill(ref_dinput.begin(), ref_dinput.end(), std::numeric_limits<T>::quiet_NaN());
        // std::fill(ref_dinput.begin(), ref_dinput.end(), 0);

        ws_sizeInBytes = miopen::GetPdistBackwardWorkspaceSize(
            handle, input.desc, output.desc, doutput.desc, dinput.desc, p);

        // if(ws_sizeInBytes == static_cast<size_t>(-1) || ws_sizeInBytes == 0)
        if(ws_sizeInBytes <= 0)
            GTEST_FAIL() << "Call GetPdistBackwardWorkspaceSize failed!";

        // std::cout << "Hit twice!!";
        workspace = tensor<T>{ws_sizeInBytes / sizeof(T)};
        std::fill(workspace.begin(), workspace.end(), static_cast<T>(0));

        input_dev     = handle.Write(input.data);
        output_dev    = handle.Write(output.data);
        doutput_dev   = handle.Write(doutput.data);
        workspace_dev = handle.Write(workspace.data);
        dinput_dev    = handle.Write(dinput.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run cpu
        cpu_pdist_forward_contiguous<T>(input, output, doutput, ref_dinput, p);

        // Run kernel
        status = miopen::PdistBackward(handle,
                                       workspace_dev.get(),
                                       ws_sizeInBytes,
                                       input.desc,
                                       input_dev.get(),
                                       output.desc,
                                       output_dev.get(),
                                       doutput.desc,
                                       doutput_dev.get(),
                                       dinput.desc,
                                       dinput_dev.get(),
                                       p);

        EXPECT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        dinput.data = handle.Read<T>(dinput_dev, dinput.data.size());
    }

    void Verify()
    {
        double threshold  = std::numeric_limits<T>::epsilon() * 10;
        auto dinput_error = miopen::rms_range(ref_dinput, dinput);

        // // print ref_dinput
        // std::cout << "ref_dinput: ";
        // for(auto i : ref_dinput)
        // {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        // // print dinput
        // std::cout << "dinput: ";
        // for(auto i : dinput)
        // {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        ASSERT_EQ(miopen::range_distance(ref_dinput), miopen::range_distance(dinput));
        EXPECT_LT(dinput_error, threshold)
            << "Error input gradient beyond tolerance Error: " << dinput_error
            << ",  Tolerance: " << threshold;
    }

    PdistTestCase pdist_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> doutput;
    tensor<T> workspace;
    tensor<T> dinput;

    tensor<T> ref_dinput;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr doutput_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr dinput_dev;

    size_t ws_sizeInBytes;
    double p;
};
