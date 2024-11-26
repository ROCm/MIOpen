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
#include <miopen/miopen.h>
#include <miopen/pdist.hpp>
#include <gtest/gtest.h>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "cpu_pdist.hpp"

struct PdistTestCase
{
    std::vector<size_t> dims;
    double p;
    bool is_contiguous;

    friend std::ostream& operator<<(std::ostream& os, const PdistTestCase& tc)
    {
        os << "dims: ( ";
        for(auto d : tc.dims)
        {
            os << d << " ";
        }
        os << ")";
        os << " is_contiguous: " << tc.is_contiguous;
        os << " p: " << tc.p;

        return os;
    }

    std::vector<size_t> GetDims() const { return dims; }
    double GetP() const { return p; }

    PdistTestCase() {}

    PdistTestCase(std::vector<size_t> dims_, double p_, bool is_contiguous_)
        : dims(dims_), p(p_), is_contiguous(is_contiguous_)
    {
    }

    std::vector<size_t> ComputeStrides() const
    {
        std::vector<size_t> inputDim = dims;
        if(!is_contiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!is_contiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

inline std::vector<PdistTestCase> PdistTestConfigs()
{
    return {
        // Currently, MIOpen doesn't support for empty tensors, so skip those tests where
        // input.shape[0] == 0 || input.shape[0] == 1
        // PdistTestCase({1, 1}, 0.0, true),          PdistTestCase({1, 1}, 1.0, true),
        // PdistTestCase({1, 1}, 2.0, true),          PdistTestCase({1, 1}, 5.0, true),
        // PdistTestCase({1, 1}, HUGE_VAL, true),

        PdistTestCase({2, 5}, 2.0, true),
        PdistTestCase({2, 5}, 2.0, false),
        PdistTestCase({2, 10}, 5.0, true),
        PdistTestCase({2, 10}, 5.0, false),
        PdistTestCase({5, 1}, 0.0, true),
        PdistTestCase({5, 1}, 0.0, false),
        PdistTestCase({5, 5}, 0.0, true),          
        PdistTestCase({5, 5}, 0.0, false),          
        PdistTestCase({10, 1}, 1.0, true),
        PdistTestCase({10, 1}, 1.0, false),
        PdistTestCase({10, 10}, 0.0, true),        
        PdistTestCase({100, 100}, 0.0, true),
        PdistTestCase({100, 100}, 1.0, true),      
        PdistTestCase({100, 100}, 2.0, true),
        PdistTestCase({100, 100}, 5.0, true),
        PdistTestCase({100, 100}, 5.0, false),

        PdistTestCase({2, 5}, HUGE_VAL, true),     
        PdistTestCase({2, 10}, HUGE_VAL, true),
        PdistTestCase({5, 1}, HUGE_VAL, true),     
        PdistTestCase({5, 5}, HUGE_VAL, true),
        PdistTestCase({10, 1}, HUGE_VAL, true),    
        PdistTestCase({10, 1}, HUGE_VAL, false),    
        PdistTestCase({10, 10}, HUGE_VAL, true),
        PdistTestCase({100, 100}, HUGE_VAL, true),
        PdistTestCase({100, 100}, HUGE_VAL, false),
    };
}

// Remove big tests with reduction from FP16 test because the result will be overflow/ underflow
inline std::vector<PdistTestCase> PdistFp16TestConfigs()
{
    // clang-format off
    return {
        PdistTestCase({2, 5}, 2.0, true),
        PdistTestCase({2, 5}, 2.0, false),
        PdistTestCase({5, 1}, 0.0, true),
        PdistTestCase({5, 5}, 0.0, true),
        PdistTestCase({10, 1}, 1.0, true),
        PdistTestCase({10, 1}, 1.0, false),
        PdistTestCase({10, 10}, 0.0, true),
        PdistTestCase({100, 100}, 0.0, true),
        PdistTestCase({100, 100}, 1.0, true),
        PdistTestCase({100, 100}, 2.0, true),
        PdistTestCase({100, 100}, 2.0, false),

        PdistTestCase({2, 5}, HUGE_VAL, true),
        PdistTestCase({2, 5}, HUGE_VAL, false),
        PdistTestCase({2, 10}, HUGE_VAL, true),
        PdistTestCase({5, 1}, HUGE_VAL, true),
        PdistTestCase({5, 5}, HUGE_VAL, true),
        PdistTestCase({5, 5}, HUGE_VAL, false),
        PdistTestCase({10, 1}, HUGE_VAL, true),
        PdistTestCase({10, 10}, HUGE_VAL, true),
        PdistTestCase({100, 100}, HUGE_VAL, true),
        PdistTestCase({100, 100}, HUGE_VAL, false),

        
    };
    // clang-format on
}

template <typename T>
struct PdistTestBackward : public ::testing::TestWithParam<PdistTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        pdist_config  = GetParam();

        auto input_dims    = pdist_config.GetDims();
        auto input_strides = pdist_config.ComputeStrides();

        p                    = pdist_config.GetP();
        auto N               = input_dims[0];
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

        input  = tensor<T>{input_dims, input_strides}.generate(gen_value);
        output = tensor<T>{output_dims}.generate(output_gen_value);

        doutput = tensor<T>{output_dims};
        std::fill(doutput.begin(), doutput.end(), 1.0);

        dinput = tensor<T>{input_dims};
        std::fill(dinput.begin(), dinput.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dinput = tensor<T>{input_dims};
        std::fill(ref_dinput.begin(), ref_dinput.end(), std::numeric_limits<T>::quiet_NaN());

        ws_sizeInBytes = miopen::pdist::GetPdistBackwardWorkspaceSize(
            handle, input.desc, output.desc, doutput.desc, dinput.desc, p);

        if(ws_sizeInBytes <= 0)
            GTEST_FAIL() << "Call GetPdistBackwardWorkspaceSize failed!";

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
        cpu_pdist_backward<T>(input, output, doutput, ref_dinput, p);

        // Run kernel
        status = miopen::pdist::PdistBackward(handle,
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

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        dinput.data = handle.Read<T>(dinput_dev, dinput.data.size());
    }

    void Verify()
    {
        double threshold  = std::numeric_limits<T>::epsilon() * 10;
        auto dinput_error = miopen::rms_range(ref_dinput, dinput);

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
