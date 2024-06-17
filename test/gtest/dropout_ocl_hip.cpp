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

#include "get_handle.hpp"
#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>

#include "driver.hpp"
#include "dropout_util.hpp"
#include "tensor_holder.hpp"

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

#include <miopen/config.h>
#include <miopen/dropout.hpp>
#include <miopen/env.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

// #define DROPOUT_DEBUG_CTEST 0
// // Workaround for issue #1128
// #define DROPOUT_SINGLE_CTEST 1

template <typename T>
tensor<T> FWDropCPU(const miopen::DropoutDescriptor& DropoutDesc,
                const miopen::TensorDescriptor& NoiseShape,
                const tensor<T>& input,
                const tensor<T>& output,
                std::vector<unsigned char>& rsvsp,
                size_t in_offset,
                size_t out_offset,
                size_t rsvsp_offset,
                bool use_rsvsp = true)
{
    auto states_cpu = std::vector<prngStates>(DropoutDesc.stateSizeInBytes);
    InitKernelStateEmulator(states_cpu, DropoutDesc);

    auto out_cpu   = output;
    auto rsvsp_cpu = rsvsp;

    DropoutForwardVerify<T>(get_handle(),
                                DropoutDesc,
                                input.desc,
                                input.data,
                                out_cpu.desc,
                                out_cpu.data,
                                rsvsp_cpu,
                                states_cpu,
                                in_offset,
                                out_offset,
                                rsvsp_offset);

    return out_cpu;
    
}


template <typename T>
tensor<T> FWDropGPU(const miopen::DropoutDescriptor& DropoutDesc,
                const miopen::TensorDescriptor& NoiseShape,
                const tensor<T>& input,
                const tensor<T>& output,
                std::vector<unsigned char>& rsvsp,
                size_t in_offset,
                size_t out_offset,
                size_t rsvsp_offset,
                bool use_rsvsp = true,
                bool use_hip = false
                )
{

        auto&& handle  = get_handle();
        auto out_gpu   = output;
        auto rsvsp_dev = handle.Write(rsvsp);
        auto in_dev    = handle.Write(input.data);
        auto out_dev   = handle.Write(output.data);

        typename std::vector<unsigned char>::iterator rsvsp_ptr;

        rsvsp_ptr = rsvsp.begin();

        if(!use_hip){

            DropoutDesc.DropoutForward(handle,
                                    input.desc,
                                    input.desc,
                                    in_dev.get(),
                                    output.desc,
                                    out_dev.get(),
                                    use_rsvsp ? rsvsp_dev.get() : nullptr,
                                    rsvsp.size(),
                                    in_offset,
                                    out_offset,
                                    rsvsp_offset);

        }else{

            DropoutDesc.DropoutForwardHIP(handle,
                                    input.desc,
                                    input.desc,
                                    in_dev.get(),
                                    output.desc,
                                    out_dev.get(),
                                    use_rsvsp ? rsvsp_dev.get() : nullptr,
                                    rsvsp.size(),
                                    in_offset,
                                    out_offset,
                                    rsvsp_offset);
        }

        out_gpu.data   = handle.Read<T>(out_dev, output.data.size());
        auto rsvsp_gpu = handle.Read<unsigned char>(rsvsp_dev, rsvsp.size());


        std::copy(rsvsp_gpu.begin(), rsvsp_gpu.end(), rsvsp_ptr);
        return out_gpu;
    
}

template <typename T>
tensor<T> BWDropCPU(const miopen::DropoutDescriptor& DropoutDesc,
                const tensor<T>& din,
                const tensor<T>& dout,
                const std::vector<unsigned char>& rsvsp,
                size_t in_offset,
                size_t out_offset,
                size_t rsvsp_offset,
                bool use_rsvsp = true)
{
    auto din_cpu   = din;
    auto rsvsp_cpu = rsvsp;

    DropoutBackwardVerify<T>(DropoutDesc,
                                dout.desc,
                                dout.data,
                                din_cpu.desc,
                                din_cpu.data,
                                rsvsp_cpu,
                                in_offset,
                                out_offset,
                                rsvsp_offset);

    return din_cpu;
}

template <typename T>
tensor<T> BWDropGPU(const miopen::DropoutDescriptor& DropoutDesc,
                            const tensor<T>& din,
                            const tensor<T>& dout,
                            const std::vector<unsigned char>& rsvsp,
                            size_t in_offset,
                            size_t out_offset,
                            size_t rsvsp_offset,
                            bool use_rsvsp = true,
                            bool use_hip = false){

    auto&& handle = get_handle();
    auto din_gpu  = din;

    auto din_dev   = handle.Write(din.data);
    auto dout_dev  = handle.Write(dout.data);
    auto rsvsp_dev = handle.Write(rsvsp);

    if(!use_hip){

        DropoutDesc.DropoutBackward(handle,
                                din.desc,
                                dout.desc,
                                dout_dev.get(),
                                din.desc,
                                din_dev.get(),
                                use_rsvsp ? rsvsp_dev.get() : nullptr,
                                rsvsp.size(),
                                in_offset,
                                out_offset,
                                rsvsp_offset);


    }
    else{

        DropoutDesc.DropoutBackwardHIP(handle,
                                din.desc,
                                dout.desc,
                                dout_dev.get(),
                                din.desc,
                                din_dev.get(),
                                use_rsvsp ? rsvsp_dev.get() : nullptr,
                                rsvsp.size(),
                                in_offset,
                                out_offset,
                                rsvsp_offset);



    }

    din_gpu.data = handle.Read<T>(din_dev, din.data.size());
    return din_gpu;

}

struct DropoutTestCase
{
    bool mask_flag;
};

std::vector<DropoutTestCase> DropoutTestConfigs()
{   // mask enable
    // clang-format off
    return {{false},
            {true}};
    // clang-format on
}

template <typename T = float>
struct DropoutTest : public ::testing::TestWithParam<DropoutTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        dropout_config     = GetParam();

        mask_flag = dropout_config.mask_flag;

        std::vector<std::vector<int>> input_dims;
        std::vector<int> in_dim;
        int rng_mode_cmd = 0;

        input_dims  = get_sub_tensor();

        input_dims.resize(1);

        in_dim = input_dims[0];
       
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        // Create tensors for the forward and backward dropout
        input_f = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        output_f = tensor<T>{in_dim};

        input_b = tensor<T>{in_dim};
        output_b = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});

        output_f_ocl = tensor<T>{in_dim};
        output_f_hip = tensor<T>{in_dim};

        input_b_ocl = tensor<T>{in_dim};
        input_b_hip = tensor<T>{in_dim};

        size_t stateSizeInBytes = std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth()) * sizeof(prngStates);
        size_t reserveSpaceSizeInBytes = input_f.desc.GetElementSize() * sizeof(bool);
        size_t total_mem = 2 * (2 * input_f.desc.GetNumBytes() + reserveSpaceSizeInBytes) + stateSizeInBytes;
        size_t device_mem = handle.GetGlobalMemorySize();

        if(total_mem >= device_mem)
        {
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        DropoutDesc.dropout          = 0.5;
        DropoutDesc.stateSizeInBytes = stateSizeInBytes;
        DropoutDesc.seed             = 0;
        DropoutDesc.rng_mode         = miopenRNGType_t(rng_mode_cmd);;
        DropoutDesc.use_mask         = mask_flag;
    }

    void RunDropout(bool mask_en = false)
    {
        DropoutDesc.use_mask = mask_en;

        auto reserveSpace = std::vector<unsigned char>(input_f.desc.GetElementSize());
        if(mask_en)
        {
            for(size_t i = 0; i < input_f.desc.GetElementSize(); i++)
            {
                reserveSpace[i] = static_cast<unsigned char>(prng::gen_canonical<float>() > DropoutDesc.dropout);
            }
        }
        
        auto&& handle  = get_handle();
        auto state_buf  = handle.Create<unsigned char>(DropoutDesc.stateSizeInBytes);
        DropoutDesc.pstates = state_buf.get();
        DropoutDesc.InitPRNGState(handle, DropoutDesc.pstates, DropoutDesc.stateSizeInBytes, DropoutDesc.seed);

        // forward pass CPU
        output_f = FWDropCPU<T>(DropoutDesc, noise_shape, input_f, output_f, reserveSpace, 0, 0, 0);
        
        // forward pass OCL
        output_f_ocl = FWDropGPU<T>(DropoutDesc, noise_shape, input_f, output_f_ocl, reserveSpace, 0, 0, 0, true, false);

        // forward pass HIP
        DropoutDesc.InitPRNGStateHIP(handle, DropoutDesc.pstates, DropoutDesc.stateSizeInBytes, DropoutDesc.seed);
        output_f_hip = FWDropGPU<T>(DropoutDesc, noise_shape, input_f, output_f_hip, reserveSpace, 0, 0, 0, true, true);

        // backward pass CPU
        input_b = BWDropCPU<T>(DropoutDesc, input_b, output_b, reserveSpace, 0, 0, 0);

        // backward pass OCL
        input_b_ocl = BWDropGPU<T>(DropoutDesc, input_b_ocl, output_b, reserveSpace, 0, 0, 0, true, false);

        // backward pass HIP
        input_b_hip = BWDropGPU<T>(DropoutDesc, input_b_hip, output_b, reserveSpace, 0, 0, 0, true, true);

        if(!mask_en)
        {
            // forward pass CPU
            output_f = FWDropCPU<T>(DropoutDesc, noise_shape, input_f, output_f, reserveSpace, 0, 0, 0, false);
        
            // forward pass OCL
            output_f_ocl = FWDropGPU<T>(DropoutDesc, noise_shape, input_f, output_f_ocl, reserveSpace, 0, 0, 0, false, false);

            // forward pass HIP
            output_f_hip = FWDropGPU<T>(DropoutDesc, noise_shape, input_f, output_f_hip, reserveSpace, 0, 0, 0, false, true);

            // backward pass CPU
            input_b = BWDropCPU<T>(DropoutDesc, input_b, output_b, reserveSpace, 0, 0, 0, false);

            // backward pass OCL
            input_b_ocl = BWDropGPU<T>(DropoutDesc, input_b_ocl, output_b, reserveSpace, 0, 0, 0, false, false);

            // backward pass HIP
            input_b_hip = BWDropGPU<T>(DropoutDesc, input_b_hip, output_b, reserveSpace, 0, 0, 0, false, true);

        }

    }


    void VerifyOCL()
    {
        auto error_f = miopen::rms_range(output_f, output_f_ocl);
        EXPECT_TRUE(miopen::range_distance(output_f) == miopen::range_distance(output_f_ocl));
        EXPECT_TRUE(error_f == 0) << "[CPU-OCL] FW Outputs do not match each other. Error:" << error_f;

        auto error_b = miopen::rms_range(input_b, input_b_ocl);
        EXPECT_TRUE(miopen::range_distance(input_b) == miopen::range_distance(input_b_ocl));
        EXPECT_TRUE(error_b == 0) << "[CPU-OCL] BW Outputs do not match each other. Error:" << error_b;

    }

    void VerifyHIP()
    {
        auto error_f = miopen::rms_range(output_f, output_f_hip);
        EXPECT_TRUE(miopen::range_distance(output_f) == miopen::range_distance(output_f_hip));
        EXPECT_TRUE(error_f == 0) << "[CPU-HIP] FW Outputs do not match each other. Error:" << error_f;

        auto error_b = miopen::rms_range(input_b, input_b_hip);
        EXPECT_TRUE(miopen::range_distance(input_b) == miopen::range_distance(input_b_hip));
        EXPECT_TRUE(error_b == 0) << "[CPU-HIP] BW Outputs do not match each other. Error:" << error_b;

    }

    void VerifyGPU()
    {
        auto error_f = miopen::rms_range(output_f_ocl, output_f_hip);
        EXPECT_TRUE(miopen::range_distance(output_f_ocl) == miopen::range_distance(output_f_hip));
        EXPECT_TRUE(error_f == 0) << "GPU FW Outputs do not match each other. Error:" << error_f;

        auto error_b = miopen::rms_range(input_b_ocl, input_b_hip);
        EXPECT_TRUE(miopen::range_distance(input_b_ocl) == miopen::range_distance(input_b_hip));
        EXPECT_TRUE(error_b == 0) << "GPU BW Outputs do not match each other. Error:" << error_b;
    }

    DropoutTestCase dropout_config;

    tensor<T> input_f; // input tensor for dropout forward
    tensor<T> output_f; // output tensor for dropout forward
    tensor<T> input_b; // input tensor for dropout backward
    tensor<T> output_b; // output tensor for dropout backward

    // Create tensors for the forward and backward dropout OCL
    tensor<T> output_f_ocl;
    tensor<T> input_b_ocl;

    // Create tensors for the forward and backward dropout HIP
    tensor<T> output_f_hip;
    tensor<T> input_b_hip;

    miopen::DropoutDescriptor DropoutDesc;
    float dropout_rate;
    miopen::TensorDescriptor noise_shape;
    bool mask_flag;

};


namespace dropout {

struct DropoutTestFloat : DropoutTest<float>
{
};

} // namespace dropout
using namespace dropout;

TEST_P(DropoutTestFloat, DropoutTest)
{
    // Run with mask
    RunDropout(mask_flag);

    // Run the verification
    // VerifyOCL();
    // VerifyHIP();

    VerifyGPU();

};

INSTANTIATE_TEST_SUITE_P(DropoutTestSet, DropoutTestFloat, testing::ValuesIn(DropoutTestConfigs()));
