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
#include "get_handle.hpp"
#include "random.hpp"
#include <verify.hpp>
#include <miopen/env.hpp>
#include <miopen/miopen.h>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <gtest/gtest.h>

#include "perf_helper.hpp"
#include <miopen/float_equal.hpp>

#define MAX_TENSOR_ELEM 17
#define PERF_ENABLE 0
#define POW_2 1

struct TensorsConfig
{
    std::vector<std::size_t> aclens;
    std::vector<std::size_t> acstrides;
    std::vector<std::size_t> blens;
    std::vector<std::size_t> bstrides;
};

template <typename T>
std::vector<TensorsConfig> TensorsConfigs()
{
    if constexpr(PERF_ENABLE)
    {
        std::vector<TensorsConfig> configs;
        const auto& handle = get_handle();
        size_t maxTotalSize;

        // Generate all NCHW tensors that are limited by L3 cache size
        // or 2xL2 cache size when L3 is not available
        if(miopen::StartsWith(handle.GetDeviceName(), "gfx90a") ||
           miopen::StartsWith(handle.GetDeviceName(), "gfx908"))
        {
            maxTotalSize = 16; // twice the 8MB L2
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx803"))
        {
            maxTotalSize = 4; // twice the 2MB L2
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx900") ||
                miopen::StartsWith(handle.GetDeviceName(), "gfx906"))
        {
            maxTotalSize = 8; // twice the 4MB L2
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx942"))
        {
            maxTotalSize = 256; // 256MB L3
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx103"))
        {
            maxTotalSize = 128; // 128MB L3
        }
        else
        {
            maxTotalSize = 4; // twice the 2MB L2, default case.
        }

        maxTotalSize = maxTotalSize * 1024ull * 1024ull / sizeof(T);

        if constexpr(POW_2)
        {
            for(size_t N = 1; N <= maxTotalSize; N *= 2)
            {
                for(size_t C = 1; C <= maxTotalSize / N; C *= 2)
                {
                    size_t totalSize = N * C;
                    // Ensure the total size does not exceed the maximum limit
                    if(totalSize <= maxTotalSize)
                    {
                        configs.push_back({{N, C}, {C, 1}, {N, C}, {C, 1}});
                        configs.push_back({{N, C}, {C, 1}, {1, C}, {C, 1}});
                        configs.push_back({{N, C}, {C, 1}, {N, 1}, {1, 1}});
                        configs.push_back({{N, C}, {C, 1}, {1, 1}, {1, 1}});
                    }
                }
            }
        }
        else
        {
            for(size_t N = 2; N <= maxTotalSize; N *= 2)
            {
                for(size_t C = 2; C <= maxTotalSize / N; C *= 2)
                {
                    size_t totalSize = (N - 1) * (C - 1);
                    // Ensure the total size does not exceed the maximum limit
                    if(totalSize <= maxTotalSize)
                    {
                        configs.push_back({{N - 1, C - 1}, {C - 1, 1}, {N - 1, C - 1}, {C - 1, 1}});
                        configs.push_back({{N - 1, C - 1}, {C - 1, 1}, {1, C - 1}, {C - 1, 1}});
                        configs.push_back({{N - 1, C - 1}, {C - 1, 1}, {N - 1, 1}, {1, 1}});
                        configs.push_back({{N - 1, C - 1}, {C - 1, 1}, {1, 1}, {1, 1}});
                    }

                    totalSize = (N + 1) * (C + 1);
                    if(totalSize <= maxTotalSize)
                    {
                        configs.push_back({{N + 1, C + 1}, {C + 1, 1}, {N + 1, C + 1}, {C + 1, 1}});
                        configs.push_back({{N + 1, C + 1}, {C + 1, 1}, {1, C + 1}, {C + 1, 1}});
                        configs.push_back({{N + 1, C + 1}, {C + 1, 1}, {N + 1, 1}, {1, 1}});
                        configs.push_back({{N + 1, C + 1}, {C + 1, 1}, {1, 1}, {1, 1}});
                    }
                }
            }
        }

        return configs;
    }
    else
    {
        return {{{16, 8}, {8, 1}, {16, 8}, {8, 1}},
                {{16, 8}, {8, 1}, {1, 8}, {8, 1}},
                {{16, 8}, {8, 1}, {16, 1}, {1, 1}},
                {{16, 8}, {8, 1}, {1, 1}, {1, 1}},
                {{20, 16}, {16, 1}, {20, 16}, {16, 1}},
                {{20, 16}, {16, 1}, {1, 16}, {16, 1}},
                {{20, 16}, {16, 1}, {20, 1}, {1, 1}},
                {{20, 16}, {16, 1}, {1, 1}, {1, 1}},
                {{32, 64}, {64, 1}, {32, 64}, {64, 1}},
                {{32, 64}, {64, 1}, {1, 64}, {64, 1}},
                {{32, 64}, {64, 1}, {32, 1}, {1, 1}},
                {{32, 64}, {64, 1}, {1, 1}, {1, 1}}};
    }
}

template <typename T>
struct Op2DTensorGenericTest
    : public ::testing::TestWithParam<std::tuple<TensorsConfig, float, float, float>>
{
protected:
    void SetUp() override
    {
        auto&& handle                                 = get_handle();
        std::tie(tensorsConfig, alpha0, alpha1, beta) = GetParam();

        data_type = miopen_type<T>{};

        // Generate elements in tensors
        tensA = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});
        tensB = tensor<T>{tensorsConfig.blens, tensorsConfig.bstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});
        tensC = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});

        // Write the device tensors
        tensA_dev = handle.Write(tensA.data);
        tensB_dev = handle.Write(tensB.data);

        // Allocate output tensors for OCL and HIP
        tensC_ocl = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};
        tensC_hip = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};

        // Prepare all parameters needed for kernel
        auto first_not_one = std::find_if(
            tensorsConfig.blens.rbegin(), tensorsConfig.blens.rend(), [](int i) { return i != 1; });
        auto d = std::distance(tensorsConfig.blens.begin(), first_not_one.base());

        int num_wg  = first_not_one != tensorsConfig.blens.rend()
                          ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                          : 1;
        work_per_wg = std::accumulate(tensorsConfig.aclens.begin() + d,
                                      tensorsConfig.aclens.end(),
                                      1,
                                      std::multiplies<int>());

        bitmap = 0;

        bitmap |= (1 << (tensorsConfig.blens.size() - d));

        for(int i = (d - 2); i >= 0; i--)
        {
            if(tensorsConfig.blens[i] != 1)
            {
                bitmap |= (1 << (tensorsConfig.blens.size() - (i + 1)));
                num_wg *= tensorsConfig.blens[i];
            }
            else
            {
                work_per_wg *= tensorsConfig.aclens[i];
            }
        }

        num_wg_orig = num_wg;
        max_num_wg  = 4096;
        num_wg      = num_wg > max_num_wg ? max_num_wg : num_wg;

        size_t local_threads = 256;

        vld = {local_threads, 1, 1};

        size_t global_threads = num_wg * local_threads;

        vgd = {global_threads, 1, 1};

        network_config += std::to_string(data_type) + "-miopenTensorOpAdd-" +
                          std::to_string(global_threads) + "-" + std::to_string(local_threads);
    }

    void runOCL()
    {
        auto&& handle = get_handle();
        // Write data to device tensor
        tensC_dev = handle.Write(tensC.data);
        std::fill(tensC_ocl.begin(), tensC_ocl.end(), std::numeric_limits<T>::quiet_NaN());

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::OpenCL{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_GENERIC";

        std::string program_name       = "MIOpenTensorKernels.cl";
        std::string network_config_ocl = network_config + "-ocl";

        handle.AddKernel("Op2dTensorGeneric",
                         network_config_ocl,
                         program_name,
                         "Op2dTensorGeneric",
                         vld,
                         vgd,
                         params)(tensA_dev.get(),
                                 static_cast<int>(tensorsConfig.acstrides[0]),
                                 tensB_dev.get(),
                                 static_cast<int>(tensorsConfig.blens[1]),
                                 static_cast<int>(tensorsConfig.bstrides[0]),
                                 tensC_dev.get(),
                                 static_cast<int>(tensorsConfig.aclens[1]),
                                 static_cast<int>(tensorsConfig.acstrides[0]),
                                 alpha0,
                                 alpha1,
                                 beta,
                                 bitmap,
                                 work_per_wg,
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int>(num_wg_orig));

        tensC_ocl.data = handle.Read<T>(tensC_dev, tensC_ocl.data.size());

        if constexpr(PERF_ENABLE)
        {
            ph.perfTest(handle,
                        "Op2dTensorGeneric",
                        network_config_ocl,
                        false,
                        tensA_dev.get(),
                        static_cast<int>(tensorsConfig.acstrides[0]),
                        tensB_dev.get(),
                        static_cast<int>(tensorsConfig.blens[1]),
                        static_cast<int>(tensorsConfig.bstrides[0]),
                        tensC_dev.get(),
                        static_cast<int>(tensorsConfig.aclens[1]),
                        static_cast<int>(tensorsConfig.acstrides[0]),
                        alpha0,
                        alpha1,
                        beta,
                        bitmap,
                        work_per_wg,
                        static_cast<int64_t>(0),
                        static_cast<int64_t>(0),
                        static_cast<int64_t>(0),
                        static_cast<int>(num_wg_orig));
        }
    }

    void runHIP()
    {
        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);

        std::fill(tensC_hip.begin(), tensC_hip.end(), std::numeric_limits<T>::quiet_NaN());

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::HIP{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_GENERIC";

        std::string program_name       = "MIOpenTensorKernelsHip.cpp";
        std::string network_config_hip = network_config + "-hip";

        handle.AddKernel("Op2dTensorGeneric",
                         network_config_hip,
                         program_name,
                         "Op2dTensorGeneric",
                         vld,
                         vgd,
                         params)(tensA_dev.get(),
                                 static_cast<int>(tensorsConfig.acstrides[0]),
                                 tensB_dev.get(),
                                 static_cast<int>(tensorsConfig.blens[1]),
                                 static_cast<int>(tensorsConfig.bstrides[0]),
                                 tensC_dev.get(),
                                 static_cast<int>(tensorsConfig.aclens[1]),
                                 static_cast<int>(tensorsConfig.acstrides[0]),
                                 alpha0,
                                 alpha1,
                                 beta,
                                 bitmap,
                                 work_per_wg,
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int>(num_wg_orig));

        tensC_hip.data = handle.Read<T>(tensC_dev, tensC_hip.data.size());

        if constexpr(PERF_ENABLE)
        {
            ph.perfTest(handle,
                        "Op2dTensorGeneric",
                        network_config_hip,
                        false,
                        tensA_dev.get(),
                        static_cast<int>(tensorsConfig.acstrides[0]),
                        tensB_dev.get(),
                        static_cast<int>(tensorsConfig.blens[1]),
                        static_cast<int>(tensorsConfig.bstrides[0]),
                        tensC_dev.get(),
                        static_cast<int>(tensorsConfig.aclens[1]),
                        static_cast<int>(tensorsConfig.acstrides[0]),
                        alpha0,
                        alpha1,
                        beta,
                        bitmap,
                        work_per_wg,
                        static_cast<int64_t>(0),
                        static_cast<int64_t>(0),
                        static_cast<int64_t>(0),
                        static_cast<int>(num_wg_orig));
        }
    }

    void verify()
    {
        auto error = miopen::rms_range(tensC_ocl, tensC_hip);
        EXPECT_TRUE(error == 0) << "GPU outputs do not match each other. Error: " << error;
    }

    void TearDown() override
    {
        if constexpr(PERF_ENABLE)
        {
            std::string stats{};
            stats += "_aclens_" + std::to_string(tensorsConfig.aclens[0]) + "_" +
                     std::to_string(tensorsConfig.aclens[1]) + "_acstrides_" +
                     std::to_string(tensorsConfig.acstrides[0]) + "_" +
                     std::to_string(tensorsConfig.acstrides[1]);
            stats += "_blens_" + std::to_string(tensorsConfig.blens[0]) + "_" +
                     std::to_string(tensorsConfig.blens[1]) + "_bstrides_" +
                     std::to_string(tensorsConfig.bstrides[0]) + "_" +
                     std::to_string(tensorsConfig.bstrides[1]) + "_";
            stats += "alpha0_" + std::to_string(alpha0) + "_alpha1_" + std::to_string(alpha1) +
                     "_beta_" + std::to_string(beta) + "_" + miopen::GetDataType(data_type);

            ph.writeStatsToCSV("tensor_2d.csv", stats);
        }
    }

    std::string network_config{};
    std::string params{};
    std::vector<size_t> vld, vgd;
    unsigned int bitmap;
    int work_per_wg;
    int num_wg_orig;
    int max_num_wg;

    tensor<T> tensA;
    tensor<T> tensB;
    tensor<T> tensC;
    tensor<T> tensC_ocl;
    tensor<T> tensC_hip;

    miopenDataType_t data_type;

    miopen::Allocator::ManageDataPtr tensA_dev;
    miopen::Allocator::ManageDataPtr tensB_dev;
    miopen::Allocator::ManageDataPtr tensC_dev;

    TensorsConfig tensorsConfig;
    float alpha0, alpha1, beta;

    PerfHelper<T> ph;
};

struct GPU_Op2dTensorGenericTest_FP32 : Op2DTensorGenericTest<float>
{
};

TEST_P(GPU_Op2dTensorGenericTest_FP32, PortTest)
{
    // run OCL kernel
    runOCL();
    // run HIP kernel
    runHIP();
    // verify if the output tensors are same
    verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Op2dTensorGenericTest_FP32,
                         testing::Combine(testing::ValuesIn(TensorsConfigs<float>()),
                                          testing::Values(1.0f),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f, 1.0f)));
