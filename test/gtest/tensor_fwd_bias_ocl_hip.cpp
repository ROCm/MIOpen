/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/datatype.hpp>
#include <gtest/gtest.h>

#include "get_handle.hpp"
#include "tensor_util.hpp"
#include "verify.hpp"

#define PERF_ENABLE 0
#if PERF_ENABLE
#include "perf_helper.hpp"
#define POW_2 1
#endif

#define MAX_TENSOR_ELEM 17

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
    std::vector<TensorsConfig> configs;
    auto insertTestCase = [&configs](size_t N, size_t C, size_t H, size_t W) {
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, C, H, W}, {C * H * W, H * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, C, H, 1}, {C * H * 1, H * 1, 1, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, C, 1, W}, {C * 1 * W, 1 * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, C, 1, 1}, {C * 1 * 1, 1 * 1, 1, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, 1, H, W}, {1 * H * W, H * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, 1, H, 1}, {1 * H * 1, H * 1, 1, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, 1, 1, W}, {1 * 1 * W, 1 * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, 1, 1, 1}, {1 * 1 * 1, 1 * 1, 1, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, C, H, W}, {C * H * W, H * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, C, H, 1}, {C * H * 1, H * 1, 1, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, C, 1, W}, {C * 1 * W, 1 * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, C, 1, 1}, {C * 1 * 1, 1 * 1, 1, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, 1, H, W}, {1 * H * W, H * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, 1, H, 1}, {1 * H * 1, H * 1, 1, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, 1, 1, W}, {1 * 1 * W, 1 * W, W, 1}});
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {1, 1, 1, 1}, {1 * 1 * 1, 1 * 1, 1, 1}});
    };

#if PERF_ENABLE
    size_t maxTotalSize = getCacheSizeLimit(get_handle().GetDeviceName()) / sizeof(T);

    // Generate all NCHW tensors that are limited by L3 cache size
    // or 2xL2 cache size when L3 is not available
#if POW_2
    for(size_t N = 1; N <= maxTotalSize; N *= 2)
    {
        for(size_t C = 1; C <= maxTotalSize / N; C *= 2)
        {
            for(size_t H = 1; H <= maxTotalSize / (N * C); H *= 2)
            {
                for(size_t W = 1; W <= maxTotalSize / (N * C * H); W *= 2)
                {
                    size_t totalSize = N * C * H * W;
                    // Ensure the total size does not exceed the maximum limit
                    if(totalSize <= maxTotalSize)
                    {
                        insertTestCase(N, C, H, W);
                    }
                }
            }
        }
    }
#else
    for(size_t N = 1; N <= maxTotalSize; N *= 2)
    {
        for(size_t C = 1; C <= maxTotalSize / N; C *= 2)
        {
            for(size_t H = 1; H <= maxTotalSize / (N * C); H *= 2)
            {
                for(size_t W = 1; W <= maxTotalSize / (N * C * W); W *= 2)
                {
                    for(int dn = -1; dn <= 1; dn += 2)
                    {
                        for(int dc = -1; dc <= 1; dc += 2)
                        {
                            for(int dh = -1; dh <= 1; dh += 2)
                            {
                                for(int dw = -1; dw <= 1; dw += 2)
                                {
                                    size_t totalSize = (N + dn) * (C + dc) * (H + dh) * (W + dw);
                                    // Ensure the total size does not exceed the maximum limit
                                    if(totalSize <= maxTotalSize)
                                    {
                                        insertTestCase((N + dn), (C + dc), (H + dh), (W + dw));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    return configs;
#else
    size_t N = 1;
    size_t C = 1;
    size_t H = 1;
    size_t W = 1;
    insertTestCase(N, C, H, W);
    C = 4;
    H = 16;
    W = 16;
    insertTestCase(N, C, H, W);
    C = 1;
    H = 20;
    W = 20;
    insertTestCase(N, C, H, W);
    return configs;
#endif
}

template <typename T>
struct OpTensorFwdBiasTest
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

        num_wg      = first_not_one != tensorsConfig.blens.rend()
                          ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                          : 1;
        work_per_wg = std::accumulate(tensorsConfig.aclens.begin() + d,
                                      tensorsConfig.aclens.end(),
                                      1,
                                      std::multiplies<int>());

        for(int i = (d - 2); i >= 0; i--)
        {
            if(tensorsConfig.blens[i] != 1)
            {
                num_wg *= tensorsConfig.blens[i];
            }
            else
            {
                work_per_wg *= tensorsConfig.aclens[i];
            }
        }

        if(num_wg > max_num_wg)
            num_wg = max_num_wg;

        size_t local_threads  = 256;
        size_t global_threads = num_wg * local_threads;
        global_threads        = (global_threads < local_threads) ? local_threads : global_threads;

        vld = {local_threads, 1, 1};
        vgd = {global_threads, 1, 1};

        network_config += std::to_string(data_type) + "-miopenTensorOpAdd-" +
                          std::to_string(global_threads) + "-" + std::to_string(local_threads);
    }

    void runOCL()
    {
        auto&& handle = get_handle();
        // Write data to device tensor
        tensC_dev = handle.Write(tensC.data);

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::OpenCL{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_FWD_BIAS";

        std::string program_name{"MIOpenTensorKernels.cl"};
        std::string network_config_ocl = network_config + "-ocl";

        handle.AddKernel(
            kernel_name, network_config_ocl, program_name, kernel_name, vld, vgd, params)(
            tensA_dev.get(),
            tensB_dev.get(),
            static_cast<int>(tensorsConfig.blens[1]), // b_c
            tensC_dev.get(),
            static_cast<int>(tensorsConfig.aclens[0]),    // c_n
            static_cast<int>(tensorsConfig.acstrides[0]), // c_nstride
            static_cast<int>(tensorsConfig.acstrides[1]), // c_cstride
            work_per_wg,
            alpha0,
            alpha1,
            beta,
            static_cast<int64_t>(0), // Aoffset
            static_cast<int64_t>(0), // Boffset
            static_cast<int64_t>(0), // Coffset
            num_wg,
            incr_wg);

        tensC_ocl.data = handle.Read<T>(tensC_dev, tensC_ocl.data.size());

#if PERF_ENABLE
        ph.perfTest(handle,
                    kernel_name,
                    network_config_ocl,
                    false,
                    tensA_dev.get(),
                    tensB_dev.get(),
                    static_cast<int>(tensorsConfig.blens[1]),
                    tensC_dev.get(),
                    static_cast<int>(tensorsConfig.aclens[0]),
                    static_cast<int>(tensorsConfig.acstrides[0]),
                    static_cast<int>(tensorsConfig.acstrides[1]),
                    work_per_wg,
                    alpha0,
                    alpha1,
                    beta,
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    num_wg,
                    incr_wg);
#endif
    }

    void runHIP()
    {
        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::HIP{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_FWD_BIAS";

        std::string program_name{"MIOpenTensorKernelsHip.cpp"};
        std::string network_config_hip = network_config + "-hip";

        handle.AddKernel(
            kernel_name, network_config_hip, program_name, kernel_name, vld, vgd, params)(
            tensA_dev.get(),
            tensB_dev.get(),
            static_cast<int>(tensorsConfig.blens[1]), // b_c
            tensC_dev.get(),
            static_cast<int>(tensorsConfig.aclens[0]),    // c_n
            static_cast<int>(tensorsConfig.acstrides[0]), // c_nstride
            static_cast<int>(tensorsConfig.acstrides[1]), // c_cstride
            work_per_wg,
            alpha0,
            alpha1,
            beta,
            static_cast<int64_t>(0), // Aoffset
            static_cast<int64_t>(0), // Boffset
            static_cast<int64_t>(0), // Coffset
            num_wg,
            incr_wg);

        tensC_hip.data = handle.Read<T>(tensC_dev, tensC_hip.data.size());

#if PERF_ENABLE
        ph.perfTest(handle,
                    kernel_name,
                    network_config_hip,
                    false,
                    tensA_dev.get(),
                    tensB_dev.get(),
                    static_cast<int>(tensorsConfig.blens[1]),
                    tensC_dev.get(),
                    static_cast<int>(tensorsConfig.aclens[0]),
                    static_cast<int>(tensorsConfig.acstrides[0]),
                    static_cast<int>(tensorsConfig.acstrides[1]),
                    work_per_wg,
                    alpha0,
                    alpha1,
                    beta,
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    num_wg,
                    incr_wg);
#endif
    }

    void verify()
    {
        auto error = miopen::rms_range(tensC_ocl, tensC_hip);
        EXPECT_TRUE(error == 0) << "GPU outputs do not match each other. Error: " << error;
    }

    void TearDown() override
    {
#if PERF_ENABLE
        std::string stats{};
        stats += "_aclens_" + std::to_string(tensorsConfig.aclens[0]) + "_" +
                 std::to_string(tensorsConfig.aclens[1]) + "_" +
                 std::to_string(tensorsConfig.aclens[2]) + "_" +
                 std::to_string(tensorsConfig.aclens[3]) + "_acstrides_" +
                 std::to_string(tensorsConfig.acstrides[0]) + "_" +
                 std::to_string(tensorsConfig.acstrides[1]) + "_" +
                 std::to_string(tensorsConfig.acstrides[2]) + "_" +
                 std::to_string(tensorsConfig.acstrides[3]);
        stats += "_blens_" + std::to_string(tensorsConfig.blens[0]) + "_" +
                 std::to_string(tensorsConfig.blens[1]) + "_" +
                 std::to_string(tensorsConfig.blens[2]) + "_" +
                 std::to_string(tensorsConfig.blens[3]) + "_bstrides_" +
                 std::to_string(tensorsConfig.bstrides[0]) + "_" +
                 std::to_string(tensorsConfig.bstrides[1]) + "_" +
                 std::to_string(tensorsConfig.bstrides[2]) + "_" +
                 std::to_string(tensorsConfig.bstrides[3]);
        stats += "_alpha0_" + std::to_string(alpha0) + "_alpha1_" + std::to_string(alpha1) +
                 "_beta_" + std::to_string(beta) + "_" + miopen::GetDataType(data_type);

        ph.writeStatsToCSV("tensor_fwd_bias.csv", stats);
#endif
    }

    const std::string kernel_name{"OpTensorFwdBias"};
    std::string network_config{};
    std::string params{};
    std::vector<size_t> vld, vgd;
    const int max_num_wg = 4096;
    int work_per_wg, num_wg, incr_wg{0};

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

#if PERF_ENABLE
    PerfHelper<float> ph;
#endif
};

struct GPU_OpTensorFwdBiasTest_FP32 : OpTensorFwdBiasTest<float>
{
};

TEST_P(GPU_OpTensorFwdBiasTest_FP32, PortTest)
{
    // run OCL kernel
    runOCL();
    // run HIP kernel
    runHIP();
    // verify if the output tensors are same
    verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_OpTensorFwdBiasTest_FP32,
                         testing::Combine(testing::ValuesIn(TensorsConfigs<float>()),
                                          testing::Values(1.0f),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f, 1.0f)));
