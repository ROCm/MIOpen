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
#endif

struct TensorsConfig
{
    std::vector<int> aclens;
    std::vector<int> acstrides;
    std::vector<int> blens;
    std::vector<int> bstrides;
};

template <typename T>
std::vector<TensorsConfig> TensorsConfigs()
{
    std::vector<TensorsConfig> configs;
    auto insertTestCase = [&configs](int N, int C, int H, int W) {
        configs.push_back(
            {{N, C, H, W}, {C * H * W, H * W, W, 1}, {N, C, H, W}, {C * H * W, H * W, W, 1}});
    };

#if PERF_ENABLE
    // Determine a cache-aware cap on total tensor elements for HIP/AMD:
    // 1) Query L2 size via HIP and use 2x L2 as working set
    // 2) Fallback to per-architecture table if L2 is not reported
    size_t maxTotalSize = 0;

    // 1) HIP L2 cache query
    int dev = -1;
    if(hipSuccess == hipGetDevice(&dev))
    {
        int L2_bytes = 0;
        if(hipSuccess == hipDeviceGetAttribute(&L2_bytes, hipDeviceAttributeL2CacheSize, dev) &&
           L2_bytes > 0)
        {
            // Use 2x L2 as a working-set heuristic
            maxTotalSize = 2ul * static_cast<size_t>(L2_bytes);
            // Convert bytes -> elements of type T
            maxTotalSize /= sizeof(T);
        }
    }

    // 2) Fallback table by architecture family
    if(maxTotalSize == 0)
        maxTotalSize = getCacheSizeLimit<T>(get_handle().GetDeviceName());

    int C = 4;
    int H = 8;
    int W = 8;
    for(int N = 32; N <= maxTotalSize / (C * H * W); N *= 2)
    {
        insertTestCase(N, C, H, W);
    }

    return configs;
#else
    int N = 1;
    int C = 1;
    int H = 1;
    int W = 1;
    insertTestCase(N, C, H, W);
    N = 1024;
    C = 4;
    H = 8;
    W = 8;
    insertTestCase(N, C, H, W);
    N = 2048;
    C = 4;
    H = 16;
    W = 16;
    insertTestCase(N, C, H, W);

    return configs;
#endif
}

template <typename T>
struct OpTensorLeadingOnesTest
    : public ::testing::TestWithParam<std::tuple<TensorsConfig, double, double, double>>
{
protected:
    void SetUp() override
    {
        auto&& handle                                 = get_handle();
        std::tie(tensorsConfig, alpha0, alpha1, beta) = GetParam();

        data_type = miopen_type<T>{};

        // Generate elements in tensors
        tensA = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{17});
        tensB = tensor<T>{tensorsConfig.blens, tensorsConfig.bstrides}.generate(
            tensor_elem_gen_integer{17});
        tensC = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};

        // Write the device tensors
        tensA_dev = handle.Write(tensA.data);
        tensB_dev = handle.Write(tensB.data);

        // Allocate output tensors for OCL and HIP
        tensC_ocl = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};
        tensC_hip = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};

        // first_not_one is incorrect if btensor size equal to 1
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

        bitmap = 0;
        // update bitmap for first_not_one
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

        // quick fix for btensor = <1,1,1,1>
        if(1 ==
           std::accumulate(
               tensorsConfig.blens.begin(), tensorsConfig.blens.end(), 1, std::multiplies<int>()))
        {
            bitmap = 4;
        }

        // Forward Convolution Bias specialization
        // for fwd-bias, bitmap looks like <0,1,0,0>
        // Is the no. of work-groups and the work for each wg balanced?
        auto fwd_conv_bias = bitmap == (1 << 2) ? 1 : 0;
        auto dims          = tensorsConfig.aclens.size();
        auto c_n           = tensorsConfig.aclens[0];
        // This block gives off indexing for 5d tensors, skipping
        if(fwd_conv_bias == 1 && dims < 5 && num_wg < 640 && work_per_wg > 256 && c_n > 0)
        { // 640 workgroups of size 256 needed to completely fill the GPU
            work_per_wg /= c_n;
            num_wg *= c_n;
        }

        if(num_wg > max_num_wg)
            num_wg = max_num_wg;

        size_t local_threads = 256;

        bool leading_ones = true;
        for(int i = (d - 2); i >= 0; i--)
        {
            bool is_one = (bitmap & (1 << (dims - 1 - i))) != 0u;
            leading_ones &= is_one;
        }
        if(leading_ones && work_per_wg < 64)
        {
            local_threads = 64;
        }

        // Special case for adding tensors in place
        size_t global_threads =
            (static_cast<int>(leading_ones) == 1 && (d - 1) == 3) ? num_wg : num_wg * local_threads;
        global_threads = (global_threads < local_threads) ? local_threads : global_threads;

        vld = {local_threads, 1, 1};
        vgd = {global_threads, 1, 1};

        network_config += std::to_string(data_type) + "-miopenTensorOpAdd-" +
                          std::to_string(global_threads) + "-" + std::to_string(local_threads);
    }

    void runOCL() // run OCL kernel
    {
        auto&& handle = get_handle();
        // Write data to device tensor
        tensC_dev = handle.Write(tensC.data);

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::OpenCL{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_LEADING_ONES";

        std::string program_name       = "MIOpenTensorKernels.cl";
        std::string network_config_ocl = network_config + "-ocl";

        handle.AddKernel(
            kernel_name, network_config_ocl, program_name, kernel_name, vld, vgd, params)(
            tensA_dev.get(),
            tensB_dev.get(),
            tensC_dev.get(),
            tensorsConfig.aclens[1],    // c_c
            tensorsConfig.aclens[2],    // c_h
            tensorsConfig.aclens[3],    // c_w
            tensorsConfig.acstrides[0], // c_nstride
            tensorsConfig.acstrides[1], // c_cstride
            tensorsConfig.acstrides[2], // c_hstride
            work_per_wg,
            alpha0,
            alpha1,
            beta,
            0L, // Aoffset
            0L, // Boffset
            0L, // Coffset
            num_wg,
            bitmap);

        tensC_ocl.data = handle.Read<T>(tensC_dev, tensC_ocl.data.size());

#if PERF_ENABLE
        ph.perfTest(handle,
                    kernel_name,
                    network_config_ocl,
                    tensA_dev.get(),
                    tensB_dev.get(),
                    tensC_dev.get(),
                    tensorsConfig.aclens[1],
                    tensorsConfig.aclens[2],
                    tensorsConfig.aclens[3],
                    tensorsConfig.acstrides[0],
                    tensorsConfig.acstrides[1],
                    tensorsConfig.acstrides[2],
                    work_per_wg,
                    alpha0,
                    alpha1,
                    beta,
                    uint64_t(0),
                    uint64_t(0),
                    uint64_t(0),
                    num_wg,
                    bitmap);
#endif
    }

    void runHIP() // run HIP kernel
    {
        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::HIP{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_LEADING_ONES";

        std::string program_name       = "MIOpenTensorKernelsHip.cpp";
        std::string network_config_hip = network_config + "-hip";

        handle.AddKernel(
            kernel_name, network_config_hip, program_name, kernel_name, vld, vgd, params)(
            tensA_dev.get(),
            tensB_dev.get(),
            tensC_dev.get(),
            tensorsConfig.aclens[1],    // c_c
            tensorsConfig.aclens[2],    // c_h
            tensorsConfig.aclens[3],    // c_w
            tensorsConfig.acstrides[0], // c_nstride
            tensorsConfig.acstrides[1], // c_cstride
            tensorsConfig.acstrides[2], // c_hstride
            work_per_wg,
            alpha0,
            alpha1,
            beta,
            uint64_t(0), // Aoffset
            uint64_t(0), // Boffset
            uint64_t(0), // Coffset
            num_wg,
            bitmap);

        tensC_hip.data = handle.Read<T>(tensC_dev, tensC_hip.data.size());

#if PERF_ENABLE
        ph.perfTest(handle,
                    kernel_name,
                    network_config_hip,
                    tensA_dev.get(),
                    tensB_dev.get(),
                    tensC_dev.get(),
                    tensorsConfig.aclens[1],
                    tensorsConfig.aclens[2],
                    tensorsConfig.aclens[3],
                    tensorsConfig.acstrides[0],
                    tensorsConfig.acstrides[1],
                    tensorsConfig.acstrides[2],
                    work_per_wg,
                    alpha0,
                    alpha1,
                    beta,
                    uint64_t(0),
                    uint64_t(0),
                    uint64_t(0),
                    num_wg,
                    bitmap);
#endif
    }

    void verify() // verify if the output tensors are same
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
                 "_beta_" + std::to_string(beta);

        std::string filename = "tensor_leading_ones-" + miopen::GetDataType(data_type) + ".csv";
        ph.writeStatsToCSV(filename, stats);
#endif
    }

    const std::string kernel_name{"OpTensorLeadingOnes"};
    std::string network_config{};
    std::string params{};
    std::vector<size_t> vld, vgd;
    unsigned int bitmap;
    int work_per_wg, num_wg;
    int max_num_wg = 4096;

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
    T alpha0, alpha1, beta;

#if PERF_ENABLE
    PerfHelper ph;
#endif
};

using GPU_OpTensorLeadingOnesTest_FP16 = OpTensorLeadingOnesTest<half_float::half>;

TEST_P(GPU_OpTensorLeadingOnesTest_FP16, PortTest)
{
    runOCL();
    runHIP();
    verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_OpTensorLeadingOnesTest_FP16,
                         testing::Combine(testing::ValuesIn(TensorsConfigs<half_float::half>()),
                                          testing::Values(1.0),
                                          testing::Values(1.0),
                                          testing::Values(0.0, 1.0)));

using GPU_OpTensorLeadingOnesTest_FP32 = OpTensorLeadingOnesTest<float>;

TEST_P(GPU_OpTensorLeadingOnesTest_FP32, PortTest)
{
    runOCL();
    runHIP();
    verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_OpTensorLeadingOnesTest_FP32,
                         testing::Combine(testing::ValuesIn(TensorsConfigs<float>()),
                                          testing::Values(1.0),
                                          testing::Values(1.0),
                                          testing::Values(0.0, 1.0)));

using GPU_OpTensorLeadingOnesTest_FP64 = OpTensorLeadingOnesTest<double>;

TEST_P(GPU_OpTensorLeadingOnesTest_FP64, PortTest)
{
    runOCL();
    runHIP();
    verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_OpTensorLeadingOnesTest_FP64,
                         testing::Combine(testing::ValuesIn(TensorsConfigs<double>()),
                                          testing::Values(1.0),
                                          testing::Values(1.0),
                                          testing::Values(0.0, 1.0)));
