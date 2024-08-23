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
#include <gtest/gtest.h>

#include "perf_helper.hpp"
#include <miopen/float_equal.hpp>

#define MAX_TENSOR_ELEM 17

struct TensorsConfig
{
    std::vector<std::size_t> aclens;
    std::vector<std::size_t> acstrides;
    std::vector<std::size_t> blens;
    std::vector<std::size_t> bstrides;
    std::vector<float> alphabeta;
};

std::vector<TensorsConfig> TensorsConfigs()
{
    return {{{16, 8}, {8, 1}, {16, 8}, {8, 1}, {1, 1, 0}},
            {{16, 8}, {8, 1}, {16, 1}, {1, 1}, {1, 1, 0}},
            {{16, 8}, {8, 1}, {1, 8}, {8, 1}, {1, 1, 0}},
            {{16, 8}, {8, 1}, {1, 1}, {1, 1}, {1, 1, 0}},
            {{16, 8}, {8, 1}, {16, 8}, {8, 1}, {-1, 1, 1}},
            {{16, 8}, {8, 1}, {16, 1}, {1, 1}, {-1, 1, 1}},
            {{16, 8}, {8, 1}, {1, 8}, {8, 1}, {-1, 1, 1}},
            {{16, 8}, {8, 1}, {1, 1}, {1, 1}, {-1, 1, 1}},
            {{16, 8}, {8, 1}, {16, 8}, {8, 1}, {1.0, 0.5, 0}},
            {{16, 8}, {8, 1}, {16, 1}, {1, 1}, {1.0, 0.5, 0}},
            {{16, 8}, {8, 1}, {1, 8}, {8, 1}, {1.0, 0.5, 0}},
            {{16, 8}, {8, 1}, {1, 1}, {1, 1}, {1.0, 0.5, 0}},
            {{20, 16}, {16, 1}, {20, 16}, {16, 1}, {1, 1, 0}},
            {{20, 16}, {16, 1}, {1, 16}, {16, 1}, {1, 1, 0}},
            {{20, 16}, {16, 1}, {20, 1}, {1, 1}, {1, 1, 0}},
            {{20, 16}, {16, 1}, {1, 1}, {1, 1}, {1, 1, 0}},
            {{20, 16}, {16, 1}, {20, 16}, {16, 1}, {-1, 1, 1}},
            {{20, 16}, {16, 1}, {1, 16}, {16, 1}, {-1, 1, 1}},
            {{20, 16}, {16, 1}, {20, 1}, {1, 1}, {-1, 1, 1}},
            {{20, 16}, {16, 1}, {1, 1}, {1, 1}, {-1, 1, 1}},
            {{20, 16}, {16, 1}, {20, 16}, {16, 1}, {1.0, 0.5, 0}},
            {{20, 16}, {16, 1}, {1, 16}, {16, 1}, {1.0, 0.5, 0}},
            {{20, 16}, {16, 1}, {20, 1}, {1, 1}, {1.0, 0.5, 0}},
            {{20, 16}, {16, 1}, {1, 1}, {1, 1}, {1.0, 0.5, 0}},
            {{32, 64}, {64, 1}, {32, 64}, {64, 1}, {1, 1, 0}},
            {{32, 64}, {64, 1}, {1, 64}, {64, 1}, {1, 1, 0}},
            {{32, 64}, {64, 1}, {32, 1}, {1, 1}, {1, 1, 0}},
            {{32, 64}, {64, 1}, {1, 1}, {1, 1}, {1, 1, 0}},
            {{32, 64}, {64, 1}, {32, 64}, {64, 1}, {-1, 1, 1}},
            {{32, 64}, {64, 1}, {1, 64}, {64, 1}, {-1, 1, 1}},
            {{32, 64}, {64, 1}, {32, 1}, {1, 1}, {-1, 1, 1}},
            {{32, 64}, {64, 1}, {1, 1}, {1, 1}, {-1, 1, 1}},
            {{32, 64}, {64, 1}, {32, 64}, {64, 1}, {1.0, 0.5, 0}},
            {{32, 64}, {64, 1}, {1, 64}, {64, 1}, {1.0, 0.5, 0}},
            {{32, 64}, {64, 1}, {32, 1}, {1, 1}, {1.0, 0.5, 0}},
            {{32, 64}, {64, 1}, {1, 1}, {1, 1}, {1.0, 0.5, 0}}};
}

struct Op2DTensorGenericTest : public ::testing::TestWithParam<TensorsConfig>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        tensorsConfig = GetParam();

        tensA = tensor<float>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});
        tensB = tensor<float>{tensorsConfig.blens, tensorsConfig.bstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});
        tensC = tensor<float>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});

        tensA_dev = handle.Write(tensA.data);
        tensB_dev = handle.Write(tensB.data);

        tensC_ocl = tensor<float>{tensorsConfig.aclens, tensorsConfig.acstrides};
        tensC_hip = tensor<float>{tensorsConfig.aclens, tensorsConfig.acstrides};
    }

    void runOCL()
    {
        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);
        std::fill(tensC_ocl.begin(), tensC_ocl.end(), std::numeric_limits<float>::quiet_NaN());

        std::string program_name = "MIOpenTensorKernels.cl";

        auto first_not_one = std::find_if(
            tensorsConfig.blens.rbegin(), tensorsConfig.blens.rend(), [](int i) { return i != 1; });
        auto d = std::distance(tensorsConfig.blens.begin(), first_not_one.base());

        int num_wg      = first_not_one != tensorsConfig.blens.rend()
                              ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                              : 1;
        int work_per_wg = std::accumulate(tensorsConfig.aclens.begin() + d,
                                          tensorsConfig.aclens.end(),
                                          1,
                                          std::multiplies<int>());

        unsigned int bitmap = 0;

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

        int num_wg_orig = num_wg;
        int max_num_wg  = 4096;
        num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

        size_t local_threads = 256;

        const std::vector<size_t> vld{local_threads, 1, 1};

        size_t global_threads = num_wg * local_threads;

        const std::vector<size_t> vgd{global_threads, 1, 1};

        std::string network_config{};
        network_config += "miopenFloat-miopenFloat-miopenTensorOpAdd-" +
                          std::to_string(global_threads) + "-" + std::to_string(local_threads);

        std::string params = " -DMIOPEN_TYPE=float -DMAX_NUM_WG=" + std::to_string(max_num_wg);

        params += miopen::GetDataTypeKernelParams(miopenFloat);
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_GENERIC";

        handle.AddKernel("Op2dTensorGeneric",
                         network_config,
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
                                 tensorsConfig.alphabeta[0],
                                 tensorsConfig.alphabeta[1],
                                 tensorsConfig.alphabeta[2],
                                 bitmap,
                                 work_per_wg,
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int>(num_wg_orig));

        tensC_ocl.data = handle.Read<float>(tensC_dev, tensC_ocl.data.size());

        ph.perfTest(handle,
                    "Op2dTensorGeneric",
                    network_config,
                    false,
                    tensA_dev.get(),
                    static_cast<int>(tensorsConfig.acstrides[0]),
                    tensB_dev.get(),
                    static_cast<int>(tensorsConfig.blens[1]),
                    static_cast<int>(tensorsConfig.bstrides[0]),
                    tensC_dev.get(),
                    static_cast<int>(tensorsConfig.aclens[1]),
                    static_cast<int>(tensorsConfig.acstrides[0]),
                    tensorsConfig.alphabeta[0],
                    tensorsConfig.alphabeta[1],
                    tensorsConfig.alphabeta[2],
                    bitmap,
                    work_per_wg,
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    static_cast<int>(num_wg_orig));
    }

    void runHIP()
    {
        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);

        std::fill(tensC_hip.begin(), tensC_hip.end(), std::numeric_limits<float>::quiet_NaN());

        std::string program_name = "MIOpenTensorKernelsHip.cpp";

        auto first_not_one = std::find_if(
            tensorsConfig.blens.rbegin(), tensorsConfig.blens.rend(), [](int i) { return i != 1; });
        auto d = std::distance(tensorsConfig.blens.begin(), first_not_one.base());

        int num_wg      = first_not_one != tensorsConfig.blens.rend()
                              ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                              : 1;
        int work_per_wg = std::accumulate(tensorsConfig.aclens.begin() + d,
                                          tensorsConfig.aclens.end(),
                                          1,
                                          std::multiplies<int>());

        unsigned int bitmap = 0;

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

        int num_wg_orig = num_wg;
        int max_num_wg  = 4096;
        num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

        size_t local_threads = 256;

        const std::vector<size_t> vld{local_threads, 1, 1};

        size_t global_threads = num_wg * local_threads;

        const std::vector<size_t> vgd{global_threads, 1, 1};

        std::string network_config{};
        network_config += "miopenFloat-miopenFloat-miopenTensorOpAdd-" +
                          std::to_string(global_threads) + "-" + std::to_string(local_threads);

        std::string params = " -DMIOPEN_TYPE=float -DMAX_NUM_WG=" + std::to_string(max_num_wg);

        params += miopen::GetDataTypeKernelParams(miopenFloat);
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_GENERIC";

        handle.AddKernel("Op2dTensorGeneric",
                         network_config,
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
                                 tensorsConfig.alphabeta[0],
                                 tensorsConfig.alphabeta[1],
                                 tensorsConfig.alphabeta[2],
                                 bitmap,
                                 work_per_wg,
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int64_t>(0),
                                 static_cast<int>(num_wg_orig));

        tensC_hip.data = handle.Read<float>(tensC_dev, tensC_hip.data.size());

        ph.perfTest(handle,
                    "Op2dTensorGeneric",
                    network_config,
                    false,
                    tensA_dev.get(),
                    static_cast<int>(tensorsConfig.acstrides[0]),
                    tensB_dev.get(),
                    static_cast<int>(tensorsConfig.blens[1]),
                    static_cast<int>(tensorsConfig.bstrides[0]),
                    tensC_dev.get(),
                    static_cast<int>(tensorsConfig.aclens[1]),
                    static_cast<int>(tensorsConfig.acstrides[0]),
                    tensorsConfig.alphabeta[0],
                    tensorsConfig.alphabeta[1],
                    tensorsConfig.alphabeta[2],
                    bitmap,
                    work_per_wg,
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    static_cast<int64_t>(0),
                    static_cast<int>(num_wg_orig));
    }

    void verify()
    {
        auto error = miopen::rms_range(tensC_ocl, tensC_hip);
        EXPECT_TRUE(error == 0) << "GPU outputs do not match each other. Error: " << error;
    }

    void TearDown() override
    {
        std::string stats{};
        stats += "_aclens_" + std::to_string(tensorsConfig.aclens[0]) + "_" +
                 std::to_string(tensorsConfig.aclens[1]) + "_acstrides_" +
                 std::to_string(tensorsConfig.acstrides[0]) + "_" +
                 std::to_string(tensorsConfig.acstrides[1]);
        stats += "_blens_" + std::to_string(tensorsConfig.blens[0]) + "_" +
                 std::to_string(tensorsConfig.blens[1]) + "_bstrides_" +
                 std::to_string(tensorsConfig.bstrides[0]) + "_" +
                 std::to_string(tensorsConfig.bstrides[1]) + "_float";

        ph.writeStatsToCSV("tensor_2d.csv", stats);
    }

    tensor<float> tensA;
    tensor<float> tensB;
    tensor<float> tensC;
    tensor<float> tensC_ocl;
    tensor<float> tensC_hip;

    miopen::Allocator::ManageDataPtr tensA_dev;
    miopen::Allocator::ManageDataPtr tensB_dev;
    miopen::Allocator::ManageDataPtr tensC_dev;

    TensorsConfig tensorsConfig;

    PerfHelper<float> ph;
};

TEST_P(Op2DTensorGenericTest, Op2DTensorTest)
{
    runOCL();
    runHIP();
    verify();
}

INSTANTIATE_TEST_SUITE_P(Op2DTensorGenericTestSet,
                         Op2DTensorGenericTest,
                         testing::ValuesIn(TensorsConfigs()));
