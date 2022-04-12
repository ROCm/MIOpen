/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include "driver.hpp"
#include "get_handle.hpp"

#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/logger.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/hip_build_utils.hpp>

#include <chrono>
#include <cstdlib>
#include <functional>

namespace miopen {

struct TestRordbEmbedFsOverrideLock
{
    TestRordbEmbedFsOverrideLock() : cached(debug::rordb_embed_fs_override())
    {
        debug::rordb_embed_fs_override() = true;
    }

    ~TestRordbEmbedFsOverrideLock() { debug::rordb_embed_fs_override() = cached; }

    private:
    bool cached;
};

static auto Duration(const std::function<void()>& func)
{
    const auto start = std::chrono::steady_clock::now();
    func();
    return std::chrono::steady_clock::now() - start;
}

struct FindDbTest : test_driver
{
    Handle handle{};
    tensor<float> x;
    tensor<float> w;
    tensor<float> y;
    Allocator::ManageDataPtr x_dev;
    Allocator::ManageDataPtr w_dev;
    Allocator::ManageDataPtr y_dev;
    // --input 16,192,28,28 --weights 32,192,5,5 --filter 2,2,1,1,1,1,
    miopen::ConvolutionDescriptor filter = {
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};

    FindDbTest()
    {
        filter.findMode.Set(FindMode::Values::Normal);
        x = {16, 192, 28, 28};
        w = {32, 192, 5, 5};
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};
    }

    void run()
    {
        x_dev = handle.Write(x.data);
        w_dev = handle.Write(w.data);
        y_dev = handle.Write(y.data);

        const TempFile temp_file{"miopen.test.find_db"};
        testing_find_db_path_override() = temp_file;
        TestRordbEmbedFsOverrideLock rordb_embed_fs_override;

        TestForward();
        TestBwdData();
        TestWeights();
    }

    private:
    void TestBwdData()
    {
        MIOPEN_LOG_I("Starting backward find-db test.");

        auto workspace_size = filter.BackwardDataGetWorkSpaceSize(handle, w.desc, y.desc, x.desc);

        auto workspace     = std::vector<char>(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        auto filterCall = [&]() {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf[1];

            filter.FindConvBwdDataAlgorithm(handle,
                                            y.desc,
                                            y_dev.get(),
                                            w.desc,
                                            w_dev.get(),
                                            x.desc,
                                            x_dev.get(),
                                            1,
                                            &ret_algo_count,
                                            perf,
                                            workspace_dev.get(),
                                            workspace_size,
                                            false);
        };

        Test(filterCall);
    }

    void TestForward()
    {
        std::cout << "Starting forward find-db test." << std::endl;

        auto workspace_size = filter.ForwardGetWorkSpaceSize(handle, w.desc, x.desc, y.desc);

        auto workspace     = std::vector<char>(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        auto filterCall = [&]() {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf[1];

            filter.FindConvFwdAlgorithm(handle,
                                        x.desc,
                                        x_dev.get(),
                                        w.desc,
                                        w_dev.get(),
                                        y.desc,
                                        y_dev.get(),
                                        1,
                                        &ret_algo_count,
                                        perf,
                                        workspace_dev.get(),
                                        workspace_size,
                                        false);
        };

        Test(filterCall);
    }

    void TestWeights()
    {
        MIOPEN_LOG_I("Starting wrw find-db test.");

        auto workspace_size =
            filter.BackwardWeightsGetWorkSpaceSize(handle, y.desc, x.desc, w.desc);

        auto workspace     = std::vector<char>(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        auto filterCall = [&]() {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf[1];

            filter.FindConvBwdWeightsAlgorithm(handle,
                                               y.desc,
                                               y_dev.get(),
                                               x.desc,
                                               x_dev.get(),
                                               w.desc,
                                               w_dev.get(),
                                               1,
                                               &ret_algo_count,
                                               perf,
                                               workspace_dev.get(),
                                               workspace_size,
                                               false);
        };

        Test(filterCall);
    }

    void Test(const std::function<void()>& func)
    {
        using mSeconds = std::chrono::duration<double, std::ratio<1, 1000>>;

        const auto time0   = Duration(func);
        const auto time0ms = std::chrono::duration_cast<mSeconds>(time0);
        MIOPEN_LOG_I("Find(), 1st call (populating kcache, updating find-db): " << time0ms.count());

        testing_find_db_enabled = false;
        const auto time1        = Duration(func);
        const auto time1ms      = std::chrono::duration_cast<mSeconds>(time1);
        MIOPEN_LOG_I("Find(), find-db disabled: " << time1ms.count());

        testing_find_db_enabled = true;
        const auto time2        = Duration(func);
        const auto time2ms      = std::chrono::duration_cast<mSeconds>(time2);
        MIOPEN_LOG_I("Find(), find-db enabled: " << time2ms.count());

        const auto find_db_speedup = time1ms / time2ms;
        MIOPEN_LOG_I("Speedup: " << find_db_speedup);
#if !MIOPEN_DISABLE_USERDB
        double limit = 3.0;
        EXPECT_OP(find_db_speedup, >=, limit);
#endif
    }
};
} // namespace miopen

int main(int argc, const char* argv[])
{
    setenv("MIOPEN_LOG_LEVEL", "6", 1);              // NOLINT (concurrency-mt-unsafe)
    setenv("MIOPEN_COMPILE_PARALLEL_LEVEL", "1", 1); // NOLINT (concurrency-mt-unsafe)
    test_drive<miopen::FindDbTest>(argc, argv);
}
