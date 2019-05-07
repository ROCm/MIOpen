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

#include <chrono>
#include <functional>

#include <cstdlib>

namespace miopen {
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
        x = {16, 192, 28, 28};
        w = {32, 192, 5, 5};
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};
    }

    void run()
    {
        x_dev = handle.Write(x.data);
        w_dev = handle.Write(w.data);
        y_dev = handle.Write(y.data);

        TestForward();
        TestBwdData();
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

    void Test(const std::function<void()>& func)
    {
        using microseconds = std::chrono::microseconds;

        const auto time0   = Duration(func);
        const auto time0us = std::chrono::duration_cast<microseconds>(time0);
        MIOPEN_LOG_I("Noncached call: " << time0us.count() << " us");

        FindDb::enabled = false;

        const auto time1   = Duration(func);
        const auto time1us = std::chrono::duration_cast<microseconds>(time1);
        MIOPEN_LOG_I("No find db call: " << time1us.count() << " us");

        FindDb::enabled = true;

        const auto time2   = Duration(func);
        const auto time2us = std::chrono::duration_cast<microseconds>(time2);
        MIOPEN_LOG_I("Find db call: " << time2us.count() << " us");

        const auto find_db_speedup = time1 / time2;
        MIOPEN_LOG_I("Speedup: " << find_db_speedup << "x");

        EXPECT_OP(find_db_speedup, >=, 3);
    }
};
} // namespace miopen

int main(int argc, const char* argv[])
{
    setenv("MIOPEN_LOG_LEVEL", "6", 1);
    test_drive<miopen::FindDbTest>(argc, argv);
}
