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
    tensor<float> input;
    tensor<float> weights;
    Allocator::ManageDataPtr in_dev;
    Allocator::ManageDataPtr wei_dev;
    miopen::ConvolutionDescriptor filter = {2, miopenConvolution, miopenPaddingDefault};

    FindDbTest()
    {
        input   = {16, 32, 8, 8};
        weights = {64, 32, 5, 5};
    }

    void run()
    {
        in_dev  = handle.Write(input.data);
        wei_dev = handle.Write(weights.data);

        TestForward();
    }

    private:
    void TestForward()
    {
        std::cout << "Starting forward find-db test." << std::endl;

        auto rout    = tensor<float>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
        auto out_dev = handle.Write(rout.data);

        auto workspace_size =
            filter.ForwardGetWorkSpaceSize(handle, weights.desc, input.desc, rout.desc);

        auto workspace     = std::vector<char>(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        auto filterCall = [&]() {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf[1];

            filter.FindConvFwdAlgorithm(handle,
                                        input.desc,
                                        in_dev.get(),
                                        weights.desc,
                                        wei_dev.get(),
                                        rout.desc,
                                        out_dev.get(),
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
