/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

// Test Suite for tensor API

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include <vector>
#include <functional>

// Compiler uses undefined behavior sanitizer
// -fsanitize=enum (or -fsanitize=undefined)
#if(defined(__clang__) || defined(__GNUG__)) && !defined(NDEBUG)
#define UBSAN_ENABLED 1
#else
#define UBSAN_ENABLED 0
#endif

// We use out-of-range values for miopenDataType_t and miopenTensorLayout_t
#define USE_OUT_OF_RANGE_ENUM (UBSAN_ENABLED == 0)

namespace {

enum class TestStatus
{
    Failed,
    Passed,
    Skipped
};

struct TensorParams
{
    miopenDataType_t dataType;
    miopenTensorLayout_t tensorLayout;
    int nbDims;
    int* dimsA;
    int* stridesA;
    bool use_strides;
};

struct TestConfig
{
    bool null_tensor_descriptor;
    TensorParams params;
    bool valid;
};

template <class L1, class L2>
bool CompareLengths(const L1 l1, const L2 l2, int size)
{
    for(int i = 0; i < size; i++)
    {
        if(l1[i] != l2[i])
            return false;
    }
    return true;
}

// Set tensor descriptor
TestStatus Set4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                 const TensorParams& params,
                                 bool check_skip)
{
    if(params.tensorLayout != miopenTensorNCHW || params.nbDims != 4 || params.dimsA == nullptr ||
       params.use_strides)
        return TestStatus::Skipped;

    if(check_skip)
        return TestStatus::Passed;

    miopenStatus_t status = miopenSet4dTensorDescriptor(tensorDesc,
                                                        params.dataType,
                                                        params.dimsA[0],
                                                        params.dimsA[1],
                                                        params.dimsA[2],
                                                        params.dimsA[3]);
    if(status == miopenStatusSuccess)
        return TestStatus::Passed;

    return TestStatus::Failed;
}

TestStatus SetNdTensorDescriptorWithLayout(miopenTensorDescriptor_t tensorDesc,
                                           const TensorParams& params,
                                           bool check_skip)
{
    if(params.use_strides)
        return TestStatus::Skipped;

    if(check_skip)
        return TestStatus::Passed;

    miopenStatus_t status = miopenSetNdTensorDescriptorWithLayout(
        tensorDesc, params.dataType, params.tensorLayout, params.dimsA, params.nbDims);
    if(status == miopenStatusSuccess)
        return TestStatus::Passed;

    return TestStatus::Failed;
}

TestStatus Set4dTensorDescriptorEx(miopenTensorDescriptor_t tensorDesc,
                                   const TensorParams& params,
                                   bool check_skip)
{
    if(params.tensorLayout < miopenTensorNCHW || params.tensorLayout > miopenTensorNDHWC ||
       params.nbDims != 4 || params.dimsA == nullptr || params.stridesA == nullptr ||
       params.use_strides == false)
        return TestStatus::Skipped;

    if(check_skip)
        return TestStatus::Passed;

    miopenStatus_t status = miopenSet4dTensorDescriptorEx(tensorDesc,
                                                          params.dataType,
                                                          params.dimsA[0],
                                                          params.dimsA[1],
                                                          params.dimsA[2],
                                                          params.dimsA[3],
                                                          params.stridesA[0],
                                                          params.stridesA[1],
                                                          params.stridesA[2],
                                                          params.stridesA[3]);
    if(status == miopenStatusSuccess)
        return TestStatus::Passed;

    return TestStatus::Failed;
}

TestStatus SetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                               const TensorParams& params,
                               bool check_skip)
{
    if(params.tensorLayout < miopenTensorNCHW || params.tensorLayout > miopenTensorNDHWC)
        return TestStatus::Skipped;
    if(!params.use_strides &&
       (params.tensorLayout != miopenTensorNCHW && params.tensorLayout != miopenTensorNCDHW))
        return TestStatus::Skipped;
    if(params.stridesA == nullptr && params.use_strides)
        return TestStatus::Skipped;

    if(check_skip)
        return TestStatus::Passed;

    miopenStatus_t status =
        miopenSetTensorDescriptor(tensorDesc,
                                  params.dataType,
                                  params.nbDims,
                                  params.dimsA,
                                  params.use_strides ? params.stridesA : nullptr);
    if(status == miopenStatusSuccess)
        return TestStatus::Passed;

    return TestStatus::Failed;
}

const auto set_tensor_descr_funcs = {Set4dTensorDescriptor,
                                     SetNdTensorDescriptorWithLayout,
                                     Set4dTensorDescriptorEx,
                                     SetTensorDescriptor};

// Get tensor descriptor
TestStatus Get4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc, const TensorParams& params)
{
    if(params.nbDims != 4 || (!params.use_strides && params.tensorLayout != miopenTensorNCHW))
        return TestStatus::Skipped;

    if(params.dimsA == nullptr || (params.stridesA == nullptr && params.use_strides))
        return TestStatus::Failed; // internal error

    miopenStatus_t status;
    miopenDataType_t dataType;
    int dims[4], strides[4];

    status = miopenGet4dTensorDescriptor(tensorDesc,
                                         &dataType,
                                         dims,
                                         dims + 1,
                                         dims + 2,
                                         dims + 3,
                                         strides,
                                         strides + 1,
                                         strides + 2,
                                         strides + 3);
    if(status != miopenStatusSuccess)
        return TestStatus::Failed;

    if(params.dataType != dataType || !CompareLengths(params.dimsA, dims, 4) ||
       (params.use_strides && !CompareLengths(params.stridesA, strides, 4)))
        return TestStatus::Failed;

    return TestStatus::Passed;
}

TestStatus GetTensorDescriptor(miopenTensorDescriptor_t tensorDesc, const TensorParams& params)
{
    if(params.dimsA == nullptr || (params.stridesA == nullptr && params.use_strides))
        return TestStatus::Failed; // internal error

    miopenStatus_t status;
    int size;

    status = miopenGetTensorDescriptorSize(tensorDesc, &size);
    if(status != miopenStatusSuccess || size < 0 || size != params.nbDims)
        return TestStatus::Failed;

    miopenDataType_t dataType;
    std::vector<int> dims(size);
    std::vector<int> strides(size);

    status = miopenGetTensorDescriptor(tensorDesc, &dataType, dims.data(), strides.data());
    if(status != miopenStatusSuccess)
        return TestStatus::Failed;

    if(params.dataType != dataType || !CompareLengths(params.dimsA, dims, size) ||
       (params.use_strides && !CompareLengths(params.stridesA, strides, size)))
        return TestStatus::Failed;

    return TestStatus::Passed;
}

const auto get_tensor_descr_funcs = {Get4dTensorDescriptor, GetTensorDescriptor};

// Generate test data and run tests
void RunValidTestConfigs(bool use_strides, std::function<void(const TestConfig&)> run_test)
{
    const auto datatypes = {miopenHalf, miopenDouble};
    const auto layouts   = {miopenTensorNCHW, miopenTensorNCDHW};
    static int dims[]    = {4, 4, 16, 9, 16};
    static int strides[] = {9216, 2304, 144, 16, 1};
    static_assert(sizeof(dims) == sizeof(strides));
    const auto max_ndims = sizeof(dims) / sizeof(dims[0]);

    for(auto datatype : datatypes)
    {
        for(auto layout : layouts)
        {
            for(int ndims = 1; ndims <= max_ndims; ndims++)
            {
                const TestConfig config = {false,
                                           {datatype,
                                            layout,
                                            ndims,
                                            dims + (max_ndims - ndims),
                                            use_strides ? (strides + (max_ndims - ndims)) : nullptr,
                                            use_strides},
                                           true};
                ASSERT_NO_FATAL_FAILURE(run_test(config));
            }
        }
    }
}

void RunValidTestConfigs(std::function<void(const TestConfig&)> run_test)
{
    ASSERT_NO_FATAL_FAILURE(RunValidTestConfigs(false, run_test));
    ASSERT_NO_FATAL_FAILURE(RunValidTestConfigs(true, run_test));
}

void RunWrongTestConfigs(const TestConfig& valid_config,
                         std::function<void(const TestConfig&)> run_test)
{
#if USE_OUT_OF_RANGE_ENUM
    const auto wrong_datatypes = {static_cast<miopenDataType_t>(miopenHalf - 1),
                                  static_cast<miopenDataType_t>(miopenBFloat8 + 1)};
    const auto wrong_layouts   = {static_cast<miopenTensorLayout_t>(miopenTensorNCHW - 1),
                                static_cast<miopenTensorLayout_t>(miopenTensorNDHWC + 1)};
#endif
    const auto wrong_ndims     = {-1, 0};
    static int wrong_dims[][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, {-1, -1, -1, -1, -1, -1, -1, -1}};

    // tensorDesc = nullptr
    {
        auto config                   = valid_config;
        config.null_tensor_descriptor = true;
        config.valid                  = false;
        ASSERT_NO_FATAL_FAILURE(run_test(config));
    }

#if USE_OUT_OF_RANGE_ENUM
    // wrong data type
    for(auto datatype : wrong_datatypes)
    {
        auto config            = valid_config;
        config.params.dataType = datatype;
        config.valid           = false;
        ASSERT_NO_FATAL_FAILURE(run_test(config));
    }
#endif

#if USE_OUT_OF_RANGE_ENUM
    // wrong layout
    for(auto layout : wrong_layouts)
    {
        auto config                = valid_config;
        config.params.tensorLayout = layout;
        config.valid               = false;
        ASSERT_NO_FATAL_FAILURE(run_test(config));
    }
#endif

    // wrong number of dimensions
    for(auto ndims : wrong_ndims)
    {
        auto config          = valid_config;
        config.params.nbDims = ndims;
        config.valid         = false;
        ASSERT_NO_FATAL_FAILURE(run_test(config));
    }

    // dimsA = nullptr
    {
        auto config         = valid_config;
        config.params.dimsA = nullptr;
        config.valid        = false;
        ASSERT_NO_FATAL_FAILURE(run_test(config));
    }

    // wrong dimensions
    for(auto dims : wrong_dims)
    {
        auto config         = valid_config;
        config.params.dimsA = dims;
        config.valid        = false;
        ASSERT_NO_FATAL_FAILURE(run_test(config));
    }

    if(valid_config.params.use_strides)
    {
        // wrong strides
        for(auto strides : wrong_dims)
        {
            auto config            = valid_config;
            config.params.stridesA = strides;
            config.valid           = false;
            ASSERT_NO_FATAL_FAILURE(run_test(config));
        }
    }
}

void RunWrongTestConfigs(std::function<void(const TestConfig&)> run_test)
{
    RunValidTestConfigs([run_test](const TestConfig& valid_config) {
        RunWrongTestConfigs(valid_config, run_test);
    });
}

} // namespace

class TestTensorApi : public ::testing::Test
{
protected:
    void SetUp() override {}

    static void RunTest(const TestConfig& config)
    {
        for(const auto set_tensor_descr_func : set_tensor_descr_funcs)
        {
            TestStatus test_status;
            miopenStatus_t status;
            miopenTensorDescriptor_t desc = nullptr;

            test_status = set_tensor_descr_func(nullptr, config.params, true);
            if(test_status == TestStatus::Skipped)
                continue;

            if(!config.null_tensor_descriptor)
            {
                status = miopenCreateTensorDescriptor(&desc);
                ASSERT_EQ(status, miopenStatusSuccess);
            }

            test_status = set_tensor_descr_func(desc, config.params, false);
            if(config.valid)
                ASSERT_NE(test_status, TestStatus::Failed);
            else
                ASSERT_NE(test_status, TestStatus::Passed);

            if(config.valid)
            {
                for(const auto get_tensor_descr_func : get_tensor_descr_funcs)
                {
                    test_status = get_tensor_descr_func(desc, config.params);
                    ASSERT_NE(test_status, TestStatus::Failed);
                }
            }

            if(!config.null_tensor_descriptor)
            {
                status = miopenDestroyTensorDescriptor(desc);
                ASSERT_EQ(status, miopenStatusSuccess);
            }
        }
    }
};

TEST_F(TestTensorApi, SetTensor)
{
    RunValidTestConfigs([](const TestConfig& valid_config) { RunTest(valid_config); });
}

TEST_F(TestTensorApi, SetWrongTensor)
{
    RunWrongTestConfigs([](const TestConfig& wrong_config) { RunTest(wrong_config); });
}
