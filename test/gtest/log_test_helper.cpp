/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include "log_test_helper.hpp"

#include <miopen/config.h>
#include <cstdlib>

#if MIOPEN_BACKEND_OPENCL
#define BKEND "OpenCL"
#elif MIOPEN_BACKEND_HIP
#define BKEND "HIP"
#endif

// Copy of struct that is in miopen.
// This is for testing purpose only.
enum class ConvDirection
{
    Fwd = 1,
    Bwd = 2,
    WrW = 4
};

const std::string logConv =
    "MIOpen(" BKEND
    "): Command [LogCmdConvolution] ./bin/MIOpenDriver conv -n 128 -c 3 -H 32 -W 32 -k "
    "64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1";
const std::string logFindConv =
    "MIOpen(" BKEND
    "): Command [LogCmdFindConvolution] ./bin/MIOpenDriver conv -n 128 -c 3 -H 32 -W 32 "
    "-k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1";

const std::string envConv = "MIOPEN_ENABLE_LOGGING_CMD";

static void setEnvironmentVariable(const std::string& name, const std::string& value)
{
    int ret = 0;

#ifdef _WIN32
    std::string env_var(name + "=" + value);
    ret = _putenv(env_var.c_str());
#else
    ret = setenv(name.c_str(), value.c_str(), 1);
#endif
    EXPECT_EQ(ret, 0);
}

static void unSetEnvironmentVariable(const std::string& name)
{
    int ret = 0;
#ifdef _WIN32
    std::string empty_env_var(name + "=");
    ret = _putenv(empty_env_var.c_str());
#else
    ret = unsetenv(name.c_str());
#endif
    EXPECT_EQ(ret, 0);
}

// Captures the std::cerr buffer and store it to a string.
struct CerrRedirect
{
public:
    CerrRedirect()
    {
        // Store the current stream buffer object and set
        // stringstream as a new stream buffer.
        old_stream_buf = std::cerr.rdbuf(str_buffer.rdbuf());
        EXPECT_TRUE(old_stream_buf != nullptr);
    }

    std::string getString() { return str_buffer.str(); }

    ~CerrRedirect()
    {
        // reset the old stream buffer back
        std::cerr.rdbuf(old_stream_buf);
    }

private:
    std::streambuf* old_stream_buf;
    std::stringstream str_buffer;
};

struct Tensor
{
    miopenTensorDescriptor_t desc{};
    size_t data_size;
#if MIOPEN_BACKEND_OPENCL
    cl_mem data;
#elif MIOPEN_BACKEND_HIP
    void* data;
#endif

    Tensor(int n, int c, int h, int w)
    {
        EXPECT_EQ(miopenCreateTensorDescriptor(&desc), 0);
        EXPECT_EQ(miopenSet4dTensorDescriptor(desc, miopenFloat, n, c, h, w), 0);
        data_size = n * c * h * w * sizeof(float);
#if MIOPEN_BACKEND_OPENCL
        cl_command_queue q{};
        miopenHandle_t handle{};
        miopenCreate(&handle);
        miopenGetStream(handle, &q);
        cl_context ctx;
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
        data = clCreateBuffer(ctx, CL_MEM_READ_WRITE, data_size, nullptr, nullptr);
#elif MIOPEN_BACKEND_HIP
        EXPECT_EQ(hipMalloc(&data, data_size), hipSuccess);
#endif
    }

    ~Tensor()
    {
        miopenDestroyTensorDescriptor(desc);
#if MIOPEN_BACKEND_OPENCL
        clReleaseMemObject(data);
#elif MIOPEN_BACKEND_HIP
        hipFree(data);
#endif
    }
};

struct Conv
{
    Tensor input;
    Tensor output;
    Tensor weights;
    miopenConvolutionDescriptor_t convDesc;

    Conv() : input(128, 3, 32, 32), output(128, 64, 32, 32), weights(64, 3, 3, 3)
    {
        EXPECT_EQ(miopenCreateConvolutionDescriptor(&convDesc), 0);
        EXPECT_EQ(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 1, 1, 1, 1, 1, 1),
                  0);
    }
    ~Conv() { miopenDestroyConvolutionDescriptor(convDesc); }
};

static bool isSubStr(const std::string& str, const std::string& sub_str)
{
    return str.find(sub_str) != std::string::npos;
}

void TestLogFun(std::function<void(const miopenTensorDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const miopenConvolutionDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const ConvDirection&,
                                   bool)> const& func,
                std::string env_var,
                std::string sub_str,
                bool set_env)
{
    // start capturing std::cerr
    CerrRedirect capture_cerr;
    // prepare tensor and convolution descriptors
    Conv test_conv_log;
    if(set_env)
        setEnvironmentVariable(env_var, "1");
    else
        unSetEnvironmentVariable(env_var);

    func(test_conv_log.input.desc,
         test_conv_log.weights.desc,
         test_conv_log.convDesc,
         test_conv_log.output.desc,
         ConvDirection::Fwd,
         false);

    // get the captured string
    std::string str = capture_cerr.getString();
    // now do the assertions
    if(set_env)
        ASSERT_TRUE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
    else
        ASSERT_FALSE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
}
