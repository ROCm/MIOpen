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
#include "log.hpp"
#include "tensor_util.hpp"
#include "get_handle.hpp"

#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/fusion_plan.hpp>
#include "../random.hpp"

namespace env = miopen::env;

#if MIOPEN_BACKEND_OPENCL
#define BKEND "OpenCL"
#elif MIOPEN_BACKEND_HIP
#define BKEND "HIP"
#endif

#ifdef _WIN32
#define MDEXE "MIOpenDriver.exe"
#else
#define MDEXE "./bin/MIOpenDriver"
#endif

const std::string logConv =
    "MIOpen(" BKEND "): Command [LogCmdConvolution] " MDEXE " conv -n 128 -c 3 -H 32 -W 32 -k "
    "64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1";
const std::string logFindConv =
    "MIOpen(" BKEND "): Command [LogCmdFindConvolution] " MDEXE " conv -n 128 -c 3 -H 32 -W 32 "
    "-k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1";

const std::string logFusionConvBiasActiv =
    "MIOpen(" BKEND "): Command [LogCmdFusion] " MDEXE " CBAInfer -F 4 -n 128 -c 3 -H 32 "
    "-W 32 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1";

const std::string logBnormActiv = "MIOpen(" BKEND "): Command [LogCmdFusion] " MDEXE
                                  " CBAInfer -F 2 -n 64 -c 64 -H 56 -W 56 -m 1";

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
    Tensor bias;
    miopenConvolutionDescriptor_t convDesc;

    Conv()
        : input(128, 3, 32, 32), output(128, 64, 32, 32), weights(64, 3, 3, 3), bias(64, 1, 32, 1)
    {
        EXPECT_EQ(miopenCreateConvolutionDescriptor(&convDesc), 0);
        EXPECT_EQ(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 1, 1, 1, 1, 1, 1),
                  0);
    }
    ~Conv() { miopenDestroyConvolutionDescriptor(convDesc); }
};

struct CreateCBAFusionPlan
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopen::OperatorArgs op_args;

    void CBAPlan()
    {
        Conv conv_obj;

        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, conv_obj.input.desc);

        auto convoOp = std::make_shared<miopen::ConvForwardOpDescriptor>(
            miopen::deref(conv_obj.convDesc), miopen::deref(conv_obj.weights.desc));
        auto biasOp =
            std::make_shared<miopen::BiasFusionOpDescriptor>(miopen::deref(conv_obj.weights.desc));
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(miopenActivationRELU);
        miopen::deref(fusePlanDesc).AddOp(convoOp);
        miopen::deref(fusePlanDesc).AddOp(biasOp);
        miopen::deref(fusePlanDesc).AddOp(activOp);
    }
};

template <typename T>
struct CreateBNormFusionPlan
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopen::OperatorArgs op_args;

    miopen::TensorDescriptor bn_desc;
    miopen::ActivationDescriptor activ_desc;
    miopenBatchNormMode_t bn_mode = miopenBNSpatial;
    tensor<T> input;
    tensor<T> output;
    tensor<T> scale;
    tensor<T> shift;
    tensor<T> estMean;
    tensor<T> estVariance;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev;
    miopen::Allocator::ManageDataPtr estMean_dev;
    miopen::Allocator::ManageDataPtr estVariance_dev;
    miopenActivationMode_t activ_mode = miopenActivationRELU;
    const float alpha                 = static_cast<float>(1.0f);
    const float beta                  = static_cast<float>(0);
    const float activ_alpha           = static_cast<double>(0.5f);
    const float activ_beta            = static_cast<double>(0.5f);
    const float activ_gamma           = static_cast<double>(0.5f);
    double epsilon                    = 1.0e-5;
    std::vector<int> input_lens       = {64, 64, 56, 56};

    void Init()
    {

        input              = tensor<T>{input_lens};
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopenDeriveBNTensorDescriptor(&derivedBnDesc, &input.desc, bn_mode);
        scale       = tensor<T>{input_lens};
        shift       = tensor<T>{input_lens};
        estMean     = tensor<T>{input_lens};
        estVariance = tensor<T>{input_lens};

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input.generate(gen_value);
        scale.generate(gen_value);
        shift.generate(gen_value);
        estMean.generate(gen_value);
        auto gen_var = [](auto...) { return static_cast<T>(1e-2 * (prng::gen_0_to_B(100) + 1)); };
        estVariance.generate(gen_var);
        activ_desc    = {activ_mode, activ_alpha, activ_beta, activ_gamma};
        output        = tensor<T>{input_lens};
        auto&& handle = get_handle();
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        in_dev          = handle.Write(input.data);
        scale_dev       = handle.Write(scale.data);
        shift_dev       = handle.Write(shift.data);
        estMean_dev     = handle.Write(estMean.data);
        estVariance_dev = handle.Write(estVariance.data);
        out_dev         = handle.Write(output.data);
    }

    void BNormActivation()
    {
        Init();
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, &input.desc);

        activ_desc = {activ_mode, activ_alpha, activ_beta, activ_gamma};

        auto bnOp =
            std::make_shared<miopen::BatchNormInferenceFusionOpDescriptor>(bn_mode, bn_desc);
        EXPECT_EQ(miopen::deref(fusePlanDesc).AddOp(bnOp), miopenStatusSuccess);
        bnOp->SetArgs(op_args,
                      &alpha,
                      &beta,
                      scale_dev.get(),
                      shift_dev.get(),
                      estMean_dev.get(),
                      estVariance_dev.get(),
                      epsilon);
        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_desc.GetMode());
        EXPECT_EQ(miopen::deref(fusePlanDesc).AddOp(activOp), miopenStatusSuccess);
        activOp->SetArgs(op_args, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    }
};

static bool isSubStr(const std::string& str, const std::string& sub_str)
{
    return str.find(sub_str) != std::string::npos;
}

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_CMD)

void TestLogFun(std::function<void(const miopenTensorDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const miopenConvolutionDescriptor_t&,
                                   const miopenTensorDescriptor_t&,
                                   const miopen::debug::ConvDirection&,
                                   bool)> const& func,
                std::string sub_str,
                bool set_env)
{
    CerrRedirect capture_cerr;
    Conv test_conv_log;
    if(set_env)
        env::update(MIOPEN_ENABLE_LOGGING_CMD, true);
    else
        env::clear(MIOPEN_ENABLE_LOGGING_CMD);

    func(test_conv_log.input.desc,
         test_conv_log.weights.desc,
         test_conv_log.convDesc,
         test_conv_log.output.desc,
         miopen::debug::ConvDirection::Fwd,
         false);

    std::string str = capture_cerr.getString();
    if(set_env)
        ASSERT_TRUE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
    else
        ASSERT_FALSE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
}

void TestLogCmdCBAFusion(std::function<void(const miopenFusionPlanDescriptor_t)> const& func,
                         std::string sub_str,
                         bool set_env)
{
    CerrRedirect capture_cerr;

    if(set_env)
        env::update(MIOPEN_ENABLE_LOGGING_CMD, true);
    else
        env::clear(MIOPEN_ENABLE_LOGGING_CMD);

    CreateCBAFusionPlan fp_cba_create;
    fp_cba_create.CBAPlan();

    func(fp_cba_create.fusePlanDesc);

    std::string str = capture_cerr.getString();

    if(set_env)
        ASSERT_TRUE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
    else
        ASSERT_FALSE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
}

void TestLogCmdBNormFusion(std::function<void(const miopenFusionPlanDescriptor_t)> const& func,
                           std::string sub_str,
                           bool set_env)
{
    CerrRedirect capture_cerr;

    if(set_env)
        env::update(MIOPEN_ENABLE_LOGGING_CMD, true);
    else
        env::clear(MIOPEN_ENABLE_LOGGING_CMD);

    CreateBNormFusionPlan<float> fp_bnorm_create;
    fp_bnorm_create.BNormActivation();

    func(fp_bnorm_create.fusePlanDesc);

    std::string str = capture_cerr.getString();

    if(set_env)
        ASSERT_TRUE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
    else
        ASSERT_FALSE(isSubStr(str, sub_str)) << "str     : " << str << "str_sub : " << sub_str;
}
