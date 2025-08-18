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

#include <gmock/gmock.h>
#include "gtest_common.hpp"
#include <tensor_util.hpp>
#include "dropout_util.hpp"

#define DROPOUT_DEBUG_CTEST 0
// Workaround for issue #1128
#define DROPOUT_SINGLE_CTEST 1

namespace {

using TestCase = std::tuple<std::vector<int>, float, unsigned long long, bool, int>;

template <class T>
struct verify_forward_dropout
{
    tensor<T> input;
    tensor<T> output;
    std::vector<unsigned char> rsvsp;
    miopen::DropoutDescriptor DropoutDesc;
    miopen::TensorDescriptor noise_shape;
    size_t in_offset;
    size_t out_offset;
    size_t rsvsp_offset;
    bool use_rsvsp;
    typename std::vector<unsigned char>::iterator rsvsp_ptr;

    verify_forward_dropout(const miopen::DropoutDescriptor& pDropoutDesc,
                           const miopen::TensorDescriptor& pNoiseShape,
                           const tensor<T>& pinput,
                           const tensor<T>& poutput,
                           std::vector<unsigned char>& prsvsp,
                           size_t pin_offset,
                           size_t pout_offset,
                           size_t prsvsp_offset,
                           bool puse_rsvsp = true)
    {
        DropoutDesc  = pDropoutDesc;
        noise_shape  = pNoiseShape;
        input        = pinput;
        output       = poutput;
        rsvsp        = prsvsp;
        in_offset    = pin_offset;
        out_offset   = pout_offset;
        rsvsp_offset = prsvsp_offset;
        use_rsvsp    = puse_rsvsp;
        rsvsp_ptr    = prsvsp.begin();
    }

    std::vector<T> cpu() const
    {
        size_t states_size = DropoutDesc.stateSizeInBytes / sizeof(rocrand_state_xorwow);
        auto states_cpu    = std::vector<rocrand_state_xorwow>(states_size);
        InitKernelStateEmulator(states_cpu, DropoutDesc);

        auto out_cpu   = output;
        auto rsvsp_cpu = rsvsp;

        DropoutForwardVerify<T>(get_handle(),
                                DropoutDesc,
                                input.desc,
                                input.data,
                                out_cpu.desc,
                                out_cpu.data,
                                rsvsp_cpu,
                                states_cpu,
                                in_offset,
                                out_offset,
                                rsvsp_offset);

        return out_cpu.data;
    }

    std::vector<T> gpu() const
    {
        auto&& handle  = get_handle();
        auto out_gpu   = output;
        auto rsvsp_dev = handle.Write(rsvsp);
        auto in_dev    = handle.Write(input.data);
        auto out_dev   = handle.Write(output.data);

        DropoutDesc.Dropout(handle,
                            input.desc,
                            input.desc,
                            in_dev.get(),
                            output.desc,
                            out_dev.get(),
                            use_rsvsp ? rsvsp_dev.get() : nullptr,
                            rsvsp.size(),
                            in_offset,
                            out_offset,
                            rsvsp_offset,
                            false /* is_backward */);

        out_gpu.data   = handle.Read<T>(out_dev, output.data.size());
        auto rsvsp_gpu = handle.Read<unsigned char>(rsvsp_dev, rsvsp.size());

        std::copy(rsvsp_gpu.begin(), rsvsp_gpu.end(), rsvsp_ptr);
        return out_gpu.data;
    }

    void fail() const
    {
        std::cout << "Forward Dropout: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_backward_dropout
{
    tensor<T> din;
    tensor<T> dout;
    std::vector<unsigned char> rsvsp;
    miopen::DropoutDescriptor DropoutDesc;

    size_t in_offset;
    size_t out_offset;
    size_t rsvsp_offset;
    bool use_rsvsp;

    verify_backward_dropout(const miopen::DropoutDescriptor& pDropoutDesc,
                            const tensor<T>& pdin,
                            const tensor<T>& pdout,
                            const std::vector<unsigned char>& prsvsp,
                            size_t pin_offset,
                            size_t pout_offset,
                            size_t prsvsp_offset,
                            bool puse_rsvsp = true)
    {
        DropoutDesc  = pDropoutDesc;
        din          = pdin;
        dout         = pdout;
        rsvsp        = prsvsp;
        in_offset    = pin_offset;
        out_offset   = pout_offset;
        rsvsp_offset = prsvsp_offset;
        use_rsvsp    = puse_rsvsp;
    }

    std::vector<T> cpu() const
    {
        auto din_cpu   = din;
        auto rsvsp_cpu = rsvsp;

        DropoutBackwardVerify<T>(DropoutDesc,
                                 dout.desc,
                                 dout.data,
                                 din_cpu.desc,
                                 din_cpu.data,
                                 rsvsp_cpu,
                                 in_offset,
                                 out_offset,
                                 rsvsp_offset);

        return din_cpu.data;
    }

    std::vector<T> gpu() const
    {
        auto&& handle = get_handle();
        auto din_gpu  = din;

        auto din_dev   = handle.Write(din.data);
        auto dout_dev  = handle.Write(dout.data);
        auto rsvsp_dev = handle.Write(rsvsp);

        DropoutDesc.Dropout(handle,
                            din.desc,
                            dout.desc,
                            dout_dev.get(),
                            din.desc,
                            din_dev.get(),
                            use_rsvsp ? rsvsp_dev.get() : nullptr,
                            rsvsp.size(),
                            in_offset,
                            out_offset,
                            rsvsp_offset,
                            true /* is_backward*/);

        din_gpu.data = handle.Read<T>(din_dev, din.data.size());
        return din_gpu.data;
    }

    void fail() const
    {
        std::cout << "Backward Dropout: " << std::endl;
        std::cout << "Doutput tensor: " << dout.desc.ToString() << std::endl;
    }
};

inline auto GenCases()
{
    auto input_dims = get_sub_tensor();

#if DROPOUT_SINGLE_CTEST
    input_dims.resize(1);
#else
#define DROPOUT_LARGE_CTEST 0

    std::set<std::vector<int>> get_inputs_set               = get_inputs(1);

#if DROPOUT_LARGE_CTEST
    std::set<std::vector<int>> get_3d_conv_input_shapes_set = get_3d_conv_input_shapes(1);
    input_dims.insert(input_dims.end(), get_inputs_set.begin(), get_inputs_set.end());
    input_dims.insert(
        input_dims.end(), get_3d_conv_input_shapes_set.begin(), get_3d_conv_input_shapes_set.end());
#else
    auto itr = get_inputs_set.begin();
    for(std::size_t i = 0; i < get_inputs_set.size(); itr++, i++)
        if(i % 6 == 0)
            input_dims.push_back(*itr);

    itr = get_3d_conv_input_shapes_set.begin();
    for(std::size_t i = 0; i < get_3d_conv_input_shapes_set.size(); itr++, i++)
        if(i % 3 == 0)
            input_dims.push_back(*itr);
#endif
#endif
    return testing::Combine(testing::ValuesIn(input_dims),
#if DROPOUT_SINGLE_CTEST
                            testing::Values(float(0.5)),
                            testing::Values(0x0ULL),
                            testing::Values(false),
#else
                            testing::Values(float(0.0), float(0.5), float(1.0)),
                            testing::Values(0x0ULL, 0xFFFFFFFFFFFFFFFFULL),
                            testing::Values(false, true),
#endif
                            testing::Values(0));
}

inline auto GetCases()
{
    static const auto cases = GenCases();
    return cases;
}
} // namespace

template <typename T>
struct DropoutCommon : public testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        std::tie(in_dim, dropout_rate, seed, mask, rng_mode_cmd) = GetParam();
    }

    void Run()
    {
        miopen::DropoutDescriptor DropoutDesc;
        uint64_t max_value       = miopen_type<T>{} == miopenHalf ? 5 : 17;
        auto&& handle            = get_handle();
        auto in                  = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        miopenRNGType_t rng_mode = miopenRNGType_t(rng_mode_cmd);

        size_t stateSizeInBytes = std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth()) *
                                  sizeof(rocrand_state_xorwow);
        size_t reserveSpaceSizeInBytes = in.desc.GetElementSize() * sizeof(bool);
        size_t total_mem =
            2 * (2 * in.desc.GetNumBytes() + reserveSpaceSizeInBytes) + stateSizeInBytes;
        size_t device_mem = handle.GetGlobalMemorySize();
#if !DROPOUT_DEBUG_CTEST
        if(total_mem >= device_mem)
        {
#endif
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
#if !DROPOUT_DEBUG_CTEST
        }
#else
        std::cout << "Input tensor requires " << in.desc.GetElementSize() << " Bytes of memory."
                  << std::endl;
        std::cout << "Output tensor requires " << in.desc.GetElementSize() << " Bytes of memory."
                  << std::endl;
        std::cout << "reserveSpace requires " << reserveSpaceSizeInBytes << " Bytes of memory."
                  << std::endl;
        std::cout << "PRNG state space requires " << stateSizeInBytes << " Bytes of memory."
                  << std::endl;
#endif
        if(total_mem >= device_mem)
        {
            FAIL() << "total_mem >= device_mem";
        }

        auto reserveSpace = std::vector<unsigned char>(in.desc.GetElementSize());
        if(mask)
        {
            for(size_t i = 0; i < in.desc.GetElementSize(); i++)
            {
                reserveSpace[i] =
                    static_cast<unsigned char>(prng::gen_canonical<float>() > dropout_rate);
            }
        }

        DropoutDesc.dropout          = dropout_rate;
        DropoutDesc.stateSizeInBytes = stateSizeInBytes;
        DropoutDesc.seed             = seed;
        DropoutDesc.use_mask         = mask;
        DropoutDesc.rng_mode         = rng_mode;

        auto state_buf      = handle.Create<unsigned char>(stateSizeInBytes);
        DropoutDesc.pstates = state_buf.get();
        DropoutDesc.InitPRNGState(
            handle, DropoutDesc.pstates, DropoutDesc.stateSizeInBytes, DropoutDesc.seed);
#if DROPOUT_DEBUG_CTEST
        std::cout <<
#if MIOPEN_BACKEND_OPENCL
            "Use OpenCL backend."
#elif MIOPEN_BACKEND_HIP
            "Use HIP backend."
#endif
                  << std::endl;
#endif

        auto out = tensor<T>{in_dim};

        VerifyForwardDropout(DropoutDesc, in.desc, in, out, reserveSpace, 0, 0, 0);

        auto dout = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        auto din  = tensor<T>{in_dim};
        VerifyBackwardDropout(DropoutDesc, din, dout, reserveSpace, 0, 0, 0);

        if(!mask)
        {
            VerifyForwardDropout(DropoutDesc, in.desc, in, out, reserveSpace, 0, 0, 0, false);
            VerifyBackwardDropout(DropoutDesc, din, dout, reserveSpace, 0, 0, 0, false);
        }
    }

private:
    void VerifyForwardDropout(const miopen::DropoutDescriptor& pDropoutDesc,
                              const miopen::TensorDescriptor& pNoiseShape,
                              const tensor<T>& pinput,
                              const tensor<T>& poutput,
                              std::vector<unsigned char>& prsvsp,
                              size_t pin_offset,
                              size_t pout_offset,
                              size_t prsvsp_offset,
                              bool puse_rsvsp = true)
    {
        verify_forward_dropout<T> forward_dropout{pDropoutDesc,
                                                  pNoiseShape,
                                                  pinput,
                                                  poutput,
                                                  prsvsp,
                                                  pin_offset,
                                                  pout_offset,
                                                  prsvsp_offset,
                                                  puse_rsvsp};
        CompareResults(forward_dropout);
    }

    void VerifyBackwardDropout(const miopen::DropoutDescriptor& pDropoutDesc,
                               const tensor<T>& pdin,
                               const tensor<T>& pdout,
                               const std::vector<unsigned char>& prsvsp,
                               size_t pin_offset,
                               size_t pout_offset,
                               size_t prsvsp_offset,
                               bool puse_rsvsp = true)
    {
        verify_backward_dropout<T> backward_dropout(
            pDropoutDesc, pdin, pdout, prsvsp, pin_offset, pout_offset, prsvsp_offset, puse_rsvsp);
        CompareResults(backward_dropout);
    }

    template <class TDirection>
    void CompareResults(const TDirection& direction)
    {
        const std::vector<T> cpu_data = direction.cpu();
        const std::vector<T> gpu_data = direction.gpu();

        // taken from the original test
        double tolerance = 80;

        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        double error     = miopen::rms_range(cpu_data, gpu_data);

        if(error > threshold)
        {
            direction.fail();
        }

        ASSERT_LE(error, threshold) << "dropout_rate: " << dropout_rate << std::endl
                                    << "seed: " << seed << std::endl
                                    << "mask: " << mask << std::endl
                                    << "rng_mode_cmd: " << rng_mode_cmd << std::endl;
    }

private:
    std::vector<int> in_dim;
    float dropout_rate      = 0.0f;
    unsigned long long seed = 0;
    bool mask               = false;
    int rng_mode_cmd        = 0;
};

using GPU_Dropout_FP32 = DropoutCommon<float>;
using GPU_Dropout_FP16 = DropoutCommon<half_float::half>;

TEST_P(GPU_Dropout_FP32, TestFloat) { this->Run(); }
TEST_P(GPU_Dropout_FP16, TestFloat16) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Dropout_FP32, GetCases());
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Dropout_FP16, GetCases());

struct TestCaseValues
{
    const std::string msg;
    const miopenStatus_t status;
    const miopen::TensorDescriptor xDesc;
    ConstData_t x;
    const miopen::TensorDescriptor yDesc;
    Data_t y;
    miopen::TensorDescriptor noise_shape;
    Data_t reserveSpace{nullptr};
    size_t reserveSpaceSizeInBytes{0};
};

static void check_vals(miopen::DropoutDescriptor& dd, TestCaseValues& vals)
{
    EXPECT_THROW(
        {
            try
            {
                dd.Dropout(get_handle(),
                           vals.noise_shape,
                           vals.xDesc,
                           vals.x,
                           vals.yDesc,
                           vals.y,
                           vals.reserveSpace,
                           vals.reserveSpaceSizeInBytes,
                           0,
                           0,
                           0,
                           false);
            }
            catch(const miopen::Exception& e)
            {
                if(vals.msg.length() > 0)
                {
                    EXPECT_THAT(e.message, ::testing::EndsWith(vals.msg));
                    EXPECT_THAT(e.what(), ::testing::EndsWith(vals.msg));
                }

                EXPECT_EQ(e.status, vals.status);
                throw;
            }
        },
        miopen::Exception);
}

TEST(CPU_dropout_NONE, test_dropout_miopen_throw)
{
    miopen::DropoutDescriptor dd;

    miopen::TensorDescriptor xDesc_int32_5(miopenInt32, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1});
    miopen::TensorDescriptor yDesc_int32_5{miopenInt32, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    miopen::TensorDescriptor xDesc_int32_6{miopenInt32, {2, 2, 2, 2, 2, 2}, {1, 1, 2, 3, 4, 5}};
    miopen::TensorDescriptor yDesc_int32_6{miopenInt32, {1, 1, 2, 3, 4, 5}, {1, 1, 2, 3, 4, 5}};

    miopen::TensorDescriptor xDesc_int32_3(miopenInt32, {2, 2, 2}, {1, 1, 1});
    miopen::TensorDescriptor yDesc_int32_3(miopenInt32, {1, 1, 1}, {1, 1, 1});

    miopen::TensorDescriptor xDesc_int8_5(miopenInt8, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1});

    miopen::TensorDescriptor noise_shape_2(miopenInt32, {9, 9}, {9, 9});
    miopen::TensorDescriptor noise_shape_5(miopenInt32, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1});

    int dumb_array[] = {1};
    ConstData_t x{dumb_array};
    Data_t y{dumb_array};
    Data_t reserveSpace{dumb_array};

    TestCaseValues testvals[] = {
        // clang-format off
        //exception message                                         exception status          xDesc          x        yDesc          y        noise_shape
        {"",                                                        miopenStatusBadParm,      xDesc_int32_5, nullptr, yDesc_int32_5, nullptr, noise_shape_2}, // x or y is nullptr
        {"",                                                        miopenStatusBadParm,      xDesc_int32_5, x,       yDesc_int32_5, nullptr, noise_shape_2}, // x or y is nullptr
        {"",                                                        miopenStatusBadParm,      xDesc_int32_5, nullptr, yDesc_int32_5, y,       noise_shape_2}, // x or y is nullptr
        {"Input/Output dimension does not match",                   miopenStatusUnknownError, xDesc_int32_3, x,       yDesc_int32_5, y,       noise_shape_2}, // xDesc and yDesc with different size
        {"Only support 1D to 5D tensors",                           miopenStatusUnknownError, xDesc_int32_6, x,       yDesc_int32_6, y,       noise_shape_2}, // xDesc size > 5
        {"Input/Output element size does not match",                miopenStatusUnknownError, xDesc_int32_3, x,       yDesc_int32_3, y,       noise_shape_2}, // xDesc and yDesc element size differs
        {"Only support dropout with regular noise shape currently", miopenStatusUnknownError, xDesc_int32_5, x,       yDesc_int32_5, y,       noise_shape_2}, // xDesc/yDesc and noise_shape element size differs
        {"Input/Output datatype does not match",                    miopenStatusUnknownError, xDesc_int8_5,  x,       yDesc_int32_5, y,       noise_shape_5}, // xDesc and yDesc diff type
        {"Invalid dropout rate",                                    miopenStatusUnknownError, xDesc_int32_5, x,       yDesc_int32_5, y,       noise_shape_5} // incorrect dropout
        // clang-format on
    };

    dd.dropout = 10.0;
    for(auto& testval : testvals)
    {
        check_vals(dd, testval);
    }

    dd.dropout  = 0.0;
    dd.use_mask = true;
    {
        // clang-format off
        //                      exception message                exception status          xDesc          x  yDesc          y  noise_shape
        TestCaseValues testval{"Insufficient reservespace size", miopenStatusUnknownError, xDesc_int32_5, x, yDesc_int32_5, y, noise_shape_5};
        // clang-format on
        check_vals(dd, testval);
    }

    dd.use_mask = false;
    {
        miopen::TensorDescriptor xDesc_test(miopenInt32, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1});
        miopen::TensorDescriptor yDesc_test(miopenInt32, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1});
        miopen::TensorDescriptor noise_shape_test(miopenInt32, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1});
        // clang-format off
        //                      exception message                exception status          xDesc          x  yDesc          y  noise_shape
        TestCaseValues testval{"Insufficient reservespace size", miopenStatusUnknownError, xDesc_test, x, yDesc_test, y, noise_shape_test, reserveSpace, sizeof(dumb_array)};
        // clang-format on
        check_vals(dd, testval);
    }

    {
        const size_t reservedSpace = get_handle().GetGlobalMemorySize();
        // clang-format off
        //                      exception message                                                      exception status          xDesc          x  yDesc          y  noise_shape
        TestCaseValues testval{"Memory required by dropout forward configs exceeds GPU memory range.", miopenStatusUnknownError, xDesc_int32_5, x, yDesc_int32_5, y, noise_shape_5, reserveSpace, reservedSpace};
        // clang-format on
        check_vals(dd, testval);
    }

    {
        // clang-format off
        //                      exception message                             exception status          xDesc          x  yDesc          y  noise_shape
        TestCaseValues testval{"Insufficient state size for parallel PRNG",   miopenStatusUnknownError, xDesc_int32_5, x, yDesc_int32_5, y, noise_shape_5, reserveSpace, sizeof(dumb_array)};
        // clang-format on
        check_vals(dd, testval);
    }
}
