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

#include "get_handle.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/softmax.hpp>

#define NEGATIVE_CUTOFF_VAL_FP32 (-1e20)
#define NEGATIVE_CUTOFF_VAL_FP16 (-1e4)

namespace {

template <typename T>
T logaddexp(T x, T y, T neg_inf)
{
    T a = std::max(x, y);
    T b = std::min(x, y);
    T c = b - a;

    return c <= neg_inf ? std::max(a, neg_inf) : std::max(T(a + log(T(1) + exp(b - a))), neg_inf);
}

struct TestCase
{
    std::vector<size_t> in_dim;
    std::vector<float> scale;
    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;
};

template <typename T>
void AddTestCasesForDifferentScales(std::vector<TestCase>& test_cases,
                                    const std::vector<size_t>& in_dim,
                                    int algo,
                                    int mode,
                                    const std::vector<std::vector<float>>& scales)
{
    /// \todo Apply mix-precision in softmax to improve the stability of fp16
    if(miopen_type<T>{} == miopenHalf)
    {
        if((in_dim[1] * in_dim[2] * in_dim[3] >= 2048) && mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
            return;

        if(in_dim[1] >= 96 && in_dim[2] >= 14 && in_dim[3] >= 14 && algo == MIOPEN_SOFTMAX_FAST)
            return;
    }

    for(const auto& scale : scales)
    {
        TestCase& test_case = test_cases.emplace_back();

        test_case.in_dim = in_dim;
        test_case.algo   = static_cast<miopenSoftmaxAlgorithm_t>(algo);
        test_case.mode   = static_cast<miopenSoftmaxMode_t>(mode);
        test_case.scale  = scale;
    }
}

template <typename T>
std::vector<TestCase> GenCases()
{
    int batch_factor = 0;

    std::set<std::vector<size_t>> in_dim_set = get_inputs<size_t>(batch_factor);

    /// \todo Resolve this workaround. Random failure on Jenkins (ROCm3.0):
    /// --float --input-dim 1 480 128 256 --algorithm 2 --mode 1 --scales 1 0 --tolerance 8000
    /// FAILED: inf
    in_dim_set.erase({1, 480, 128, 256});

    /// \todo Resolve this workaround. Regular failures on Radeon VII, ROCm 3.3:
    /// --float --input-dim 1 1 8 8 --algorithm 0 --mode 1 --scales 1 0 --tolerance 8000
    /// FAILED: -nan
    in_dim_set.erase({1, 1, 8, 8});
    in_dim_set.erase({1, 1, 14, 14});
    in_dim_set.erase({1, 1, 27, 27});
    in_dim_set.erase({1, 32, 7, 7});
    in_dim_set.erase({1, 32, 8, 8});

    std::vector<int> algos                 = {0, 1, 2};
    std::vector<int> modes                 = {0, 1};
    std::vector<std::vector<float>> scales = {{1.0f, 0.0f}, {0.5f, 0.5f}};

    std::vector<TestCase> test_cases;

    for(const auto& in_dim : in_dim_set)
        for(const int algo : algos)
            for(const int mode : modes)
            {
                AddTestCasesForDifferentScales<T>(test_cases, in_dim, algo, mode, scales);
            }

    return test_cases;
}

template <typename T>
auto GetCases()
{
    static const auto cases = testing::ValuesIn(GenCases<T>());
    return cases;
}

} // namespace

template <typename T>
struct SoftmaxCommon : public testing::TestWithParam<TestCase>
{
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        const TestCase& test_case = GetParam();

        uint64_t max_value =
            miopen_type<T>{} == miopenHalf ? (test_case.algo == MIOPEN_SOFTMAX_LOG ? 3 : 5) : 17;

        input            = tensor<T>{test_case.in_dim}.generate(tensor_elem_gen_integer{max_value});
        size_t total_mem = 2 * input.desc.GetNumBytes(); // estimate based on backward pass
        size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;

            GTEST_SKIP();
        }

        output = tensor<T>{test_case.in_dim}.generate(tensor_elem_gen_integer{max_value});

        std::vector<T> tensorCpuDataForward = GetForwardCpu();
        std::vector<T> tensorGpuDataForward = GetForwardGpu();

        // check forward results
        CompareResults(tensorGpuDataForward, tensorCpuDataForward);

        dout   = tensor<T>{test_case.in_dim}.generate([&](int n, int c, int h, int w) {
            T x      = input(n, c, h, w);
            double y = (877 * n + 547 * c + 701 * h + 1049 * w + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });
        dinput = tensor<T>{test_case.in_dim}.generate(tensor_elem_gen_integer{max_value});

        std::vector<T> tensorCpuDataBackward = GetBackwardCpu();
        std::vector<T> tensorGpuDataBackward = GetBackwardGpu();

        // check backward results
        CompareResults(tensorGpuDataBackward, tensorCpuDataBackward);
    }

    std::vector<T> GetForwardCpu() const
    {
        const TestCase& test_case = GetParam();

        auto out = output;

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

        int in_nstr, in_cstr, in_hstr;
        std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = miopen::tien<4>(input.desc.GetStrides());

        int out_nstr, out_cstr, out_hstr;
        std::tie(out_nstr, out_cstr, out_hstr, std::ignore) =
            miopen::tien<4>(out.desc.GetStrides());

        float alpha = test_case.scale[0];
        float beta  = test_case.scale[1];

        if(test_case.mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        {
            par_ford(in_n)([&](int o) {
                if(test_case.algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                            alpha * (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]) /
                                     sum) +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                    });
                }
                else
                {
                    T max_c = std::numeric_limits<T>::lowest();
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        max_c = std::max(max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });

                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = input.desc.GetType() == miopenHalf
                                             ? NEGATIVE_CUTOFF_VAL_FP16
                                             : NEGATIVE_CUTOFF_VAL_FP32;
                        double sum     = neg_inf;
                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum = logaddexp(
                                double(input[o * in_nstr + w * in_cstr + i * in_hstr + j] - max_c),
                                sum,
                                neg_inf);
                        });

                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha * (input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                         max_c - sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                            max_c);
                        });

                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha *
                                    (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                              max_c) /
                                     sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                }
            });
        }
        else
        {
            par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
                if(test_case.algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    ford(in_c)([&](int w) {
                        sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });
                    ford(in_c)([&](int w) {
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                            alpha * (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]) /
                                     sum) +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                    });
                }
                else
                {
                    T max_c = std::numeric_limits<T>::lowest();
                    ford(in_c)([&](int w) {
                        max_c = std::max(max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });

                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = input.desc.GetType() == miopenHalf
                                             ? NEGATIVE_CUTOFF_VAL_FP16
                                             : NEGATIVE_CUTOFF_VAL_FP32;
                        double sum     = neg_inf;
                        ford(in_c)([&](int w) {
                            sum = logaddexp(
                                double(input[o * in_nstr + w * in_cstr + i * in_hstr + j] - max_c),
                                sum,
                                neg_inf);
                        });

                        ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha * (input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                         max_c - sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        ford(in_c)([&](int w) {
                            sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                            max_c);
                        });

                        ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha *
                                    (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                              max_c) /
                                     sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                }
            });
        }
        return out.data;
    }

    std::vector<T> GetForwardGpu() const
    {
        const TestCase& test_case = GetParam();
        auto&& handle             = get_handle();
        // auto out      = output;

        auto in_dev  = handle.Write(input.data);
        auto out_dev = handle.Write(output.data);

        miopen::SoftmaxForward(handle,
                               &test_case.scale[0],
                               &test_case.scale[1],
                               input.desc,
                               in_dev.get(),
                               output.desc,
                               out_dev.get(),
                               test_case.algo,
                               test_case.mode);

        return handle.Read<T>(out_dev, output.data.size());
    }

    std::vector<T> GetBackwardCpu() const
    {
        const TestCase& test_case = GetParam();

        auto din = dinput;

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(din.desc.GetLengths());

        int in_nstr, in_cstr, in_hstr;
        std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = miopen::tien<4>(din.desc.GetStrides());

        int out_nstr, out_cstr, out_hstr;
        std::tie(out_nstr, out_cstr, out_hstr, std::ignore) =
            miopen::tien<4>(dout.desc.GetStrides());

        float alpha = test_case.scale[0];
        float beta  = test_case.scale[1];

        if(test_case.mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        {
            par_ford(in_n)([&](int o) {
                double sum = 0;
                ford(in_c, in_h, in_w)([&](int c, int i, int j) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        sum += dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                    }
                    else
                    {
                        sum += output[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                               dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                    }
                });

                ford(in_c, in_h, in_w)([&](int c, int i, int j) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            T(alpha * (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] -
                                       sum * std::exp(output[o * out_nstr + c * out_cstr +
                                                             i * out_hstr + j])) +
                              beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j]);
                    }
                    else
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            alpha * (output[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                                     (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] - sum)) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j];
                    }
                });
            });
        }
        else
        {
            par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
                double sum = 0;
                ford(in_c)([&](int c) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        sum += dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                    }
                    else
                    {
                        sum += output[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                               dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                    }
                });

                ford(in_c)([&](int c) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            alpha *
                                (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] -
                                 sum *
                                     std::exp(
                                         output[o * out_nstr + c * out_cstr + i * out_hstr + j])) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j];
                    }
                    else
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            alpha * (output[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                                     (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] - sum)) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j];
                    }
                });
            });
        }
        return din.data;
    }

    std::vector<T> GetBackwardGpu() const
    {
        const TestCase& test_case = GetParam();

        auto&& handle = get_handle();
        // auto din      = dinput;

        auto din_dev  = handle.Write(dinput.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(output.data);

        miopen::SoftmaxBackward(handle,
                                &test_case.scale[0],
                                output.desc,
                                out_dev.get(),
                                dout.desc,
                                dout_dev.get(),
                                &test_case.scale[1],
                                dinput.desc,
                                din_dev.get(),
                                test_case.algo,
                                test_case.mode);

        return handle.Read<T>(din_dev, dinput.data.size());
    }

    void CompareResults(const std::vector<T>& tensorGPUData, const std::vector<T>& tensorCPUData)
    {
        const TestCase& test_case = GetParam();

        // taken from the original c test
        double tolerance = 8000;

        if(std::is_same_v<T, half_float::half>)
        {
            tolerance = 80;
        }

        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        double error     = miopen::rms_range(tensorCPUData, tensorGPUData);

        ASSERT_LE(error, threshold)
            << "Tensor Dims: " << test_case.in_dim[0] << ", " << test_case.in_dim[1] << ", "
            << test_case.in_dim[2] << ", " << test_case.in_dim[3] << ", "
            << "Alpha / Beta: " << test_case.scale[0] << ", " << test_case.scale[1]
            << ". Algo: " << test_case.algo << ". Mode: " << test_case.mode << std::endl;
    }

private:
    tensor<T> input;
    tensor<T> output;

    tensor<T> dinput;
    tensor<T> dout;
};

using GPU_Softmax_FP32 = SoftmaxCommon<float>;
using GPU_Softmax_FP16 = SoftmaxCommon<half_float::half>;

TEST_P(GPU_Softmax_FP32, TestFloat) { this->Run(); }
TEST_P(GPU_Softmax_FP16, TestFloat16) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_FP32, GetCases<float>());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_FP16, GetCases<half_float::half>());
